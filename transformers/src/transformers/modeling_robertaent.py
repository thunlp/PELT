# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """


import logging
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from .configuration_roberta import RobertaConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu, BertPooler, BertLayer, BertEncoder
from .modeling_utils import create_position_ids_from_input_ids
import numpy as np
import torch.nn.functional as F


logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = 1 #config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        return super().forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
        )

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.

        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class EntityEmbeddings(nn.Module):
    def __init__(self, config):
        super(EntityEmbeddings, self).__init__()
        self.config = config
        try:
            save_on_gpu = config.save_on_gpu
        except:
            save_on_gpu = False

        if save_on_gpu:
            self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.hidden_size)#, padding_idx=0)
            if config.entity_emb_size != config.hidden_size:
                self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,  position_ids=None, entity_ids = None, entity_embeddings = None,
    ):

        token_type_ids = torch.zeros((position_ids.shape[0], position_ids.shape[1]), dtype=torch.long, device=position_ids.device)
        
        if entity_embeddings is None:
            entity_embeddings = self.entity_embeddings(entity_ids)
            if self.config.entity_emb_size != self.config.hidden_size:
                entity_embeddings = self.entity_embedding_dense(entity_embeddings)
        
        if position_ids.dim() == 3:
            position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
            position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
            position_embeddings = position_embeddings * position_embedding_mask
            position_embeddings = torch.sum(position_embeddings, dim=-2)
            position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=1e-7)
        else:
            position_embeddings = self.position_embeddings(position_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

ROBERTA_START_DOCSTRING = r""""""

ROBERTA_INPUTS_DOCSTRING = r""""""

class RobertaEntModel(BertPreTrainedModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = RobertaEmbeddings(config)

        self.entity_embeddings = EntityEmbeddings(config)
        self.entity_embeddings.position_embeddings.weight = self.embeddings.position_embeddings.weight

        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
            
    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        entity_ids = None,
        entity_position_ids = None,
        entity_attention_mask = None,
        entity_embeddings = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # return_tuple = return_tuple if return_tuple is not None else self.config.use_return_tuple

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)


        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        # if self.config.is_decoder and encoder_hidden_states is not None:
        #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        #     if encoder_attention_mask is None:
        #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        #     encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        # else:
        #     encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=1)

        if entity_ids is not None or entity_embeddings is not None:
            entity_embedding_output = self.entity_embeddings(position_ids=entity_position_ids, entity_ids=entity_ids, entity_embeddings=entity_embeddings)
            embedding_output = torch.cat([embedding_output, entity_embedding_output], dim=1)


        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, embedding_output.size()[:-1], device)


        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_extended_attention_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)


        return (sequence_output, pooled_output) + encoder_outputs[1:]



class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x




class RobertaEntForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaEntModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        entity_ids=None,
        entity_embeddings=None,
        entity_attention_mask=None,
        entity_position_ids=None,
        ht_position=None,
    ): 
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            entity_ids=entity_ids,
            entity_embeddings=entity_embeddings,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )
        sequence_output = outputs[0]#[:, : input_ids.size(1), :]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)





# class RobertaEntForMarkerSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.roberta = RobertaEntModel(config)
#         self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         entity_ids=None,
#         entity_embeddings=None,
#         entity_attention_mask=None,
#         entity_position_ids=None,
#         ht_position=None,
#     ): 
#         outputs = self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             entity_ids=entity_ids,
#             entity_embeddings=entity_embeddings,
#             entity_attention_mask=entity_attention_mask,
#             entity_position_ids=entity_position_ids,
#         )
#         sequence_output = outputs[0]#[:, : input_ids.size(1), :]
#         bsz = sequence_output.shape[0]
#         h_rep = sequence_output[torch.arange(bsz), ht_position[:,0]]
#         t_rep = sequence_output[torch.arange(bsz), ht_position[:,1]]

#         ht_rep = torch.cat([h_rep, t_rep], dim=-1)
#         ht_rep = self.dropout(ht_rep)

#         logits = self.classifier(ht_rep)

#         outputs = (logits,) + outputs[2:]
#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs  # (loss), logits, (hidden_states), (attentions)





class RobertaEntForQuestionAnswering(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaEntModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        entity_ids=None,
        entity_embeddings=None,
        entity_attention_mask=None,
        entity_position_ids=None,
        start_positions=None,
        end_positions=None,
    ): 
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            entity_ids=entity_ids,
            entity_embeddings=entity_embeddings,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )
        sequence_output = outputs[0][:, : input_ids.size(1), :]
        
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            start_losses = [loss_fct(start_logits, _start_positions) for _start_positions in torch.unbind(start_positions, dim=1)]
            end_losses = [loss_fct(end_logits, _end_positions) for _end_positions in torch.unbind(end_positions, dim=1)]

            start_loss = sum(start_losses)
            end_loss = sum(end_losses)

            loss = torch.mean(start_loss+end_loss) / 2  

            outputs = (loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)



class RobertaEntForAppendSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaEntModel(config)
        self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
        # self.classifier = nn.Linear(config.hidden_size*4, config.num_labels)
        self.dropout = nn.Dropout(config.output_dropout_prob)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        entity_embeddings=None,
        entity_position_ids=None,
    ): 
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_embeddings=entity_embeddings,
            entity_position_ids=entity_position_ids,
        )
        bsz, seq_len = input_ids.shape
        sequence_output = outputs[0] 
        bsz = sequence_output.shape[0]
        h_rep = sequence_output[:, seq_len]
        t_rep = sequence_output[:, seq_len+1]

        # h2_rep = sequence_output[:, seq_len+2]
        # t2_rep = sequence_output[:, seq_len+3]

        ht_rep = torch.cat([h_rep, t_rep], dim=-1)
        # ht_rep = torch.cat([h_rep, h2_rep, t_rep, t2_rep], dim=-1)
        ht_rep = self.dropout(ht_rep)

        logits = self.classifier(ht_rep)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)






class RobertaEntLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.ent_decoder = nn.Linear(config.hidden_size, config.entity_vocab_size, bias=False)
        self.ent_bias = nn.Parameter(torch.zeros(config.entity_vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.ent_decoder.bias = self.ent_bias

    def forward(self, features, masked_token_indexes=None):

        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.ent_decoder(x)

        return x

class RobertaEntForLinkPrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaEntModel(config)
        self.lm_head = RobertaEntLMHead(config)
        self.lm_head.ent_decoder.weight = self.roberta.entity_embeddings.entity_embeddings.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        entity_ids=None,
        entity_position_ids=None,
        mask_position=None,
    ): 
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
        )
        bsz, seq_len = input_ids.shape
        sequence_output = outputs[0] 
        states = sequence_output[torch.arange(bsz), mask_position]

        logits = self.lm_head(states)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)






class RobertaEntForTyping(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaEntModel(config)
        self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        entity_ids=None,
        entity_embeddings=None,
        entity_attention_mask=None,
        entity_position_ids=None,
        m_position=None,
    ): 
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            entity_ids=entity_ids,
            entity_embeddings=entity_embeddings,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )
        sequence_output = outputs[0]#[:, : input_ids.size(1), :]
        bsz = sequence_output.shape[0]
        h_rep = sequence_output[torch.arange(bsz), m_position[:,0]]
        t_rep = sequence_output[torch.arange(bsz), m_position[:,1]]

        ht_rep = torch.cat([h_rep, t_rep], dim=-1)
        ht_rep = self.dropout(ht_rep)

        logits = self.classifier(ht_rep)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)








class RobertaEntForMarkerSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaEntModel(config)
        self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        entity_ids=None,
        entity_embeddings=None,
        entity_attention_mask=None,
        entity_position_ids=None,
        ht_position=None,
    ): 
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            entity_ids=entity_ids,
            entity_embeddings=entity_embeddings,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )
        sequence_output = outputs[0]#[:, : input_ids.size(1), :]
        bsz = sequence_output.shape[0]
        h_rep = sequence_output[torch.arange(bsz), ht_position[:,0]]
        t_rep = sequence_output[torch.arange(bsz), ht_position[:,1]]

        ht_rep = torch.cat([h_rep, t_rep], dim=-1)
        ht_rep = self.dropout(ht_rep)

        logits = self.classifier(ht_rep)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)





class RobertaEntForSpanClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaEntModel(config)
        self.classifer = nn.Linear(config.hidden_size*2, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        entity_ids=None,
        entity_embeddings=None,
        entity_attention_mask=None,
        entity_position_ids=None,
        placeholder_idxs=None,
        answer_positions=None,
    ): 
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            entity_ids=entity_ids,
            entity_embeddings=entity_embeddings,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )
        sequence_output = outputs[0]
        bsz = sequence_output.shape[0]

        entity_emb = sequence_output[torch.arange(bsz).unsqueeze(-1), answer_positions]
        placeholder_emb = sequence_output[torch.arange(bsz), placeholder_idxs]
        feature_vector = torch.cat([placeholder_emb.unsqueeze(1).expand_as(entity_emb), entity_emb], dim=-1)
        feature_vector = self.dropout(feature_vector)
        logits = self.classifer(feature_vector)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  



class RobertaEntForSpanClassificationLinear(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaEntModel(config)
        self.pre_linear = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.classifer = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        entity_ids=None,
        entity_embeddings=None,
        entity_attention_mask=None,
        entity_position_ids=None,
        placeholder_idxs=None,
        answer_positions=None,
    ): 
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            entity_ids=entity_ids,
            entity_embeddings=entity_embeddings,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )
        sequence_output = outputs[0]
        bsz = sequence_output.shape[0]

        entity_emb = sequence_output[torch.arange(bsz).unsqueeze(-1), answer_positions]
        placeholder_emb = sequence_output[torch.arange(bsz), placeholder_idxs]
        feature_vector = torch.cat([placeholder_emb.unsqueeze(1).expand_as(entity_emb), entity_emb], dim=-1)
        feature_vector = self.dropout(feature_vector)
        feature_vector = self.dropout(F.gelu(self.pre_linear(feature_vector)))
        logits = self.classifer(feature_vector)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  




class RobertaEntForSpanClassificationNoQ(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaEntModel(config)
        self.classifer = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        entity_ids=None,
        entity_embeddings=None,
        entity_attention_mask=None,
        entity_position_ids=None,
        placeholder_idxs=None,
        answer_positions=None,
    ): 
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            entity_ids=entity_ids,
            entity_embeddings=entity_embeddings,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )
        sequence_output = outputs[0]
        bsz = sequence_output.shape[0]

        entity_emb = sequence_output[torch.arange(bsz).unsqueeze(-1), answer_positions]
        # placeholder_emb = sequence_output[torch.arange(bsz), placeholder_idxs]
        # feature_vector = torch.cat([placeholder_emb.unsqueeze(1).expand_as(entity_emb), entity_emb], dim=-1)
        feature_vector = self.dropout(entity_emb)
        # feature_vector = self.dropout(F.gelu(self.pre_linear(feature_vector)))
        logits = self.classifer(feature_vector)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  

class RobertaEntForSpanQA(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaEntModel(config)
        self.pre_linear = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.classifer = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        entity_ids=None,
        entity_embeddings=None,
        entity_attention_mask=None,
        entity_position_ids=None,
        placeholder_idxs=None,
        answer_positions=None,
    ): 
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            entity_ids=entity_ids,
            entity_embeddings=entity_embeddings,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )
        sequence_output = outputs[0]
        bsz = sequence_output.shape[0]

        entity_emb = sequence_output[torch.arange(bsz).unsqueeze(-1), answer_positions]
        placeholder_emb = sequence_output[torch.arange(bsz), placeholder_idxs]
        feature_vector = torch.cat([placeholder_emb.unsqueeze(1).expand_as(entity_emb), entity_emb], dim=-1)
        feature_vector = self.dropout(feature_vector)
        feature_vector = self.dropout(F.gelu(self.pre_linear(feature_vector)))
        logits = self.classifer(feature_vector).squeeze(-1)

        outputs = (logits,) + outputs[2:]
        if labels is not None:

            # loss_fct = CrossEntropyLoss(ignore_index=-1)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = [loss_fct(logits, _labels) for _labels in torch.unbind(labels, dim=1)]
            loss = sum(loss)
            loss = torch.mean(loss)

            outputs = (loss,) + outputs

        return outputs  
        

class RobertaEntForSpanQANoQ(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaEntModel(config)
        # self.pre_linear = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.classifer = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        entity_ids=None,
        entity_embeddings=None,
        entity_attention_mask=None,
        entity_position_ids=None,
        placeholder_idxs=None,
        answer_positions=None,
    ): 
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            entity_ids=entity_ids,
            entity_embeddings=entity_embeddings,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )
        sequence_output = outputs[0]
        bsz = sequence_output.shape[0]

        entity_emb = sequence_output[torch.arange(bsz).unsqueeze(-1), answer_positions]
        # placeholder_emb = sequence_output[torch.arange(bsz), placeholder_idxs]
        # feature_vector = torch.cat([placeholder_emb.unsqueeze(1).expand_as(entity_emb), entity_emb], dim=-1)
        feature_vector = self.dropout(entity_emb)
        # feature_vector = self.dropout(F.gelu(self.pre_linear(feature_vector)))
        logits = self.classifer(feature_vector).squeeze(-1)

        outputs = (logits,) + outputs[2:]
        if labels is not None:

            # loss_fct = CrossEntropyLoss(ignore_index=-1)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = [loss_fct(logits, _labels) for _labels in torch.unbind(labels, dim=1)]
            loss = sum(loss)
            loss = torch.mean(loss)

            outputs = (loss,) + outputs

        return outputs  
        


class RobertaEntForTriviaQuestionAnswering(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaEntModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def or_softmax_cross_entropy_loss_one_doc(self, logits, target, ignore_index=-1, dim=-1):
        """loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf"""
        # assert logits.ndim == 2
        # assert target.ndim == 2
        # assert logits.size(0) == target.size(0)

        # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
        # here, the numerator is the sum of a few potential targets, where some of them is the correct answer
        bsz = logits.shape[0]

        # compute a target mask
        target_mask = target == ignore_index
        # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
        masked_target = target * (1 - target_mask.long())
        # gather logits
        gathered_logits = logits.gather(dim=dim, index=masked_target)
        # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
        gathered_logits[target_mask] = -10000.0#float('-inf')

        # each batch is one example
        gathered_logits = gathered_logits.view(bsz, -1)
        logits = logits.view(bsz, -1)

        # numerator = log(sum(exp(gathered logits)))
        log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)
        # denominator = log(sum(exp(logits)))
        log_norm = torch.logsumexp(logits, dim=dim, keepdim=False)

        # compute the loss
        loss = -(log_score - log_norm)

        # some of the examples might have a loss of `inf` when `target` is all `ignore_index`.
        # remove those from the loss before computing the sum. Use sum instead of mean because
        # it is easier to compute
        # return loss[~torch.isinf(loss)].sum()
        return loss.mean()#loss.sum() / len(loss)    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        answer_masks=None,
        entity_ids=None,
        entity_embeddings=None,
        entity_attention_mask=None,
        entity_position_ids=None,
    ):
        bsz, max_segment, seq_len = input_ids.shape

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-2), attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        entity_embeddings = entity_embeddings.view(-1, entity_embeddings.size(-2), entity_embeddings.size(-1)) if entity_embeddings is not None else None
        entity_position_ids = entity_position_ids.view(-1, entity_position_ids.size(-2), entity_position_ids.size(-1)) if entity_position_ids is not None else None


        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # entity_ids=entity_ids,
            entity_embeddings=entity_embeddings,
            # entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )

        sequence_output = outputs[0]
        sequence_output = sequence_output[:, :seq_len]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        start_logits = start_logits.view(bsz, max_segment, -1) # (bsz, max_segment, seq_length)
        end_logits = end_logits.view(bsz, max_segment, -1) # (bsz, max_segment, seq_length)


        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:

            start_loss = self.or_softmax_cross_entropy_loss_one_doc(start_logits, start_positions, ignore_index=-1)
            end_loss = self.or_softmax_cross_entropy_loss_one_doc(end_logits, end_positions, ignore_index=-1)

            total_loss = (start_loss + end_loss) / 2

            outputs = (total_loss,) + outputs

        return outputs  
