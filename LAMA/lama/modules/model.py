import logging
import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertIntermediate,
    BertOutput,
    BertPooler,
    BertSelfOutput,
)
from transformers.modeling_roberta import RobertaEmbeddings, gelu
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
BertLayerNorm = torch.nn.LayerNorm

logger = logging.getLogger(__name__)


class LukeConfig(BertConfig):
    def __init__(
        self, vocab_size: int, entity_vocab_size: int, bert_model_name: str, entity_emb_size: int = None, **kwargs
    ):
        super(LukeConfig, self).__init__(vocab_size, **kwargs)

        self.entity_vocab_size = entity_vocab_size
        self.bert_model_name = bert_model_name
        if entity_emb_size is None:
            self.entity_emb_size = self.hidden_size
        else:
            self.entity_emb_size = entity_emb_size


class EntityEmbeddings(nn.Module):
    def __init__(self, config: LukeConfig):
        super(EntityEmbeddings, self).__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, entity_ids: torch.LongTensor, position_ids: torch.LongTensor, token_type_ids: torch.LongTensor = None
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = torch.sum(position_embeddings, dim=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LukeModel(nn.Module):
    def __init__(self, config: LukeConfig):
        super(LukeModel, self).__init__()

        self.config = config

        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
            self.embeddings = RobertaEmbeddings(config)
            self.embeddings.token_type_embeddings.requires_grad = False
        else:
            self.embeddings = BertEmbeddings(config)
        self.entity_embeddings = EntityEmbeddings(config)

    def forward(
        self,
        word_ids: torch.LongTensor,
        word_attention_mask: torch.LongTensor,
        word_segment_ids: torch.LongTensor = None,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        entity_segment_ids: torch.LongTensor = None,
        entity_attention_mask: torch.LongTensor = None,
    ):
        word_seq_size = word_ids.size(1)

        embedding_output = self.embeddings(word_ids, word_segment_ids)

        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)
        if entity_ids is not None:
            entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
            embedding_output = torch.cat([embedding_output, entity_embedding_output], dim=1)

        encoder_outputs = self.encoder(embedding_output, attention_mask, [None] * self.config.num_hidden_layers)
        sequence_output = encoder_outputs[0]
        word_sequence_output = sequence_output[:, :word_seq_size, :]
        pooled_output = self.pooler(sequence_output)

        if entity_ids is not None:
            entity_sequence_output = sequence_output[:, word_seq_size:, :]
            return (word_sequence_output, entity_sequence_output, pooled_output,) + encoder_outputs[1:]
        else:
            return (word_sequence_output, pooled_output,) + encoder_outputs[1:]

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            if module.embedding_dim == 1:  # embedding for bias parameters
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_bert_weights(self, state_dict: Dict[str, torch.Tensor]):
        state_dict = state_dict.copy()
        for key in list(state_dict.keys()):
            new_key = key.replace("gamma", "weight").replace("beta", "bias")
            if new_key.startswith("roberta."):
                new_key = new_key[8:]
            elif new_key.startswith("bert."):
                new_key = new_key[5:]

            if key != new_key:
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self, prefix="")
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    self.__class__.__name__, sorted(unexpected_keys)
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(self.__class__.__name__, "\n\t".join(error_msgs))
            )

    def _compute_extended_attention_mask(
        self, word_attention_mask: torch.LongTensor, entity_attention_mask: torch.LongTensor
    ):
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


class LukeEntityAwareAttentionModel(LukeModel):
    def __init__(self, config):
        super(LukeEntityAwareAttentionModel, self).__init__(config)
        self.config = config
        self.encoder = EntityAwareEncoder(config)

    def forward(
        self,
        word_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_attention_mask,
        word_segment_ids=None,
        entity_segment_ids=None,
    ):
        word_embeddings = self.embeddings(word_ids, word_segment_ids)
        entity_embeddings = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

        return self.encoder(word_embeddings, entity_embeddings, attention_mask)

    def load_state_dict(self, state_dict, *args, **kwargs):
        new_state_dict = state_dict.copy()

        for num in range(self.config.num_hidden_layers):
            for attr_name in ("weight", "bias"):
                if f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}" not in state_dict:
                    new_state_dict[f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}"] = state_dict[
                        f"encoder.layer.{num}.attention.self.query.{attr_name}"
                    ]

                if f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}" not in state_dict:
                    new_state_dict[f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}"] = state_dict[
                        f"encoder.layer.{num}.attention.self.query.{attr_name}"
                    ]

                if f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}" not in state_dict:
                    new_state_dict[f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}"] = state_dict[
                        f"encoder.layer.{num}.attention.self.query.{attr_name}"
                    ]

        kwargs["strict"] = False
        super(LukeEntityAwareAttentionModel, self).load_state_dict(new_state_dict, *args, **kwargs)


class EntityAwareSelfAttention(nn.Module):
    def __init__(self, config):
        super(EntityAwareSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_size = word_hidden_states.size(1)

        w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
        w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
        e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
        e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

        key_layer = self.transpose_for_scores(self.key(torch.cat([word_hidden_states, entity_hidden_states], dim=1)))

        w2w_key_layer = key_layer[:, :, :word_size, :]
        e2w_key_layer = key_layer[:, :, :word_size, :]
        w2e_key_layer = key_layer[:, :, word_size:, :]
        e2e_key_layer = key_layer[:, :, word_size:, :]

        w2w_attention_scores = torch.matmul(w2w_query_layer, w2w_key_layer.transpose(-1, -2))
        w2e_attention_scores = torch.matmul(w2e_query_layer, w2e_key_layer.transpose(-1, -2))
        e2w_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2))
        e2e_attention_scores = torch.matmul(e2e_query_layer, e2e_key_layer.transpose(-1, -2))

        word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=3)
        entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=3)
        attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.transpose_for_scores(
            self.value(torch.cat([word_hidden_states, entity_hidden_states], dim=1))
        )
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer[:, :word_size, :], context_layer[:, word_size:, :]


class EntityAwareAttention(nn.Module):
    def __init__(self, config):
        super(EntityAwareAttention, self).__init__()
        self.self = EntityAwareSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_self_output, entity_self_output = self.self(word_hidden_states, entity_hidden_states, attention_mask)
        hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)
        self_output = torch.cat([word_self_output, entity_self_output], dim=1)
        output = self.output(self_output, hidden_states)
        return output[:, : word_hidden_states.size(1), :], output[:, word_hidden_states.size(1) :, :]


class EntityAwareLayer(nn.Module):
    def __init__(self, config):
        super(EntityAwareLayer, self).__init__()

        self.attention = EntityAwareAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_attention_output, entity_attention_output = self.attention(
            word_hidden_states, entity_hidden_states, attention_mask
        )
        attention_output = torch.cat([word_attention_output, entity_attention_output], dim=1)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output[:, : word_hidden_states.size(1), :], layer_output[:, word_hidden_states.size(1) :, :]


class EntityAwareEncoder(nn.Module):
    def __init__(self, config):
        super(EntityAwareEncoder, self).__init__()
        self.layer = nn.ModuleList([EntityAwareLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        for layer_module in self.layer:
            word_hidden_states, entity_hidden_states = layer_module(
                word_hidden_states, entity_hidden_states, attention_mask
            )
        return word_hidden_states, 




class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        # self.decoder.bias = self.bias

    def forward(self, features, masked_token_indexes=None):

        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class LukeForMaskedLM(LukeEntityAwareAttentionModel):
    def __init__(self, config):
        super(LukeForMaskedLM, self).__init__(config)
        self.lm_head = RobertaLMHead(config)

        self.lm_head.decoder.weight = self.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids,
        attention_mask,
        entity_ids=None,
        entity_attention_mask=None,
        entity_position_ids=None,
        masked_lm_labels=None,
    ):
        
        outputs = super(LukeForMaskedLM, self).forward(
            word_ids=input_ids,
            word_attention_mask=attention_mask,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )

        sequence_output = outputs[0][:, : input_ids.size(1), :]


        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


# import logging
# import math
# from typing import Dict

# import torch
# import torch.nn.functional as F
# from torch import nn
# from transformers.modeling_bert import (
#     BertConfig,
#     BertEmbeddings,
#     BertEncoder,
#     BertIntermediate,
#     BertOutput,
#     BertPooler,
#     BertSelfOutput,
# )
# from transformers.modeling_roberta import RobertaEmbeddings, gelu
# from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

# BertLayerNorm = torch.nn.LayerNorm

# logger = logging.getLogger(__name__)


# class LukeConfig(BertConfig):
#     def __init__(
#         self, vocab_size: int, entity_vocab_size: int, bert_model_name: str, entity_emb_size: int = None, **kwargs
#     ):
#         super(LukeConfig, self).__init__(vocab_size, **kwargs)

#         self.entity_vocab_size = entity_vocab_size
#         self.bert_model_name = bert_model_name
#         if entity_emb_size is None:
#             self.entity_emb_size = self.hidden_size
#         else:
#             self.entity_emb_size = entity_emb_size


# class EntityEmbeddings(nn.Module):
#     def __init__(self, config: LukeConfig, save_on_gpu=False):
#         super(EntityEmbeddings, self).__init__()
#         self.config = config
#         self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)
#         if config.entity_emb_size != config.hidden_size:
#             self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

#         self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
#         self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

#         self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(
#         self,  position_ids: torch.LongTensor, token_type_ids: torch.LongTensor = None,
#         entity_ids = None,
#         entity_embeddings = None,
#     ):
#         if token_type_ids is None:
#             token_type_ids = torch.zeros((position_ids.shape[0], position_ids.shape[1]), dtype=torch.int64).to(position_ids)

#         if entity_embeddings is None:
#             entity_embeddings = self.entity_embeddings(entity_ids)
#             if self.config.entity_emb_size != self.config.hidden_size:
#                 entity_embeddings = self.entity_embedding_dense(entity_embeddings)

#         position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
#         position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
#         position_embeddings = position_embeddings * position_embedding_mask
#         position_embeddings = torch.sum(position_embeddings, dim=-2)
#         position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=1e-7)

#         token_type_embeddings = self.token_type_embeddings(token_type_ids)

#         embeddings = entity_embeddings + position_embeddings + token_type_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)

#         return embeddings


# class LukeModel(nn.Module):
#     def __init__(self, config: LukeConfig):
#         super(LukeModel, self).__init__()

#         self.config = config

#         self.encoder = BertEncoder(config)
#         self.pooler = BertPooler(config)

#         if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
#             self.embeddings = RobertaEmbeddings(config)
#             self.embeddings.token_type_embeddings.requires_grad = False
#         else:
#             self.embeddings = BertEmbeddings(config)
#         self.entity_embeddings = EntityEmbeddings(config, save_on_gpu=False)

#         self.entity_embeddings.position_embeddings.weight = self.embeddings.position_embeddings.weight
#         # self.entity_embeddings.LayerNorm.weight = self.embeddings.LayerNorm.weight
#         # self.entity_embeddings.LayerNorm.bias = self.embeddings.LayerNorm.bias
#         # self.entity_embeddings.token_type_embeddings.weight = self.embeddings.token_type_embeddings.weight



#     def forward(
#         self,
#         word_ids: torch.LongTensor,
#         word_segment_ids: torch.LongTensor,
#         word_attention_mask: torch.LongTensor,
#         entity_ids: torch.LongTensor = None,
#         entity_position_ids: torch.LongTensor = None,
#         entity_segment_ids: torch.LongTensor = None,
#         entity_attention_mask: torch.LongTensor = None,
#         entity_embeddings = None,
#     ):
#         word_seq_size = word_ids.size(1)

#         embedding_output = self.embeddings(word_ids, word_segment_ids)

#         attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)
#         if entity_ids is not None or entity_embeddings is not None:
#             entity_embedding_output = self.entity_embeddings(entity_position_ids, entity_segment_ids, entity_ids=entity_ids, entity_embeddings=entity_embeddings)
#             embedding_output = torch.cat([embedding_output, entity_embedding_output], dim=1)

#         encoder_outputs = self.encoder(embedding_output, attention_mask, [None] * self.config.num_hidden_layers)
#         sequence_output = encoder_outputs[0]
#         word_sequence_output = sequence_output[:, :word_seq_size, :]
#         pooled_output = self.pooler(sequence_output)

#         if entity_ids is not None:
#             entity_sequence_output = sequence_output[:, word_seq_size:, :]
#             return (word_sequence_output, entity_sequence_output, pooled_output,) + encoder_outputs[1:]
#         else:
#             return (word_sequence_output, pooled_output,) + encoder_outputs[1:]

#     def init_weights(self, module: nn.Module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#         elif isinstance(module, nn.Embedding):
#             if module.embedding_dim == 1:  # embedding for bias parameters
#                 module.weight.data.zero_()
#             else:
#                 module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#         elif isinstance(module, BertLayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()

#     def load_bert_weights(self, state_dict: Dict[str, torch.Tensor]):
#         state_dict = state_dict.copy()
#         for key in list(state_dict.keys()):
#             new_key = key.replace("gamma", "weight").replace("beta", "bias")
#             if new_key.startswith("roberta."):
#                 new_key = new_key[8:]
#             elif new_key.startswith("bert."):
#                 new_key = new_key[5:]

#             if key != new_key:
#                 state_dict[new_key] = state_dict[key]
#                 del state_dict[key]

#         missing_keys = []
#         unexpected_keys = []
#         error_msgs = []

#         metadata = getattr(state_dict, "_metadata", None)
#         state_dict = state_dict.copy()
#         if metadata is not None:
#             state_dict._metadata = metadata

#         def load(module, prefix=""):
#             local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
#             module._load_from_state_dict(
#                 state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
#             )
#             for name, child in module._modules.items():
#                 if child is not None:
#                     load(child, prefix + name + ".")

#         load(self, prefix="")
#         if len(unexpected_keys) > 0:
#             logger.info(
#                 "Weights from pretrained model not used in {}: {}".format(
#                     self.__class__.__name__, sorted(unexpected_keys)
#                 )
#             )
#         if len(error_msgs) > 0:
#             raise RuntimeError(
#                 "Error(s) in loading state_dict for {}:\n\t{}".format(self.__class__.__name__, "\n\t".join(error_msgs))
#             )

#     def _compute_extended_attention_mask(
#         self, word_attention_mask: torch.LongTensor, entity_attention_mask: torch.LongTensor
#     ):
#         attention_mask = word_attention_mask
#         if entity_attention_mask is not None:
#             attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=1)
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

#         return extended_attention_mask


# class LukeEntityAwareAttentionModel(LukeModel):
#     def __init__(self, config):
#         super(LukeEntityAwareAttentionModel, self).__init__(config)
#         self.config = config
#         self.encoder = EntityAwareEncoder(config)

#     def forward(
#         self,
#         word_ids,
#         word_attention_mask=None,
#         word_segment_ids=None,
#         entity_ids=None,
#         entity_embeddings=None,
#         entity_attention_mask=None,
#         entity_segment_ids=None,
#         entity_position_ids=None,
#     ):
#         word_embeddings = self.embeddings(word_ids, word_segment_ids)
#         entity_embeddings = self.entity_embeddings(entity_position_ids, entity_segment_ids, entity_ids=entity_ids, entity_embeddings=entity_embeddings)
#         attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

#         return self.encoder(word_embeddings, entity_embeddings, attention_mask)

#     def load_state_dict(self, state_dict, *args, **kwargs):
#         new_state_dict = state_dict.copy()

#         for num in range(self.config.num_hidden_layers):
#             for attr_name in ("weight", "bias"):
#                 if f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}" not in state_dict:
#                     new_state_dict[f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}"] = state_dict[
#                         f"encoder.layer.{num}.attention.self.query.{attr_name}"
#                     ]
#                 if f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}" not in state_dict:
#                     new_state_dict[f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}"] = state_dict[
#                         f"encoder.layer.{num}.attention.self.query.{attr_name}"
#                     ]
#                 if f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}" not in state_dict:
#                     new_state_dict[f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}"] = state_dict[
#                         f"encoder.layer.{num}.attention.self.query.{attr_name}"
#                     ]

#         if 'entity_embeddings.position_embeddings.weight' not in state_dict:
#             print ('copying position_embeddings')
#             new_state_dict['entity_embeddings.position_embeddings.weight'] = state_dict['embeddings.position_embeddings.weight']

#         if 'entity_embeddings.token_type_embeddings.weight' not in state_dict:
#             print ('copying token_type_embeddings')
#             new_state_dict['entity_embeddings.token_type_embeddings.weight'] = state_dict['embeddings.token_type_embeddings.weight']

#         if 'entity_embeddings.LayerNorm.weight' not in state_dict:
#             print ('copying entity_embeddings.LayerNorm')
#             new_state_dict['entity_embeddings.LayerNorm.weight'] = state_dict['embeddings.LayerNorm.weight']
#             new_state_dict['entity_embeddings.LayerNorm.bias'] = state_dict['embeddings.LayerNorm.bias']



#         kwargs["strict"] = False
#         super(LukeEntityAwareAttentionModel, self).load_state_dict(new_state_dict, *args, **kwargs)


# class EntityAwareSelfAttention(nn.Module):
#     def __init__(self, config):
#         super(EntityAwareSelfAttention, self).__init__()

#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         return x.view(*new_x_shape).permute(0, 2, 1, 3)

#     def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
#         word_size = word_hidden_states.size(1)

#         w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
#         w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
#         e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
#         e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

#         key_layer = self.transpose_for_scores(self.key(torch.cat([word_hidden_states, entity_hidden_states], dim=1)))

#         w2w_key_layer = key_layer[:, :, :word_size, :]
#         e2w_key_layer = key_layer[:, :, :word_size, :]
#         w2e_key_layer = key_layer[:, :, word_size:, :]
#         e2e_key_layer = key_layer[:, :, word_size:, :]

#         w2w_attention_scores = torch.matmul(w2w_query_layer, w2w_key_layer.transpose(-1, -2))
#         w2e_attention_scores = torch.matmul(w2e_query_layer, w2e_key_layer.transpose(-1, -2))
#         e2w_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2))
#         e2e_attention_scores = torch.matmul(e2e_query_layer, e2e_key_layer.transpose(-1, -2))

#         word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=3)
#         entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=3)
#         attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2)

#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         attention_scores = attention_scores + attention_mask

#         attention_probs = F.softmax(attention_scores, dim=-1)
#         attention_probs = self.dropout(attention_probs)

#         value_layer = self.transpose_for_scores(
#             self.value(torch.cat([word_hidden_states, entity_hidden_states], dim=1))
#         )
#         context_layer = torch.matmul(attention_probs, value_layer)

#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)

#         return context_layer[:, :word_size, :], context_layer[:, word_size:, :]


# class EntityAwareAttention(nn.Module):
#     def __init__(self, config):
#         super(EntityAwareAttention, self).__init__()
#         self.self = EntityAwareSelfAttention(config)
#         self.output = BertSelfOutput(config)

#     def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
#         word_self_output, entity_self_output = self.self(word_hidden_states, entity_hidden_states, attention_mask)
#         hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)
#         self_output = torch.cat([word_self_output, entity_self_output], dim=1)
#         output = self.output(self_output, hidden_states)
#         return output[:, : word_hidden_states.size(1), :], output[:, word_hidden_states.size(1) :, :]


# class EntityAwareLayer(nn.Module):
#     def __init__(self, config):
#         super(EntityAwareLayer, self).__init__()

#         self.attention = EntityAwareAttention(config)
#         self.intermediate = BertIntermediate(config)
#         self.output = BertOutput(config)

#     def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
#         word_attention_output, entity_attention_output = self.attention(
#             word_hidden_states, entity_hidden_states, attention_mask
#         )
#         attention_output = torch.cat([word_attention_output, entity_attention_output], dim=1)
#         intermediate_output = self.intermediate(attention_output)
#         layer_output = self.output(intermediate_output, attention_output)

#         return layer_output[:, : word_hidden_states.size(1), :], layer_output[:, word_hidden_states.size(1) :, :]


# class EntityAwareEncoder(nn.Module):
#     def __init__(self, config):
#         super(EntityAwareEncoder, self).__init__()
#         self.layer = nn.ModuleList([EntityAwareLayer(config) for _ in range(config.num_hidden_layers)])

#     def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
#         for layer_module in self.layer:
#             word_hidden_states, entity_hidden_states = layer_module(
#                 word_hidden_states, entity_hidden_states, attention_mask
#             )
#         return word_hidden_states, entity_hidden_states




# class LukeForReadingComprehension(LukeEntityAwareAttentionModel):
#     def __init__(self, args):
#         super(LukeForReadingComprehension, self).__init__(args.model_config)

#         self.qa_outputs = nn.Linear(self.config.hidden_size, 2)
#         self.apply(self.init_weights)

#     def forward(
#         self,
#         word_ids,
#         word_segment_ids,
#         word_attention_mask,
#         entity_ids,
#         entity_position_ids,
#         entity_segment_ids,
#         entity_attention_mask,
#         start_positions=None,
#         end_positions=None,
#     ):
#         encoder_outputs = super(LukeForReadingComprehension, self).forward(
#             word_ids,
#             word_segment_ids,
#             word_attention_mask,
#             entity_ids,
#             entity_position_ids,
#             entity_segment_ids,
#             entity_attention_mask,
#         )

#         word_hidden_states = encoder_outputs[0][:, : word_ids.size(1), :]
#         logits = self.qa_outputs(word_hidden_states)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)

#         if start_positions is not None and end_positions is not None:
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)

#             ignored_index = start_logits.size(1)
#             start_positions.clamp_(0, ignored_index)
#             end_positions.clamp_(0, ignored_index)

#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2
#             outputs = (total_loss,)
#         else:
#             outputs = tuple()

#         return outputs + (start_logits, end_logits,)


# class RobertaClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""

#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

#     def forward(self, features, **kwargs):
#         x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x


# class LukeForSequenceClassification(LukeEntityAwareAttentionModel):
#     def __init__(self, config):
#         super(LukeForSequenceClassification, self).__init__(config)
#         self.num_labels = self.config.num_labels

#         self.classifier = RobertaClassificationHead(self.config)
#         self.apply(self.init_weights)

#     def forward(
#         self,
#         input_ids,
#         attention_mask,
#         token_type_ids=None,
#         entity_ids=None,
#         entity_embeddings=None,
#         entity_attention_mask=None,
#         entity_segment_ids=None,
#         entity_position_ids=None,
#         labels=None,
#     ):
#         outputs = super(LukeForSequenceClassification, self).forward(
#             input_ids,
#             word_attention_mask=attention_mask,
#             word_segment_ids=token_type_ids,
#             entity_ids=entity_ids,
#             entity_embeddings=entity_embeddings,
#             entity_attention_mask=entity_attention_mask,
#             entity_segment_ids=entity_segment_ids,
#             entity_position_ids=entity_position_ids,
#         )

#         sequence_output = outputs[0][:, : input_ids.size(1), :]

#         logits = self.classifier(sequence_output)

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


# class RobertaClassificationHeadType(nn.Module):
#     """Head for sentence-level classification tasks."""

#     def __init__(self, config):
#         super(RobertaClassificationHeadType, self).__init__()
#         self.dense0 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.activation = nn.Tanh()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels, False)

#     def forward(self, features, pos, **kwargs):
#         #x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = (pos.unsqueeze(2).float()*features).sum(1)
#         x = self.dense0(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x

# class LukeForTyping(LukeEntityAwareAttentionModel):
#     def __init__(self, config):
#         super(LukeForTyping, self).__init__(config)
#         self.num_labels = self.config.num_labels

#         self.classifier = RobertaClassificationHeadType(config)
#         self.apply(self.init_weights)


#     def forward(
#         self,
#         input_ids,
#         attention_mask,
#         token_type_ids=None,
#         entity_ids=None,
#         entity_embeddings=None,
#         entity_attention_mask=None,
#         entity_segment_ids=None,
#         entity_position_ids=None,
#         labels=None,
#         pos=None
#     ):
#         outputs = super(LukeForTyping, self).forward(
#             input_ids,
#             word_attention_mask=attention_mask,
#             word_segment_ids=token_type_ids,
#             entity_ids=entity_ids,
#             entity_embeddings=entity_embeddings,
#             entity_attention_mask=entity_attention_mask,
#             entity_segment_ids=entity_segment_ids,
#             entity_position_ids=entity_position_ids,
#         )

#         sequence_output = outputs[0][:, : input_ids.size(1), :]

#         logits = self.classifier(sequence_output, pos=pos)

#         outputs = (logits,) + outputs[2:]
#         if labels is not None:
#             loss_fct = BCEWithLogitsLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
#             outputs = (loss,) + outputs

#         return outputs  # (loss), logits, (hidden_states), (attentions)






# class LukeForTriviaQuestionAnswering(LukeEntityAwareAttentionModel):
#     def __init__(self, config):
#         super(LukeForTriviaQuestionAnswering, self).__init__(config)

#         self.qa_outputs = nn.Linear(self.config.hidden_size, 2)
#         self.apply(self.init_weights)


#     def or_softmax_cross_entropy_loss_one_doc(self, logits, target, ignore_index=-1, dim=-1):
#         """loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf"""
#         # assert logits.ndim == 2
#         # assert target.ndim == 2
#         # assert logits.size(0) == target.size(0)

#         # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
#         # here, the numerator is the sum of a few potential targets, where some of them is the correct answer
#         bsz = logits.shape[0]

#         # compute a target mask
#         target_mask = target == ignore_index
#         # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
#         masked_target = target * (1 - target_mask.long())
#         # gather logits
#         gathered_logits = logits.gather(dim=dim, index=masked_target)
#         # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
#         gathered_logits[target_mask] = -10000.0#float('-inf')

#         # each batch is one example
#         gathered_logits = gathered_logits.view(bsz, -1)
#         logits = logits.view(bsz, -1)

#         # numerator = log(sum(exp(gathered logits)))
#         log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)
#         # denominator = log(sum(exp(logits)))
#         log_norm = torch.logsumexp(logits, dim=dim, keepdim=False)

#         # compute the loss
#         loss = -(log_score - log_norm)

#         # some of the examples might have a loss of `inf` when `target` is all `ignore_index`.
#         # remove those from the loss before computing the sum. Use sum instead of mean because
#         # it is easier to compute
#         # return loss[~torch.isinf(loss)].sum()
#         return loss.mean()#loss.sum() / len(loss)    

#     def forward(
#         self,
#         input_ids,
#         attention_mask,
#         token_type_ids=None,
#         entity_ids=None,
#         entity_embeddings=None,
#         entity_attention_mask=None,
#         entity_segment_ids=None,
#         entity_position_ids=None,
#         start_positions=None,
#         end_positions=None,
#         answer_masks=None,
#     ):
#         bsz = input_ids.shape[0]
#         max_segment = input_ids.shape[1]

#         input_ids = input_ids.view(-1, input_ids.size(-1))
#         attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
#         token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None


#         entity_ids = entity_ids.view(-1, entity_ids.size(-1)) if entity_ids is not None else None
#         entity_embeddings = entity_embeddings.view(bsz*max_segment, entity_embeddings.size(-2), entity_embeddings.size(-1)) if entity_embeddings is not None else None
#         entity_attention_mask = entity_attention_mask.view(-1, entity_attention_mask.size(-1)) if entity_attention_mask is not None else None
#         entity_segment_ids = entity_segment_ids.view(-1, entity_segment_ids.size(-1)) if entity_segment_ids is not None else None
#         entity_position_ids = entity_position_ids.view(bsz*max_segment, entity_position_ids.size(-2), entity_position_ids.size(-1)) if entity_position_ids is not None else None

#         outputs = super(LukeForTriviaQuestionAnswering, self).forward(
#             input_ids,
#             word_attention_mask=attention_mask,
#             word_segment_ids=token_type_ids,
#             entity_ids=entity_ids,
#             entity_embeddings=entity_embeddings,
#             entity_attention_mask=entity_attention_mask,
#             entity_segment_ids=entity_segment_ids,
#             entity_position_ids=entity_position_ids,
#         )

#         sequence_output = outputs[0][:, : input_ids.size(1), :]


#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)

#         start_logits = start_logits.view(bsz, max_segment, -1) # (bsz, max_segment, seq_length)
#         end_logits = end_logits.view(bsz, max_segment, -1) # (bsz, max_segment, seq_length)

#         outputs = (start_logits, end_logits,) + outputs[2:]

#         if start_positions is not None and end_positions is not None:

#             start_loss = self.or_softmax_cross_entropy_loss_one_doc(start_logits, start_positions, ignore_index=-1)
#             end_loss = self.or_softmax_cross_entropy_loss_one_doc(end_logits, end_positions, ignore_index=-1)

#             total_loss = (start_loss + end_loss) / 2

#             outputs = (total_loss,) + outputs

#         return outputs  




# class LukeForQuestionAnsweringHotpotSeg(LukeEntityAwareAttentionModel):
#     def __init__(self, config):
#         super(LukeForQuestionAnsweringHotpotSeg, self).__init__(config)

#         self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
#         self.sent_linear = nn.Linear(config.hidden_size*2, config.hidden_size) 
#         self.sent_classifier = nn.Linear(config.hidden_size, 2) 

#         self.apply(self.init_weights)


#     def or_softmax_cross_entropy_loss_one_doc(self, logits, target, ignore_index=-1, dim=-1):
#         """loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf"""
#         # assert logits.ndim == 2
#         # assert target.ndim == 2
#         # assert logits.size(0) == target.size(0)

#         # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
#         # here, the numerator is the sum of a few potential targets, where some of them is the correct answer
#         bsz = logits.shape[0]

#         # compute a target mask
#         target_mask = target == ignore_index
#         # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
#         masked_target = target * (1 - target_mask.long())
#         # gather logits
#         gathered_logits = logits.gather(dim=dim, index=masked_target)
#         # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
#         gathered_logits[target_mask] = -10000.0#float('-inf')

#         # each batch is one example
#         gathered_logits = gathered_logits.view(bsz, -1)
#         logits = logits.view(bsz, -1)

#         # numerator = log(sum(exp(gathered logits)))
#         log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)
#         # denominator = log(sum(exp(logits)))
#         log_norm = torch.logsumexp(logits, dim=dim, keepdim=False)

#         # compute the loss
#         loss = -(log_score - log_norm)

#         # some of the examples might have a loss of `inf` when `target` is all `ignore_index`.
#         # remove those from the loss before computing the sum. Use sum instead of mean because
#         # it is easier to compute
#         # return loss[~torch.isinf(loss)].sum()
#         return loss.mean()#loss.sum() / len(loss)    

#     def forward(
#         self,
#         input_ids,
#         attention_mask,
#         token_type_ids=None,
#         entity_ids=None,
#         entity_embeddings=None,
#         entity_attention_mask=None,
#         entity_segment_ids=None,
#         entity_position_ids=None,
#         start_positions=None,
#         end_positions=None,
#         answer_masks=None,
#         sent_start_mapping=None,
#         sent_end_mapping=None,
#         sent_labels=None,
#     ):
#         bsz = input_ids.shape[0]
#         max_segment = input_ids.shape[1]

#         input_ids = input_ids.view(-1, input_ids.size(-1))
#         attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
#         token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None


#         entity_ids = entity_ids.view(-1, entity_ids.size(-1)) if entity_ids is not None else None
#         entity_embeddings = entity_embeddings.view(bsz*max_segment, entity_embeddings.size(-2), entity_embeddings.size(-1)) if entity_embeddings is not None else None
#         entity_attention_mask = entity_attention_mask.view(-1, entity_attention_mask.size(-1)) if entity_attention_mask is not None else None
#         entity_segment_ids = entity_segment_ids.view(-1, entity_segment_ids.size(-1)) if entity_segment_ids is not None else None
#         entity_position_ids = entity_position_ids.view(bsz*max_segment, entity_position_ids.size(-2), entity_position_ids.size(-1)) if entity_position_ids is not None else None

#         outputs = super(LukeForQuestionAnsweringHotpotSeg, self).forward(
#             input_ids,
#             word_attention_mask=attention_mask,
#             word_segment_ids=token_type_ids,
#             entity_ids=entity_ids,
#             entity_embeddings=entity_embeddings,
#             entity_attention_mask=entity_attention_mask,
#             entity_segment_ids=entity_segment_ids,
#             entity_position_ids=entity_position_ids,
#         )

#         sequence_output = outputs[0][:, : input_ids.size(1), :]


#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)

#         start_logits = start_logits.view(bsz, max_segment, -1) # (bsz, max_segment, seq_length)
#         end_logits = end_logits.view(bsz, max_segment, -1) # (bsz, max_segment, seq_length)

#         if sent_start_mapping is not None:
#             start_rep = torch.matmul(sent_start_mapping, sequence_output)
#             end_rep = torch.matmul(sent_end_mapping, sequence_output)
#             sent_rep = torch.cat([start_rep, end_rep], dim=-1)

#             sent_logits = gelu(self.sent_linear(sent_rep))
#             sent_logits = self.sent_classifier(sent_logits).squeeze(-1)
#         else:
#             sent_logits = None

#         outputs = (start_logits, end_logits, sent_logits) + outputs[2:]

#         if start_positions is not None and end_positions is not None:

#             start_loss = self.or_softmax_cross_entropy_loss_one_doc(start_logits, start_positions, ignore_index=-1)
#             end_loss = self.or_softmax_cross_entropy_loss_one_doc(end_logits, end_positions, ignore_index=-1)

#             sent_loss = loss_fct(sent_logits.view(-1, 2), sent_labels.view(-1))

#             total_loss = (start_loss + end_loss) / 2 + sent_loss

#             outputs = (total_loss,) + outputs

#         return outputs  






# class LukeForWikihopMulti(LukeEntityAwareAttentionModel):
#     def __init__(self, config):
#         super(LukeForWikihopMulti, self).__init__(config)

#         self.qa_outputs = nn.Linear(self.config.hidden_size, 1)
#         self.apply(self.init_weights)

 
#     def forward(
#         self,
#         input_ids,
#         attention_mask,
#         token_type_ids=None,
#         entity_ids=None,
#         entity_embeddings=None,
#         entity_attention_mask=None,
#         entity_segment_ids=None,
#         entity_position_ids=None,
#         answer_index=None,
#         instance_mask=None,
#         candidate_pos=None,
#         candidate_num=None,

#     ):
#         bsz = input_ids.shape[0]
#         max_segment = input_ids.shape[1]

#         input_ids = input_ids.view(-1, input_ids.size(-1))
#         attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
#         token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None


#         entity_ids = entity_ids.view(-1, entity_ids.size(-1)) if entity_ids is not None else None
#         entity_embeddings = entity_embeddings.view(bsz*max_segment, entity_embeddings.size(-2), entity_embeddings.size(-1)) if entity_embeddings is not None else None
#         entity_attention_mask = entity_attention_mask.view(-1, entity_attention_mask.size(-1)) if entity_attention_mask is not None else None
#         entity_segment_ids = entity_segment_ids.view(-1, entity_segment_ids.size(-1)) if entity_segment_ids is not None else None
#         entity_position_ids = entity_position_ids.view(bsz*max_segment, entity_position_ids.size(-2), entity_position_ids.size(-1)) if entity_position_ids is not None else None

#         candidate_pos = candidate_pos.view(-1, candidate_pos.size()[2], candidate_pos.size()[3])


#         outputs = super(LukeForWikihopMulti, self).forward(
#             input_ids,
#             word_attention_mask=attention_mask,
#             word_segment_ids=token_type_ids,
#             entity_ids=entity_ids,
#             entity_embeddings=entity_embeddings,
#             entity_attention_mask=entity_attention_mask,
#             entity_segment_ids=entity_segment_ids,
#             entity_position_ids=entity_position_ids,
#         )

#         sequence_output = outputs[0][:, : input_ids.size(1), :]


#         context_hidden = torch.matmul(candidate_pos, sequence_output)

#         logits = self.qa_outputs(context_hidden).squeeze(-1)
#         logits = logits.view(bsz, max_segment, -1)

#         # ignore_index = -1
#         candidate_mask = torch.sum(candidate_pos, dim = 2)
#         candidate_mask = candidate_mask.view(bsz, max_segment, -1)
#         ignore_mask = candidate_mask > 0
#         ignore_mask = 1 - ignore_mask.long()
#         ignore_mask = ignore_mask.bool()
#         logits[ignore_mask] = 0
#         logits = torch.sum(logits, dim = 1)

#         candidate_mask = torch.sum(candidate_mask, dim = 1)
#         ignore_mask = candidate_mask > 0
#         ignore_mask = 1 - ignore_mask.long()
#         ignore_mask = ignore_mask.bool()
        
#         logits[ignore_mask] = -10000.0
#         outputs = (logits,) + outputs[2:]
#         if answer_index is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits, answer_index)
#             outputs = (loss,) + outputs

#         return outputs  





