# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# from fairseq.models.roberta import RobertaModel
# from fairseq import utils

from fairseq.models.roberta import RobertaModel
from fairseq import utils
import torch
from lama.modules.base_connector import *
from transformers import RobertaForMaskedLM, AutoTokenizer, RobertaTokenizer, RobertaConfig, RobertaEntForMaskedLM
# from lama.modules.model import LukeForMaskedLM
import json
import pickle
import numpy as np
import torch.nn.functional as F
import unicodedata
import os
import math
import urllib.parse

class RobertaVocab(object):
    def __init__(self, roberta):
        self.roberta = roberta

    def __getitem__(self, arg):
        value = ""
        try:
            predicted_token_bpe = self.roberta.task.source_dictionary.string([arg])
            if (
                predicted_token_bpe.strip() == ROBERTA_MASK
                or predicted_token_bpe.strip() == ROBERTA_START_SENTENCE
            ):
                value = predicted_token_bpe.strip()
            else:
                value = self.roberta.bpe.decode(str(predicted_token_bpe)).strip()
        except Exception as e:
            print(arg)
            print(predicted_token_bpe)
            print(value)
            print("Exception {} for input {}".format(e, arg))
        return value


class RobertaConcat(Base_Connector):
    def __init__(self, args):
        super().__init__()
        roberta_model_dir = args.roberta_model_dir
        roberta_model_name = args.roberta_model_name
        roberta_vocab_name = args.roberta_vocab_name
        self.dict_file = "{}/{}".format(roberta_model_dir, roberta_vocab_name)

        self.model = RobertaModel.from_pretrained(
            roberta_model_dir, checkpoint_file=roberta_model_name
        )
        self.bpe = self.model.bpe
        self.task = self.model.task
        self._build_vocab()
        self._init_inverse_vocab()
        self.max_sentence_length = args.max_sentence_length
        self.modL = args.modL

        self.tokenizer = RobertaTokenizer.from_pretrained( args.luke_model_dir )


        self.model =  RobertaEntForMaskedLM.from_pretrained( args.luke_model_dir )
        self.model.roberta.entity_embeddings.token_type_embeddings.weight.data.copy_(self.model.roberta.embeddings.token_type_embeddings.weight.data)
        self.model.roberta.entity_embeddings.LayerNorm.weight.data.copy_(self.model.roberta.embeddings.LayerNorm.weight.data)
        self.model.roberta.entity_embeddings.LayerNorm.bias.data.copy_(self.model.roberta.embeddings.LayerNorm.bias.data)


        self.dim = dim = 768
        if self.modL>0:
            embed_path = '/home/yedeming/PELT/wikiembed_roberta/'
            self.name2pageid =  json.load(open(embed_path+'name2id.json'))  # Metadata in Tsinghua Cloud
            self.qid2pageid =  json.load(open(embed_path +'qid2pageid.json'))
            self.pageid2id = pickle.load(open(embed_path+'wiki_pageid2embedid.pkl', 'rb'))
            self.tot_entity_embed = np.load(embed_path + 'wiki_entity_embed_256.npy')
            L = np.linalg.norm(self.tot_entity_embed, axis=1)
            self.tot_entity_embed = self.tot_entity_embed / np.expand_dims(L, axis=-1) * self.modL
            self.dim = self.tot_entity_embed.shape[1]

        else:
            self.name2pageid = {}
            self.qid2pageid = {}   

    def _cuda(self):
        self.model.cuda()
        self.model.cuda()


    def _is_subword(self, token):
        if isinstance(self.tokenizer, RobertaTokenizer):
            token = self.tokenizer.convert_tokens_to_string(token)
            if not token.startswith(" ") and not self._is_punctuation(token[0]):
                return True
        elif token.startswith("##"):
            return True

        return False

    @staticmethod
    def _is_punctuation(char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    @staticmethod
    def _normalize_mention(text):
        return " ".join(text.split(" ")).strip()


    def _detect_mentions(self, tokens, name):
        name = self._normalize_mention(name)

        for start, token  in enumerate(tokens):

            if self._is_subword(token) and (start>1):
                continue
        
            for end in range(start, len(tokens)):
                if end < len(tokens) and self._is_subword(tokens[end]) and (end>1):
                    continue
                mention_text = self.tokenizer.convert_tokens_to_string(tokens[start:end])
                mention_text = self._normalize_mention(mention_text)
                if len(mention_text) > len(name):
                    break
                if mention_text.lower()==name.lower():
                    return start, end


        return -1, -1


    def _detech_mentions_squad(self, tokens, ent2id):
        mentions = []
        cur = 0
        for start, token  in enumerate(tokens):

            if start < cur:
                continue

            if self._is_subword(token) and (start>1):
                continue

            for end in range(min(start + 30, len(tokens)), start, -1):
                if end < len(tokens) and self._is_subword(tokens[end]) and (end>1):
                    continue

                mention_text = self.tokenizer.convert_tokens_to_string(tokens[start:end])
                mention_text = self._normalize_mention(mention_text)
                if mention_text in ent2id:

                    cur = end
                    pageid = ent2id[mention_text]

                    mentions.append((pageid, start, end))

                    break
                    


        return mentions



    def _build_vocab(self):
        self.vocab = []
        for key in range(ROBERTA_VOCAB_SIZE):
            predicted_token_bpe = self.task.source_dictionary.string([key])
            try:
                value = self.bpe.decode(predicted_token_bpe)

                if value[0] == " ":  # if the token starts with a whitespace
                    value = value.strip()
                else:
                    # this is subword information
                    value = "_{}_".format(value)

                if value in self.vocab:
                    # print("WARNING: token '{}' is already in the vocab".format(value))
                    value = "{}_{}".format(value, key)

                self.vocab.append(value)

            except Exception as e:
                self.vocab.append(predicted_token_bpe.strip())

    def get_id(self, input_string):
        # Roberta predicts ' London' and not 'London'
        string = " " + str(input_string).strip()
        text_spans_bpe = self.bpe.encode(string.rstrip())
        tokens = self.task.source_dictionary.encode_line(
            text_spans_bpe, append_eos=False
        )
        return [element.item() for element in tokens.long().flatten()]

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True, sub_labels=None, sub_ids=None):
        # try_cuda = False   # for debug
        if not sentences_list:
            return None
        if try_cuda:
            self.model.cuda()

            

        masked_indices_list = []
        max_len = 0
        output_tokens_list = []
        input_embeds_list = []
        attention_mask_list = []
        position_ids_list = []
        input_ids_list = []

        entity_embeddings_list = []
        entity_attention_mask_list = []
        entity_position_ids_list = []

        entity_K = 1
        if sub_ids is None:
            sub_ids = [-1]* len(sentences_list)
        for masked_inputs_list, sub_label, sub_id in zip(sentences_list, sub_labels, sub_ids):

            assert(len(masked_inputs_list)==1)
            for idx, masked_input in enumerate(masked_inputs_list):
                if sub_id in self.qid2pageid:
                    sub_pageid = self.qid2pageid[sub_id]
                else:
                    sub_label_align = urllib.parse.unquote(sub_label) # Added
                    if sub_label_align in self.name2pageid:
                        sub_pageid =  self.name2pageid[sub_label_align]
                    elif sub_label_align.lower() in self.name2pageid:
                        sub_pageid =  self.name2pageid[sub_label_align.lower()]
                    else:
                        sub_pageid = -1


                masked_input = masked_input.replace(MASK, ROBERTA_MASK)

                tokens = self.tokenizer.tokenize(masked_input, add_special_tokens=False)
                tokens = [self.tokenizer.cls_token] + tokens #+ [self.tokenizer.sep_token]
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                mask_s = -1
                for k in range(len(tokens)):
                    if tokens[k]==ROBERTA_MASK:
                        mask_s = k
                        break
                assert(mask_s!=-1)

                if input_ids[mask_s-1]==1437:
                    input_ids = input_ids[:mask_s-1] + input_ids[mask_s:]
                    tokens = tokens[:mask_s-1] + tokens[mask_s:]
                    mask_s -= 1

                l_id = self.tokenizer.encode(' (', add_special_tokens=False)
                assert(len(l_id)==1)
                r_id = self.tokenizer.encode(' )', add_special_tokens=False)
                assert(len(r_id)==1)


                output_tokens = []
                mentions = []

                if sub_label=='Squad':
                    assert (False)
                    ents = self.qid2ents[sub_id]
                    ent2id = {}
                    for x in ents:
                        if x[1]!='MASK':
                            ent2id[self._normalize_mention(x[1])] = x[0]
                    mentions = self._detech_mentions_squad(tokens, ent2id)

                elif  sub_pageid>=0 and self.modL>0:
                    sub_s, sub_e = self._detect_mentions(tokens, sub_label)
                    if( sub_s>=0):

                        mentions = [(sub_pageid, sub_s, sub_e, sub_e+1)]
                        input_ids = input_ids[:sub_e] + l_id + [self.tokenizer.mask_token_id] + r_id + input_ids[sub_e:]
                        mask_s += 3

                    else:
                        print (tokens, sub_label)
                        mentions = []

                input_ids = input_ids[:self.max_sentence_length-1] + [self.tokenizer.sep_token_id]
                entity_embeddings = []
                entity_position_ids = []
                # entity_attention_mask = []
                np.random.shuffle(mentions)

                for page_id, sub_s, sub_e, pos_ent in mentions:

                    if page_id in self.pageid2id:
                        embed_id = self.pageid2id[page_id]
                        entity_embedding = np.array(self.tot_entity_embed[embed_id], dtype=np.float32)

                        entity_embeddings.append(entity_embedding)
                        entity_position_ids.append(pos_ent + 2)
                    if len(entity_embeddings)>=entity_K:
                        break

                L = len(input_ids)
                max_len = max(max_len, L)

                padding_length = self.max_sentence_length - L
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask = [1] * L + [0] * padding_length
                attention_mask += [1] * len(entity_position_ids) + [0] * (entity_K-len(entity_position_ids))
                for page_id, sub_s, sub_e, pos_ent in mentions:
                    attention_mask[pos_ent] = 0



                while len(entity_embeddings) < entity_K:
                    entity_embeddings.append(np.zeros((self.dim, ), dtype=np.float32))
                    entity_position_ids.append(0)

                output_tokens_list.append(np.array(input_ids, dtype=np.int64))

                input_ids_list.append(input_ids)

                attention_mask_list.append(attention_mask)
                masked_indices_list.append(mask_s)
                entity_embeddings_list.append(torch.tensor(entity_embeddings, dtype=torch.float32))
                entity_position_ids_list.append(entity_position_ids)



        input_ids_list = torch.tensor(input_ids_list, dtype=torch.int64)
        attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.int64)
        masked_indices_list = torch.tensor(masked_indices_list, dtype=torch.int64)

        entity_embeddings_list = torch.stack(entity_embeddings_list,  dim=0)
        entity_position_ids_list = torch.tensor(entity_position_ids_list,  dtype=torch.int64)

        with torch.no_grad():
            self.model.eval()
            if try_cuda:
                outputs = self.model(

                    input_ids=input_ids_list.cuda(),
                    attention_mask=attention_mask_list.cuda(),
                    entity_embeddings=entity_embeddings_list.cuda(),
                    entity_position_ids=entity_position_ids_list.cuda()

                )
            else:
                outputs = self.model(
                    input_ids=input_ids_list,
                    attention_mask=attention_mask_list,
                    entity_embeddings=entity_embeddings_list,
                    entity_position_ids=entity_position_ids_list
                )

            log_probs = outputs[0]



        return log_probs.cpu(), output_tokens_list, masked_indices_list.unsqueeze(1)

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):
        # TBA
        return None

