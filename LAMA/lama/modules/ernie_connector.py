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
from transformers import AutoTokenizer, RobertaTokenizer, RobertaConfig
# from lama.modules.model import LukeForMaskedLM
import json
import pickle
import numpy as np
import torch.nn.functional as F
import unicodedata
import os
import math
from .knowledge_bert.modeling import ERNIEForMaskedLM
from tqdm import tqdm

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


class ERNIE(Base_Connector):
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
        self.max_seq_length = args.max_sentence_length
        print ('Loading KG')
        self.tokenizer = RobertaTokenizer.from_pretrained( '/home/yedeming/bert_models/roberta-base' )

        self.model = ERNIEForMaskedLM.from_pretrained(args.model_name_or_path)[0]
        self.args = args
        vecs = []
        vecs.append([0]*100)
        with open(os.path.join(args.model_name_or_path, "kg_embed/entity2vec.vec"), 'r') as fin:
            for line in tqdm(fin):
                vec = line.strip().split('\t')
                vec = [float(x) for x in vec]
                vecs.append(vec)
                # if len(vecs)>=100:
                #     break

        self.ent_emb = torch.FloatTensor(vecs)
        self.ent_emb = torch.nn.Embedding.from_pretrained(self.ent_emb)
        self.entity2id  = {}
        with open(os.path.join(args.model_name_or_path, "kg_embed/entity2id.txt")) as fin:
            fin.readline()
            for line in fin:
                qid, eid = line.strip().split('\t')
                self.entity2id[qid] = int(eid)            
                # if len(self.entity2id)>=100:
                #     break

    def _cuda(self):
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

        input_ent_emb_list = []
        ent_mask_list = []

        # pad_id = self.task.source_dictionary.pad()
        # pad_id = self.tokenizer.pad_id
        if sub_ids is None:
            sub_ids = [-1]* len(sentences_list)
        for masked_inputs_list, sub_label, sub_id in zip(sentences_list, sub_labels, sub_ids):

            assert(len(masked_inputs_list)==1)
            for idx, masked_input in enumerate(masked_inputs_list):
                if sub_id in self.entity2id:
                    sub_embid = self.entity2id[sub_id]
                else:
                    sub_embid = -1


                masked_input = masked_input.replace(MASK, ROBERTA_MASK)

                tokens = self.tokenizer.tokenize(masked_input)
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


                output_tokens = []
                mentions = []

                sub_s, sub_e = self._detect_mentions(tokens, sub_label)


                if( sub_s>=0) and (sub_embid>=0):
                    mentions = [(sub_embid, sub_s, sub_e)]


                input_ids = input_ids[:self.max_seq_length-1] + [self.tokenizer.sep_token_id]

                input_ent = torch.zeros((self.max_seq_length,), dtype=torch.int64)
                ent_mask = torch.zeros((self.max_seq_length,), dtype=torch.int64)

                for embedid, sub_s, sub_e in mentions:

                    for p in range(sub_s, sub_e):
                        if p >= self.max_seq_length-1:
                            break
                        input_ent[p] = embedid + 1
                        ent_mask[p] = 1
                input_ent_emb = self.ent_emb(input_ent)

                L = len(input_ids)
                max_len = max(max_len, L)

                padding_length = self.max_seq_length - L
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask = [1] * L + [0] * padding_length

                output_tokens_list.append(np.array(input_ids, dtype=np.int64))

                input_ids_list.append(input_ids)

                attention_mask_list.append(attention_mask)
                masked_indices_list.append(mask_s)
                input_ent_emb_list.append(input_ent_emb)
                ent_mask_list.append(ent_mask)



        input_ids_list = torch.tensor(input_ids_list, dtype=torch.int64)
        attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.int64)
        masked_indices_list = torch.tensor(masked_indices_list, dtype=torch.int64)

        input_ent_emb_list = torch.stack(input_ent_emb_list,  dim=0)
        ent_mask_list = torch.stack(ent_mask_list,  dim=0)

        with torch.no_grad():
            self.model.eval()
            if try_cuda:
                outputs = self.model(

                    input_ids=input_ids_list.cuda(),
                    attention_mask=attention_mask_list.cuda(), 
                    input_ent=input_ent_emb_list.cuda(), 
                    ent_mask=ent_mask_list.cuda()

                )
            else:
                outputs = self.model(
                    input_ids=input_ids_list,
                    attention_mask=attention_mask_list, 
                    input_ent=input_ent_emb_list,
                    ent_mask=ent_mask_list
                )

            log_probs = outputs[0]



        return log_probs.cpu(), output_tokens_list, masked_indices_list.unsqueeze(1)

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):
        # TBA
        return None

