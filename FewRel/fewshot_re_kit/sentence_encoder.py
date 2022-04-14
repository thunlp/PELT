import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from . import network
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, RobertaEntModel
from .model import LukeEntityAwareAttentionModel


class RobertaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False): 
        nn.Module.__init__(self)
        self.pretrain_path = pretrain_path
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_path)
        self.cat_entity_rep = cat_entity_rep
        self.model = RobertaEntModel.from_pretrained(pretrain_path) 
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_path)


        if self.tokenizer.convert_tokens_to_ids('[unused0]')==self.tokenizer.unk_token_id:
            special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.m1, self.m2, self.m3, self.m4 = self.tokenizer.convert_tokens_to_ids(['[unused0]', '[unused1]', '[unused2]', '[unused3]'])


    def forward(self, inputs):
        if not self.cat_entity_rep:
            _, x = self.model(inputs['word'], attention_mask=inputs['mask'])
            return x
        else:
            outputs = self.model(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)
            return state

    
    def tokenize(self, words, pos_head, pos_tail):
        subwords_idx = [0]
        tokens = []
        for i in range(len(words)):
            word = words[i]
            if i > 0:
                word = ' ' + word
            tokens.extend(self.tokenizer.tokenize(word))
            subwords_idx.append(len(tokens))

        hiL = subwords_idx[pos_head[0]]
        hiR = subwords_idx[pos_head[-1]+1]

        tiL = subwords_idx[pos_tail[0]]
        tiR = subwords_idx[pos_tail[-1]+1]

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        ins = [(hiL, [self.m1]), (hiR, [self.m2]), (tiL, [self.m3]), (tiR, [self.m4])]
        ins = sorted(ins)
        h_pos = [-1, -1]
        t_pos = [-1, -1]
        num_inserted = 0
        for i in range(0, 4):
            insert_pos = ins[i][0] + num_inserted
            indexed_tokens = indexed_tokens[:insert_pos] + ins[i][1] + indexed_tokens[insert_pos:]
            if ins[i][0]==hiL:
                h_pos[0] = insert_pos
            if ins[i][0]==hiR:
                h_pos[1] = insert_pos 

            if ins[i][0]==tiL:
                t_pos[0] = insert_pos
            if ins[i][0]==tiR:
                t_pos[1] = insert_pos

            num_inserted += len(ins[i][1])
        assert(h_pos[0]>=0 and h_pos[1]>=0 and t_pos[0]>=0 and t_pos[1]>=0)

        return indexed_tokens, h_pos, t_pos




class RobertaEntSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False): 
        nn.Module.__init__(self)
        self.pretrain_path = pretrain_path
        self.model = RobertaEntModel.from_pretrained(pretrain_path) 
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_path)

        self.model.entity_embeddings.token_type_embeddings.weight.data.copy_(self.model.embeddings.token_type_embeddings.weight.data)
        self.model.entity_embeddings.LayerNorm.weight.data.copy_(self.model.embeddings.LayerNorm.weight.data)
        self.model.entity_embeddings.LayerNorm.bias.data.copy_(self.model.embeddings.LayerNorm.bias.data)


        if self.tokenizer.convert_tokens_to_ids('[unused0]')==self.tokenizer.unk_token_id:
            special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.mask_id = self.tokenizer.mask_token_id
        self.m1, self.m2, self.m3, self.m4 = self.tokenizer.convert_tokens_to_ids(['[unused0]', '[unused1]', '[unused2]', '[unused3]'])
        l_bracket_id = self.tokenizer.encode(' (', add_special_tokens=False)
        assert(len(l_bracket_id)==1)
        self.l_bracket_id = l_bracket_id[0]

        r_bracket_id = self.tokenizer.encode(' )', add_special_tokens=False)
        assert(len(r_bracket_id)==1)
        self.r_bracket_id = r_bracket_id[0]


        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size 
        self.cat_entity_rep = cat_entity_rep


    def forward(self, inputs):

        if not self.cat_entity_rep:
            _, _, x = self.model(inputs['word'], attention_mask=inputs['mask'], entity_embeddings=inputs['entity_embeddings'], \
                        entity_position_ids=inputs['entity_position_ids'])
            return x
        else:
            outputs = self.model(inputs['word'], attention_mask=inputs['mask'], entity_embeddings=inputs['entity_embeddings'], \
                        entity_position_ids=inputs['entity_position_ids'])
            
            sequence_output = outputs[0]            
            tensor_range = torch.arange(sequence_output.shape[0])
            h_state = sequence_output[tensor_range, inputs["pos1"]]
            t_state = sequence_output[tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)
            return state


    def tokenize(self, words, pos_head, pos_tail):
        subwords_idx = [0]
        tokens = []
        for i in range(len(words)):
            word = words[i]
            if i > 0:# and self.args.model_type.find('roberta')!=-1:
                word = ' ' + word
            tokens.extend(self.tokenizer.tokenize(word))
            subwords_idx.append(len(tokens))

        hiL = subwords_idx[pos_head[0]]
        hiR = subwords_idx[pos_head[-1]+1]

        tiL = subwords_idx[pos_tail[0]]
        tiR = subwords_idx[pos_tail[-1]+1]

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        ins = [(hiL, [self.m1]), (hiR, [self.l_bracket_id, self.mask_id, self.r_bracket_id, self.m2]), (tiL, [self.m3]), (tiR, [self.l_bracket_id, self.mask_id, self.r_bracket_id, self.m4])]

        ins = sorted(ins)
        h_pos = [-1, -1]
        t_pos = [-1, -1]
        num_inserted = 0
        for i in range(0, 4):
            insert_pos = ins[i][0] + num_inserted
            indexed_tokens = indexed_tokens[:insert_pos] + ins[i][1] + indexed_tokens[insert_pos:]
            if ins[i][0]==hiL:
                h_pos[0] = insert_pos
            if ins[i][0]==hiR:
                h_pos[1] = insert_pos 

            if ins[i][0]==tiL:
                t_pos[0] = insert_pos
            if ins[i][0]==tiR:
                t_pos[1] = insert_pos 

            num_inserted += len(ins[i][1])
        assert(h_pos[0]>=0 and h_pos[1]>=0 and t_pos[0]>=0 and t_pos[1]>=0)
        h_pos[1] += 3
        t_pos[1] += 3

            
        return indexed_tokens, h_pos, t_pos



class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity

    def forward(self, inputs):
        if not self.cat_entity_rep:
            _, x = self.bert(inputs['word'], attention_mask=inputs['mask'])
            return x
        else:
            outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)
            return state
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        pos1_end_index = self.max_length
        pos2_end_index = self.max_length
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
                pos1_end_index = len(tokens)
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
                pos2_end_index = len(tokens)
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]


        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, [pos1_in_index - 1, pos1_end_index-1], [pos2_in_index - 1, pos2_end_index-1]



# class LukeSentenceEncoder(nn.Module):

#     def __init__(self, pretrain_path, max_length, cat_entity_rep=False): 
#         nn.Module.__init__(self)

#         config = RobertaConfig.from_pretrained(pretrain_path)
#         config.bert_model_name = 'roberta-base'
#         config.bert_model_name = 'roberta-base'
#         config.entity_emb_size = 256
#         config.entity_vocab_size = 500000
#         config.is_decoder = False

#         config.num_labels = 2

#         self.roberta = LukeEntityAwareAttentionModel(config) 
#         self.roberta.load_state_dict(torch.load(os.path.join(pretrain_path, 'pytorch_model.bin')), strict=False)

#         self.max_length = max_length
#         self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_path)
#         self.vocab_size = self.tokenizer.vocab_size 
#         self.max_mention_length = 30
#         self.cat_entity_rep = cat_entity_rep


#     def forward(self, inputs):

#         if not self.cat_entity_rep:
#             _, _, x = self.roberta(inputs['word'], word_attention_mask=inputs['mask'], entity_ids=inputs['entity_ids'], \
#                         entity_attention_mask=inputs['entity_attention_mask'], \
#                         entity_position_ids=inputs['entity_position_ids'])
#             return x
#         else:
#             outputs = self.roberta(inputs['word'], word_attention_mask=inputs['mask'], entity_ids=inputs['entity_ids'], \
#                         entity_attention_mask=inputs['entity_attention_mask'], \
#                         entity_position_ids=inputs['entity_position_ids'])
#             tensor_range = torch.arange(inputs['word'].size()[0])
#             h_state = outputs[0][tensor_range, inputs["pos1"]]
#             t_state = outputs[0][tensor_range, inputs["pos2"]]
#             state = torch.cat((h_state, t_state), -1)
#             return state


#     def tokenize(self, raw_tokens, pos_head, pos_tail):
#         def getIns(bped, bpeTokens, tokens, L):
#             resL = 0
#             tkL = " ".join(tokens[:L])
#             bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
#             if bped.find(bped_tkL) == 0:
#                 resL = len(bped_tkL.split())
#             else:
#                 tkL += " "
#                 bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
#                 if bped.find(bped_tkL) == 0:
#                     resL = len(bped_tkL.split())
#                 else:
#                     raise Exception("Cannot locate the position")
#             return resL


#         s = " ".join(raw_tokens)
#         sst = self.tokenizer.tokenize(s)
#         headL = pos_head[0]
#         headR = pos_head[-1] + 1
#         hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
#         hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

#         tailL = pos_tail[0]
#         tailR = pos_tail[-1] + 1
#         tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
#         tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)


#         indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)

#         E1b = 'madeupword0001'
#         E1e = 'madeupword0002'
#         E2b = 'madeupword0003'
#         E2e = '#'

#         ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
#         ins = sorted(ins)
#         h_pos = [-1, -1]
#         t_pos = [-1, -1]
#         for i in range(0, 4):
#             indexed_tokens.insert(ins[i][0] + i, ins[i][1])
#             if ins[i][1]==E1b:
#                 h_pos[0] = ins[i][0] + i + 1
#             if ins[i][1]==E1e:
#                 h_pos[1] = ins[i][0] + i 

#             if ins[i][1]==E2b:
#                 t_pos[0] = ins[i][0] + i + 1
#             if ins[i][1]==E2e:
#                 t_pos[1] = ins[i][0] + i 

#         assert(h_pos[0]>=0 and h_pos[1]>=0 and t_pos[0]>=0 and t_pos[1]>=0)
#         return indexed_tokens, h_pos, t_pos



# class LukeBERTSentenceEncoder(nn.Module):

#     def __init__(self, pretrain_path, max_length, cat_entity_rep=False): 
#         nn.Module.__init__(self)
#         config = BertConfig.from_pretrained('../bert_models/bert-base-uncased')
#         config.bert_model_name = 'bert-base'
#         config.entity_emb_size = 768

#         config.num_labels = 2

#         self.roberta = LukeEntityAwareAttentionModel(config) 

#         self.roberta.load_state_dict(torch.load(os.path.join(pretrain_path, 'pytorch.pt')), strict=False)

#         self.max_length = max_length
#         self.tokenizer = BertTokenizer.from_pretrained('../bert_models/bert-base-uncased')

#         self.vocab_size = self.tokenizer.vocab_size 
#         self.max_mention_length = 30
#         self.cat_entity_rep = cat_entity_rep


#     def forward(self, inputs):

#         if not self.cat_entity_rep:
#             _, _, x = self.roberta(inputs['word'], word_attention_mask=inputs['mask'], entity_embeddings=inputs['entity_embeddings'], \
#                         entity_attention_mask=inputs['entity_attention_mask'], \
#                         entity_position_ids=inputs['entity_position_ids'])
#             return x
#         else:
#             outputs = self.roberta(inputs['word'], word_attention_mask=inputs['mask'], entity_embeddings=inputs['entity_embeddings'], \
#                         entity_attention_mask=inputs['entity_attention_mask'], \
#                         entity_position_ids=inputs['entity_position_ids'])
#             tensor_range = torch.arange(inputs['word'].size()[0])
#             h_state = outputs[0][tensor_range, inputs["pos1"]]
#             t_state = outputs[0][tensor_range, inputs["pos2"]]
#             state = torch.cat((h_state, t_state), -1)
#             return state


#     def tokenize(self, raw_tokens, pos_head, pos_tail):
#         def getIns(bped, bpeTokens, tokens, L):
#             resL = 0
#             tkL = " ".join(tokens[:L])
#             bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
#             if bped.find(bped_tkL) == 0:
#                 resL = len(bped_tkL.split())
#             else:
#                 tkL += " "
#                 bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
#                 if bped.find(bped_tkL) == 0:
#                     resL = len(bped_tkL.split())
#                 else:
#                     raise Exception("Cannot locate the position")
#             return resL


#         s = " ".join(raw_tokens)
#         sst = self.tokenizer.tokenize(s)
#         headL = pos_head[0]
#         headR = pos_head[-1] + 1
#         hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
#         hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

#         tailL = pos_tail[0]
#         tailR = pos_tail[-1] + 1
#         tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
#         tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)


#         indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)
#         ids = self.tokenizer.convert_tokens_to_ids(['[unused0]', '[unused1]', '[unused2]', '[unused3]'])
#         E1b = ids[0]
#         E1e = ids[1]
#         E2b = ids[2]
#         E2e = ids[2]

#         ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
#         ins = sorted(ins)
#         h_pos = [-1, -1]
#         t_pos = [-1, -1]
#         for i in range(0, 4):
#             indexed_tokens.insert(ins[i][0] + i, ins[i][1])
#             if ins[i][1]==E1b:
#                 h_pos[0] = ins[i][0] + i + 1
#             if ins[i][1]==E1e:
#                 h_pos[1] = ins[i][0] + i 

#             if ins[i][1]==E2b:
#                 t_pos[0] = ins[i][0] + i + 1
#             if ins[i][1]==E2e:
#                 t_pos[1] = ins[i][0] + i 

#         assert(h_pos[0]>=0 and h_pos[1]>=0 and t_pos[0]>=0 and t_pos[1]>=0)
#         return indexed_tokens, h_pos, t_pos



    
# class LukePAIRSentenceEncoder(nn.Module):

#     def __init__(self, pretrain_path, max_length): 
#         nn.Module.__init__(self)
#         config = RobertaConfig.from_pretrained(pretrain_path)
#         config.num_labels = 2
#         config.bert_model_name = 'roberta-base'
#         config.entity_emb_size = 768
        
#         self.roberta.load_state_dict(torch.load(os.path.join(pretrain_path, 'pytorch.pt')), strict=False)

#         self.max_length = max_length
#         self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_path)
#         self.vocab_size = self.tokenizer.vocab_size 
#         self.max_mention_length = 30


#     def forward(self, inputs):
#         entity_attention_mask = inputs['entity_attention_mask']

#         outputs = self.roberta(word_ids=inputs['word'], word_attention_mask=inputs['mask'],  \
#                     # word_segment_ids=inputs['seg'], \
#                     entity_embeddings=inputs['entity_embeddings'], entity_attention_mask=entity_attention_mask, \
#                     # entity_segment_ids=inputs['entity_segment_ids'], \
#                     entity_position_ids=inputs['entity_position_ids'] )
        
#         x = outputs[0]
#         return x
    
#     def tokenize(self, raw_tokens, pos_head, pos_tail):
#         def getIns(bped, bpeTokens, tokens, L):
#             resL = 0
#             tkL = " ".join(tokens[:L])
#             bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
#             if bped.find(bped_tkL) == 0:
#                 resL = len(bped_tkL.split())
#             else:
#                 tkL += " "
#                 bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
#                 if bped.find(bped_tkL) == 0:
#                     resL = len(bped_tkL.split())
#                 else:
#                     raise Exception("Cannot locate the position")
#             return resL

#         s = " ".join(raw_tokens)
#         sst = self.tokenizer.tokenize(s)
#         headL = pos_head[0]
#         headR = pos_head[-1] + 1
#         hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
#         hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

#         tailL = pos_tail[0]
#         tailR = pos_tail[-1] + 1
#         tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
#         tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

#         indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)

#         E1b = self.vocab_size   #'madeupword0000'
#         E1e = self.vocab_size+1 #'madeupword0001'
#         E2b = self.vocab_size+2 #'madeupword0002'
#         E2e = self.vocab_size+3 #'madeupword0003'

#         ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
#         ins = sorted(ins)
#         h_pos = [-1, -1]
#         t_pos = [-1, -1]
#         for i in range(0, 4):
#             indexed_tokens.insert(ins[i][0] + i, ins[i][1])
#             if ins[i][1]==E1b:
#                 h_pos[0] = ins[i][0] + i + 1
#             if ins[i][1]==E1e:
#                 h_pos[1] = ins[i][0] + i 

#             if ins[i][1]==E2b:
#                 t_pos[0] = ins[i][0] + i + 1
#             if ins[i][1]==E2e:
#                 t_pos[1] = ins[i][0] + i 


#         return indexed_tokens, h_pos, t_pos



# class BERTPAIRSentenceEncoder(nn.Module):

#     def __init__(self, pretrain_path, max_length): 
#         nn.Module.__init__(self)
#         self.bert = BertForSequenceClassification.from_pretrained(
#                 pretrain_path,
#                 num_labels=2)
#         self.max_length = max_length
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     def forward(self, inputs):
#         x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0]
#         return x
    
#     def tokenize(self, raw_tokens, pos_head, pos_tail):
#         tokens = []
#         cur_pos = 0
#         pos1_in_index = 0
#         pos2_in_index = 0
#         for token in raw_tokens:
#             token = token.lower()
#             if cur_pos == pos_head[0]:
#                 tokens.append('[unused0]')
#                 pos1_in_index = len(tokens)
#             if cur_pos == pos_tail[0]:
#                 tokens.append('[unused1]')
#                 pos2_in_index = len(tokens)
#             tokens += self.tokenizer.tokenize(token)
#             if cur_pos == pos_head[-1]:
#                 tokens.append('[unused2]')
#             if cur_pos == pos_tail[-1]:
#                 tokens.append('[unused3]')
#             cur_pos += 1
#         indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
#         return indexed_tokens






# class RobertaPAIRSentenceEncoder(nn.Module):

#     def __init__(self, pretrain_path, max_length): 
#         nn.Module.__init__(self)
#         self.roberta = RobertaForSequenceClassification.from_pretrained(
#                 pretrain_path,
#                 num_labels=2)
#         self.max_length = max_length
#         self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#         self.vocab_size = self.tokenizer.vocab_size 

#     def forward(self, inputs):
#         x = self.roberta(inputs['word'], attention_mask=inputs['mask'])[0]
#         return x
    
#     def tokenize(self, raw_tokens, pos_head, pos_tail):
#         def getIns(bped, bpeTokens, tokens, L):
#             resL = 0
#             tkL = " ".join(tokens[:L])
#             bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
#             if bped.find(bped_tkL) == 0:
#                 resL = len(bped_tkL.split())
#             else:
#                 tkL += " "
#                 bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
#                 if bped.find(bped_tkL) == 0:
#                     resL = len(bped_tkL.split())
#                 else:
#                     raise Exception("Cannot locate the position")
#             return resL

#         s = " ".join(raw_tokens)
#         sst = self.tokenizer.tokenize(s)
#         headL = pos_head[0]
#         headR = pos_head[-1] + 1
#         hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
#         hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

#         tailL = pos_tail[0]
#         tailR = pos_tail[-1] + 1
#         tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
#         tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

#         # E1b = 'madeupword0000'
#         # E1e = 'madeupword0001'
#         # E2b = 'madeupword0002'
#         # E2e = 'madeupword0003'
#         # E1b, E1e, E2b, E2e = self.tokenizer.convert_tokens_to_ids(['madeupword0000', 'madeupword0001', 'madeupword0002', 'madeupword0003'])
#         E1b = self.vocab_size   #'madeupword0000'
#         E1e = self.vocab_size+1 #'madeupword0001'
#         E2b = self.vocab_size+2 #'madeupword0002'
#         E2e = self.vocab_size+3 #'madeupword0003'

#         ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
#         ins = sorted(ins)

#         indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)

#         for i in range(0, 4):
#             # sst.insert(ins[i][0] + i, ins[i][1])
#             indexed_tokens.insert(ins[i][0] + i, ins[i][1])

#         return indexed_tokens, [10000, 10000], [10000, 10000]
        
    
    
    

# class RobertaEntPAIRSentenceEncoder(nn.Module):

#     def __init__(self, pretrain_path, max_length): 
#         nn.Module.__init__(self)
#         config = RobertaConfig.from_pretrained(pretrain_path)
#         config.num_labels = 2
#         # config.bert_model_name = 'roberta-base'
#         config.entity_emb_size = 768
#         config.save_on_gpu = False
#         print (pretrain_path)


#         self.roberta = RobertaEntForSequenceClassification.from_pretrained(pretrain_path, config=config) 

#         self.max_length = max_length
#         self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_path)
#         self.vocab_size = self.tokenizer.vocab_size 
#         self.max_mention_length = 30


#     def forward(self, inputs):
#         entity_attention_mask = inputs['entity_attention_mask']

#         outputs = self.roberta(input_ids=inputs['word'], attention_mask=inputs['mask'],  \
#                     entity_embeddings=inputs['entity_embeddings'], entity_attention_mask=entity_attention_mask, \
#                     entity_position_ids=inputs['entity_position_ids'] )
        
#         x = outputs[0]
#         return x
    
#     def tokenize(self, raw_tokens, pos_head, pos_tail):
#         def getIns(bped, bpeTokens, tokens, L):
#             resL = 0
#             tkL = " ".join(tokens[:L])
#             bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
#             if bped.find(bped_tkL) == 0:
#                 resL = len(bped_tkL.split())
#             else:
#                 tkL += " "
#                 bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
#                 if bped.find(bped_tkL) == 0:
#                     resL = len(bped_tkL.split())
#                 else:
#                     raise Exception("Cannot locate the position")
#             return resL

#         s = " ".join(raw_tokens)
#         sst = self.tokenizer.tokenize(s)
#         headL = pos_head[0]
#         headR = pos_head[-1] + 1
#         hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
#         hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

#         tailL = pos_tail[0]
#         tailR = pos_tail[-1] + 1
#         tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
#         tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

#         indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)

#         E1b = self.vocab_size   #'madeupword0000'
#         E1e = self.vocab_size+1 #'madeupword0001'
#         E2b = self.vocab_size+2 #'madeupword0002'
#         E2e = self.vocab_size+3 #'madeupword0003'

#         ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
#         ins = sorted(ins)
#         h_pos = [-1, -1]
#         t_pos = [-1, -1]
#         for i in range(0, 4):
#             indexed_tokens.insert(ins[i][0] + i, ins[i][1])
#             if ins[i][1]==E1b:
#                 h_pos[0] = ins[i][0] + i + 1
#             if ins[i][1]==E1e:
#                 h_pos[1] = ins[i][0] + i 

#             if ins[i][1]==E2b:
#                 t_pos[0] = ins[i][0] + i + 1
#             if ins[i][1]==E2e:
#                 t_pos[1] = ins[i][0] + i 

#         # return indexed_tokens, [10000, 10000], [10000, 10000]

#         return indexed_tokens, h_pos, t_pos

