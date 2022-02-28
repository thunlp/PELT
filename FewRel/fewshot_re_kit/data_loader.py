import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import pickle

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name, med_fewrel=False, modL=None):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.modL = modL 
        self.use_entembed = ('ent' in encoder_name)
        self.dim = dim = 768

        if self.use_entembed:
            if name.find('wiki')!=-1:
                embed_path = '../wiki_roberta_fewrel'
                self.qid2id = pickle.load(open(os.path.join(embed_path, "qid2embedid.pkl"), 'rb'),)
                self.entity_embed = np.load(os.path.join(embed_path, 'wiki_entity_embed_256.npy'))
                # if self.modL>0:
                L = np.linalg.norm(self.entity_embed, axis=-1)
                self.entity_embed = self.entity_embed / np.expand_dims(L, -1) * self.modL

            else:
                embed_path = '../medembed_roberta_fewrel/'
                self.qid2id = pickle.load(open(embed_path + "qid2embedid.pkl", 'rb'),)
         
                self.entity_embed = np.load(os.path.join(embed_path, 'med_entity_embed_256.npy'))
                L = np.linalg.norm(self.entity_embed, axis=-1)
                self.entity_embed = self.entity_embed / np.expand_dims(L, -1) * self.modL

        else:
            self.qid2id = {}

   
        self.max_length = encoder.max_length
        self.max_entity_length = 2


    def __additem__(self, d, item):
        max_length = self.max_length

        input_ids, h_pos, t_pos = self.encoder.tokenize(item['tokens'], item['h'][2][0], item['t'][2][0])
        h_id = item['h'][1]
        t_id = item['t'][1]

        h_pos[0] += 1
        h_pos[1] += 1
        t_pos[0] += 1
        t_pos[1] += 1

        input_ids = input_ids[ : max_length - 2]
        input_ids = [self.encoder.tokenizer.cls_token_id] + input_ids + [self.encoder.tokenizer.sep_token_id] 
        # Zero-pad up to the sequence length.
        L = len(input_ids)
        padding_length = max_length - L

        input_ids += [self.encoder.tokenizer.pad_token_id] * padding_length

        attention_mask = [1] * L + [0] * padding_length

        attention_mask += [1] * self.max_entity_length

        e1_pos = h_pos[1] - 2
        e2_pos = t_pos[1] - 2

        if (h_id in self.qid2id and e1_pos < max_length-1) and self.use_entembed:
            h_id = self.qid2id[h_id]
            # h_embed = torch.tensor(np.array(self.entity_embed[h_id]), dtype=torch.float32)
            h_embed = torch.tensor(self.entity_embed[h_id], dtype=torch.float32)

            attention_mask[e1_pos] = 0
        else:
            h_embed = torch.ones((self.dim), dtype=torch.float32)
            attention_mask[max_length] = 0

        if (t_id in self.qid2id and e2_pos < max_length-1) and self.use_entembed:
            t_id = self.qid2id[t_id]
            # t_embed = torch.tensor(np.array(self.entity_embed[t_id]), dtype=torch.float32)
            t_embed = torch.tensor(self.entity_embed[t_id], dtype=torch.float32)
            attention_mask[e2_pos] = 0
        else:
            t_embed = torch.ones((self.dim), dtype=torch.float32)
            attention_mask[max_length+1] = 0

        entity_embeddings = torch.stack([h_embed, t_embed], dim=0)
        # if self.modL>0:
        #     L = torch.norm(entity_embeddings, dim=-1)
        #     entity_embeddings = entity_embeddings / L.unsqueeze(-1) * self.modL


        entity_position_ids = torch.tensor([min(e1_pos, max_length-1), min(e2_pos, max_length-1)])
        if 'roberta' in self.encoder_name:
            entity_position_ids += 2

        pos1 = min(h_pos[0], max_length-1)
        pos2 = min(t_pos[0], max_length-1)
        # pos1 = min(e1_pos, max_length-1)
        # pos2 = min(e2_pos, max_length-1)

        if not ('ent' in self.encoder_name):
            attention_mask = attention_mask[:max_length]
            

        d['word'].append(torch.tensor(input_ids))
        d['pos1'].append(torch.tensor(pos1, dtype=torch.int64))
        d['pos2'].append(torch.tensor(pos2, dtype=torch.int64))
        d['mask'].append(torch.tensor(attention_mask))
        d['entity_embeddings'].append(entity_embeddings)
        d['entity_position_ids'].append(entity_position_ids)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        # query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }

        support_set = {'word': [], 'mask': [],  'entity_embeddings': [],   'entity_position_ids': [],  'pos1': [], 'pos2': [] }
        query_set = {'word': [], 'mask': [],  'entity_embeddings': [],  'entity_position_ids': [],  'pos1': [], 'pos2': [] }

        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:

                if count < self.K:
                    self.__additem__(support_set,  self.json_data[class_name][j])

                else:
                    self.__additem__(query_set, self.json_data[class_name][j])

                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                    list(range(len(self.json_data[cur_class]))),
                    1, False)[0]
            self.__additem__(query_set, self.json_data[cur_class][index])

        query_label += [self.N] * Q_na

        return support_set, query_set, query_label
    
    def __len__(self):
        return 1000000000

def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [],  'entity_embeddings': [],  'entity_position_ids': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [],  'entity_embeddings': [],  'entity_position_ids': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label

def get_loader(name, encoder, N, K, Q, batch_size, 
        num_workers=4, collate_fn=collate_fn, na_rate=0, root='/home/yedeming/PELT/FewRel/data', encoder_name=None, med_fewrel=False, modL=6): # for debug
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root, encoder_name, med_fewrel=med_fewrel, modL=modL)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,  # for debug
            collate_fn=collate_fn) 
    return iter(data_loader)

class FewRelDatasetPair(data.Dataset):
    """
    FewRel Pair Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name, med_fewrel=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print (path)
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.max_length = encoder.max_length
        assert (False)
        if name.find('wiki')!=-1:
            self.qid2id = pickle.load(open("../wikipedia_data/qid2embedid.pkl", 'rb'))
            self.entity_embed = np.memmap('../wiki_fewrel_data/tot_embed_mask.memmap', mode='r',  dtype=np.float16)
        else:
            self.qid2id = pickle.load(open(os.path.join(root, "med_fewrel_qid2embedid.pkl"), 'rb'))
            self.entity_embed = np.memmap('../med_fewrel_data/tot_embed_mask.memmap', mode='r',  dtype=np.float16)

        self.dim = dim = 768
        num_example = self.entity_embed.shape[0]//dim
        self.entity_embed = np.reshape(self.entity_embed, (num_example, dim))
        self._max_mention_length = 30
        

    def __getraw__(self, item):
        word, h_pos, t_pos = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])

        h_id = item['h'][1]
        t_id = item['t'][1]

        entity_attention_mask = []

        if h_id in self.qid2id:
            h_id = self.qid2id[h_id]
            h_embed = torch.tensor(np.array(self.entity_embed[h_id]), dtype=torch.float32)
            entity_attention_mask.append(1)
        else:
            h_embed = torch.zeros((self.dim, ), dtype=torch.float32)
            entity_attention_mask.append(0)

        if t_id in self.qid2id:
            t_id = self.qid2id[t_id]
            t_embed = torch.tensor(np.array(self.entity_embed[t_id]), dtype=torch.float32)
            entity_attention_mask.append(1)
        else:
            t_embed = torch.zeros((self.dim, ), dtype=torch.float32)
            entity_attention_mask.append(0)
            
        entity_embed = torch.stack([h_embed, t_embed], dim=0)

        return word, entity_embed, entity_attention_mask, h_pos, t_pos

    def __additem__(self, d, word, pos1, pos2, mask):
        assert(False)
        # d['word'].append(word)
        # d['pos1'].append(pos1)
        # d['pos2'].append(pos2)
        # d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support = []
        query = []
        fusion_set = {'word': [], 'mask': [], 'seg': [], 'entity_embeddings': [], 'entity_attention_mask': [], \
                        'entity_segment_ids': [], 'entity_position_ids': []}

        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, ins_entity_embed, ins_attention_mask, ins_h_pos, ins_t_pos  = self.__getraw__(
                        self.json_data[class_name][j])
                if count < self.K:
                    support.append((word, ins_entity_embed, ins_attention_mask, ins_h_pos, ins_t_pos))
                else:
                    query.append((word, ins_entity_embed, ins_attention_mask, ins_h_pos, ins_t_pos))
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                    list(range(len(self.json_data[cur_class]))),
                    1, False)[0]
            word, ins_entity_embed, ins_attention_mask, ins_h_pos, ins_t_pos = self.__getraw__(
                    self.json_data[cur_class][index])
            query.append((word, ins_entity_embed, ins_attention_mask, ins_h_pos, ins_t_pos))
        query_label += [self.N] * Q_na

        for word_query, ins_entity_embed_query, ins_attention_mask_query, ins_h_pos_query, ins_t_pos_query in query:
            for word_support, ins_entity_embed_support, ins_attention_mask_support, ins_h_pos_support, ins_t_pos_support in support:
                if self.encoder_name == 'bert':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                    word_tensor = torch.zeros((self.max_length)).long()
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])     
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])
                    word_tensor = torch.ones((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query 
                new_word = new_word[ : self.max_length-1] + SEP        # fix bug
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0

                entity_embeddings = torch.cat([ins_entity_embed_query, ins_entity_embed_support], axis=0) 
                entity_attention_mask = torch.tensor(ins_attention_mask_query+ins_attention_mask_support).long()
                entity_segment_ids = torch.tensor( (0, 0, 1, 1),  dtype=torch.int64)


                entity_position_ids = -torch.ones((4 , self._max_mention_length), dtype=torch.int64)

                for i, (start, end) in enumerate([ins_h_pos_query, ins_t_pos_query, ins_h_pos_support, ins_t_pos_support]):
                    if i < 2:
                        dL = len(word_support)+2     
                    else: 
                        dL = 1                       
                    start += dL
                    end += dL
                    end = min(end, self.max_length)

                    if end - start > 0 and start-dL!=-1 and end-dL!=-1:
                        for j in range(start, end):
                            entity_position_ids[i, j - start] = j
                    else:
                        entity_attention_mask[i] = 0


                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)
                
                fusion_set['entity_embeddings'].append(entity_embeddings)
                fusion_set['entity_attention_mask'].append(entity_attention_mask)
                fusion_set['entity_segment_ids'].append(entity_segment_ids)
                fusion_set['entity_position_ids'].append(entity_position_ids)

        return fusion_set, query_label
    
    def __len__(self):
        return 1000000000

def collate_fn_pair(data):
    batch_set = {'word': [], 'seg': [], 'mask': [], 'entity_embeddings': [], 'entity_attention_mask': [], \
                        'entity_segment_ids': [], 'entity_position_ids': []}


    batch_label = []
    fusion_sets, query_labels = zip(*data)

    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_set, batch_label

def get_loader_pair(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_pair, na_rate=0, root='data', encoder_name='bert', med_fewrel=False):
    dataset = FewRelDatasetPair(name, encoder, N, K, Q, na_rate, root, encoder_name, med_fewrel=med_fewrel)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

class FewRelUnsupervisedDataset(data.Dataset):
    """
    FewRel Unsupervised Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        total = self.N * self.K
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }

        indices = np.random.choice(list(range(len(self.json_data))), total, False)
        for j in indices:
            word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[j])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(support_set, word, pos1, pos2, mask)

        return support_set
    
    def __len__(self):
        return 1000000000

def collate_fn_unsupervised(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    return batch_support

def get_loader_unsupervised(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_unsupervised, na_rate=0, root='./data'):
    dataset = FewRelUnsupervisedDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


