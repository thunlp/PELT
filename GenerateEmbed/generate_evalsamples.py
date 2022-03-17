import pickle
import h5py
import numpy as np
import os
from tqdm import tqdm, trange
import sys
from transformers import AutoTokenizer, RobertaTokenizer, BertTokenizer
from tqdm import tqdm
import pickle
from collections import defaultdict
working_dir = 'wiki_data/'
datasetname = 'wiki' 
K = 256      
num_split = 4 # for multiprocessing

tokenizer = RobertaTokenizer.from_pretrained("../../bert_models/roberta-base/", do_lower_case=False)


neg_samples = []
groups = pickle.load(open(os.path.join(working_dir, 'entity_pos.pkl'),  'rb'))

f = h5py.File(os.path.join(working_dir, 'input_ids.h5'), 'r')  
input_ids = f['input_ids'] 



samples = []
split = len(groups) // num_split + 1


for i in range(num_split):
    samples.append([])

tot = 0
pageid2freq = {}
pageid2embedid = {}
embedid = 0
print (len(groups))

groups = sorted(list(groups.items()))
for i, (page_id, group) in tqdm(enumerate(groups)):

    pageid2freq[page_id] = len(group)

    w = int(np.random.randint(num_split))
    if len(group)>K:
        group_idxs = np.random.choice(len(group), K, replace=False)
        _group = []
        for idx in group_idxs:
            _group.append(group[idx])
        group = _group

    if len(group) > 0:
        assert (page_id not in pageid2embedid)
        if i < 5:
            print ()
        for j in range(len(group)):


            cur_item = group[j]
            samples[w].append( (cur_item[0], cur_item[1], cur_item[2], embedid) )

            if i < 5 and j<5:
                tokens = tokenizer.convert_ids_to_tokens( input_ids[cur_item[0]][cur_item[1]:cur_item[2]] )
                name = tokenizer.convert_tokens_to_string(tokens)
                print (name, page_id)

        pageid2embedid[page_id] = embedid        
        embedid += 1

        assert(len(pageid2embedid) ==  embedid)

    tot += len(group)

print ('tot_sent:', tot)
print ('num_ent:', len(pageid2embedid), embedid, len(pageid2freq))
pickle.dump(pageid2embedid, open(os.path.join(working_dir, datasetname+'_pageid2embedid.pkl'), 'wb'))
pickle.dump(pageid2freq, open(os.path.join(working_dir, datasetname+'_pageid2freq.pkl'), 'wb'))


f = h5py.File(os.path.join(working_dir, datasetname+'_all_instances_'+str(K)+'.h5'), 'w')
for i in range(num_split):
    _samples = np.array(samples[i], dtype=np.int32)
    print (_samples.shape)
    f['samples_'+str(i)] = _samples
f.close()
print ('outputed')





