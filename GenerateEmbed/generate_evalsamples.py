import pickle
import h5py
import numpy as np
import os
from tqdm import tqdm, trange
import sys
from transformers import AutoTokenizer
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("../bert_models/roberta-base-unk4")

working_dir = '../wikipedia_data'
prefix = 'wiki'
neg_samples = []
groups = pickle.load(open(os.path.join(working_dir, 'entity_pos.pkl'),  'rb'))


f = h5py.File(os.path.join(working_dir, 'input_ids.h5'), 'r')  
input_ids = f['input_ids'] 


num_split = 1 # for multiprocessing
K = 256       # maxinum setences for an entity
split = len(groups) // num_split + 1

samples = []


for i in range(num_split):
    samples.append([])

tot = 0
pageid2freq = {}
for i, (page_id, group) in tqdm(enumerate(groups.items())):
    pageid2freq[page_id] = len(group)

    w = np.random.randint(num_split)
    if len(group)>K:
        group_idxs = np.random.choice(len(group), K, replace=False)
        _group = []
        for idx in group_idxs:
            _group.append(group[idx])
        group = _group

    for j in range(len(group)):
        cur_item = group[j]
        samples[w].append((cur_item[0], cur_item[1], cur_item[2], page_id))

    tot += len(group)

print (tot)
pickle.dump(pageid2freq, open(os.path.join(working_dir, 'pageid2freq.pkl'), 'wb'))


f = h5py.File(os.path.join(working_dir, 'all_instances_'+prefix+'.h5'), 'w')
for i in range(num_split):
    _samples = np.array(samples[i], dtype=np.int32)
    print (_samples.shape)
    f['samples_'+str(i)] = _samples

f.close()

