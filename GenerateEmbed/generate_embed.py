import h5py
import numpy as np
import os
from transformers import RobertaTokenizer, AutoTokenizer
from tqdm import trange
import copy
import pickle


data_dir = '../wikipedia_data'

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

num_split = 1
dim = 768
prefix = 'wiki'


f = h5py.File(os.path.join(data_dir, 'all_instances_'+prefix+'.h5'), 'r')  
tot_idxs_list = []
for i in range(num_split):
    _samples = f['samples_'+str(i)]
    tot_idxs_list.append(_samples)
tot_idxs_cat = np.concatenate(tot_idxs_list, axis=0)

tot_embeddings_list = []
for rank in trange(num_split):
    all_embeds = np.memmap(filename= os.path.join(data_dir, 'samples_maskembed_'+str(rank)+'.memmap'), mode='r', dtype=np.float16)
    num_example = all_embeds.shape[0]//dim
    all_embeds = np.reshape(all_embeds, (num_example, dim))
    tot_embeddings_list.append(all_embeds)

pageid2id = {}
tot = 0 
for i in range(len(tot_idxs_cat)):
    page_id = tot_idxs_cat[i, -1]
    if page_id not in pageid2id:
        pageid2id[page_id] = tot
        tot += 1

print (tot)
pickle.dump(pageid2id, open(os.path.join(data_dir, 'pageid2id.pkl'), 'wb'))


all_embeds = np.memmap(filename= os.path.join(data_dir, 'tot_embed_mask.memmap'), shape=(tot, dim), mode='w+', dtype=np.float16)

t = 0
for tot_idxs, tot_embeddings in zip(tot_idxs_list, tot_embeddings_list):  
    last_i = 0

    for i in trange(len(tot_idxs)):
        if i+1==len(tot_idxs) or tot_idxs[i+1,-1]!=tot_idxs[i,-1]:
            ent_embed = np.mean(tot_embeddings[last_i: i+1], axis=0)
            all_embeds[t] = ent_embed

            t += 1
            last_i = i+1


