import json
import os
from collections import defaultdict, Counter
from tqdm import tqdm
import random
import copy
import pickle
import numpy as np
from transformers import RobertaTokenizer, BertTokenizer, AutoTokenizer
import random
import h5py
import sys
import transformers

data_dir = 'wiki_ids_data/'

is_bert = False
if not is_bert:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
else:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, use_fast=True)


files = sorted(os.listdir(data_dir))
files.sort()

max_seq_length = 64 

input_ids_memap = []
entity_pos = defaultdict(list)

num_sent = 0
for file in files:
    num = int(file[4:])
    
    print(file, num)
    with open(os.path.join(data_dir, file)) as f:
        for line in tqdm(f):
            sent = json.loads(line.strip())
            input_ids = sent['input_ids']
            a_list = sent['a_list']
            L = len(input_ids)
            if L<=max_seq_length and L>4: 
                flag = False
                for x in a_list:
                    page_id = int(x[0])
                    entity_pos[page_id].append((num_sent, x[1]+1, x[2]+1)) 
                    flag = True
                if flag:
                    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id] 
                    input_ids = input_ids + [tokenizer.pad_token_id] * (max_seq_length-len(input_ids))

                    input_ids_memap.append(input_ids)
                    if num_sent<2:
                        for x in a_list:
                            page_id = int(x[0])
                            a, b, c = entity_pos[page_id][-1]
                            check_ids = input_ids_memap[a]
                            print (x[0], tokenizer.convert_ids_to_tokens( check_ids[b:c]) )

                    num_sent += 1 

print (num_sent)

output_dir = 'wiki_data'
os.makedirs(output_dir, exist_ok=True)
f = h5py.File(os.path.join(output_dir, 'input_ids.h5'), 'w')       
f['input_ids'] = np.array(input_ids_memap, dtype=np.int32)
f.close()     
print ('finish 1')
pickle.dump(entity_pos, open(os.path.join(output_dir, 'entity_pos.pkl'),'wb'))
print ('finish 2')

