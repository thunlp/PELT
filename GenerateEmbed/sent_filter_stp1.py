import json
import os
from collections import defaultdict, Counter
from tqdm import tqdm
import random
import copy
import urllib.parse
import pickle
import numpy as np
from transformers import RobertaTokenizer, BertTokenizer, AutoTokenizer
import random
import h5py
import sys
import transformers

wiki_path = 'article_further_links/'
output_dir = 'wiki_ids_data/'
os.makedirs(output_dir, exist_ok=True)

is_bert = False
if not is_bert:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    vocab_size = tokenizer.vocab_size
else:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, use_fast=True)

files = sorted(os.listdir(wiki_path)) 
files.sort()

max_seq_length = 64

num_sent = 0
debug = False

for file in files:
    num = int(file[4:])
    print(file, num)

    output_name = output_dir + '/hms_'  +str(num)
    if not debug:
        if os.path.exists(output_name):
            continue
        w = open(output_name,"w")

    with open(os.path.join(wiki_path,  file)) as f:
        for line in tqdm(f):            
            article = json.loads(line.strip())
            all_doc_tokens = []
            all_alist = []
            for para in article['paras_info']:
                for sent in para:
                    context = sent['context']
                    words = sent['a_list']

                    if len(words)>0:
                        r = tokenizer._tokenizer.encode(context)
                        tokens = r.tokens[1:-1]
                        offset = r.offsets[1:-1]  
                        input_ids = r.ids[1:-1]   
                        char_to_word_offset = {}
                        for idx, (s, e) in enumerate(offset):
                            for x in range(s,e):
                                char_to_word_offset[x] = idx 

                        rare_list = []

                        for word_id, start, end, name, _ in words:
                            if start in char_to_word_offset and end-1 in char_to_word_offset:                            
                                start = char_to_word_offset[start]
                                end = char_to_word_offset[end-1]+1
                                rare_list.append( ( word_id, start, end, name) )
                        rare_list = list(set(rare_list))   # duplicate removal, an entity may be linked by several methods
                        if len(input_ids)+2<= max_seq_length and len(input_ids)>=8 and len(rare_list)>0:
                            if not debug:
                                item = {'input_ids':input_ids, 'a_list': rare_list}
                                w.write(json.dumps(item)+'\n')
                                
                            num_sent += 1


print (num_sent)




