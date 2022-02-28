import pickle
import json
# a = pickle.load(open(''))

qid2pageid = json.load(open('qid2pageid.json'))
pageid2freq = pickle.load(open('/wiki_pageid2freq.pkl', 'rb'))

all_ent_results = pickle.load(open('pelt_ent_results.pkl', "rb"))

results = [[0,0], [0,0], [0,0], [0,0]] #, [0,0]
for k, v in all_ent_results.items():
    assert(v[0]<=v[1])
    if k in qid2pageid:
        pageid = qid2pageid[k]
        freq = pageid2freq.get(pageid, -1)
        if freq<10:
            results[0][0]+=v[0]
            results[0][1]+=v[1]
        elif freq<50:
            results[1][0]+=v[0]
            results[1][1]+=v[1]
        elif freq<100:
            results[2][0]+=v[0]
            results[2][1]+=v[1]
        else:
            results[3][0]+=v[0]
            results[3][1]+=v[1]

for i in range(len(results)):
    print (results[i][0]/results[i][1])

