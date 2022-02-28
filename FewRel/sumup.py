import json
import numpy as np
prefix = 'results/' + 'roberta-base-robertaent_test_wiki_5_1_1_'
suffix = '_1500_7.txt'

results = []
for i in range(42,47):
    try:
        filename = prefix+str(i)+suffix
        data = open(filename).read()
        prec = float(data.strip())
        print (filename, prec)
        results.append(prec)
    except:
        pass
results = np.array(results)
print (np.mean(results))
print (np.std(results))