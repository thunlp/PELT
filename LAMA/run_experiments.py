# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from lama.modules import build_model_by_name
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict
import pickle
import sys


working_dir = '/home/yedeming/PELT/LAMA/'
if int(sys.argv[1])>0:
    suffix = '_UHN'
else:
    suffix = ''

LMs = [
   
    # {
    #     "lm": "ourroberta",  # for huggface format checkpoint
    #     "label": "roberta_base",
    #     "models_names": ["ourroberta"],
    #     "roberta_model_dir": "./roberta.base",
    #     "roberta_model_name": "model.pt",
    #     "roberta_vocab_name": "dict.txt",
    #     "luke_model_dir": "../../bert_models/roberta-base",

    # },

    {
        "lm": "robertaconcat",
        "label": "roberaent_base",
        "models_names": ["robertaconcat"],
        "roberta_model_name": "model.pt",
        "roberta_vocab_name": "dict.txt",
        "roberta_model_dir": working_dir+"roberta.base",
        "luke_model_dir": "../../bert_models/roberta-base",
        'modL': float(sys.argv[2]),
    },

 
    # {
    #     "lm": "bert",
    #     "label": "bert-base-uncased",
    #     "models_names": ["bert"],
    #     "bert_model_name": "bert-base-cased",
    #     "bert_model_dir": "/data3/private/yedeming/bert_models/bert-base-uncased",
    # },


    # {
    #     "lm": "bertconcat",
    #     "label": "bert-base-uncased",
    #     "models_names": ["bertconcat"],
    #     "model_name_or_path": "/data3/private/yedeming/bert_models/bert-base-uncased", 
    #     'modL': float(sys.argv[2]),
    # },


    # {
    #     "lm": "bart",  # for huggface format checkpoint
    #     "label": "bart_base",
    #     "models_names": ["bart"],
    #     # "roberta_model_dir": "/data/private/yedeming/PELT/LAMA/roberta.base",
    #     "roberta_model_dir": "/data3/private/yedeming/PELT_remote/LAMA/roberta.base",
    #     "roberta_model_name": "model.pt",
    #     "roberta_vocab_name": "dict.txt",
    #     # "luke_model_dir": "/data/private/yedeming/bert_models/bart-base",
    #     "luke_model_dir": "/data3/private/yedeming/bert_models/bart-base",

    # },

    # {
    #     "lm": "bartconcat",   
    #     "label": "bart_base",
    #     "models_names": ["bartconcat"],
    #     # "roberta_model_dir": "/home/yedeming/PELT/LAMA/roberta.base",
    #     "roberta_model_dir": "/data3/private/yedeming/PELT_remote/LAMA/roberta.base",
    #     "roberta_model_name": "model.pt",
    #     "roberta_vocab_name": "dict.txt",
    #     # "luke_model_dir": "/home/yedeming/bert_models/bart-base",
    #     "luke_model_dir": "/data3/private/yedeming/bert_models/bart-base",
    #     'modL': float(sys.argv[2]),
    # },

    # {
    #     "lm": "ernie",  # for huggface format checkpoint
    #     "label": "ernie",
    #     "models_names": ["ernie"],
    #     "roberta_model_dir": working_dir+"roberta.base",
    #     "roberta_model_name": "model.pt",
    #     "roberta_vocab_name": "dict.txt",
    #     "model_name_or_path": "/data3/private/ydm_tmp/bert_models/roberta-ernie/",
    # },


    # {
    #     "lm": "luke",
    #     "label": "luke_base",
    #     "models_names": ["luke"],
    #     "roberta_model_name": "model.pt",
    #     "roberta_vocab_name": "dict.txt",
    #     "roberta_model_dir": "/home/yedeming/PELT/LAMA/roberta.base/",
    #     "luke_model_dir": "/home/yedeming/bert_models/luke-base/",
    # },

]


def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bertlm"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    results_file = open(working_dir+"last_results.csv", "w+")
    all_ent_results = {}
    print ('start')
    if 'uncased' in input_param['label']:
        lowercase = True
        common_vocab_filename = working_dir+"common_vocab_lowercased.txt"
        print ('uncased')
    else:
        lowercase = False
        common_vocab_filename = working_dir+"common_vocab_cased.txt"
    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": common_vocab_filename,
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 64,
            "logdir": "output",
            "full_logdir": working_dir+"output/results/{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": lowercase,
            "max_sentence_length": 128,
            "threads": -1,
            "interactive": False,#True,
            "use_negated_probes": use_negated_probes,
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]
            if use_negated_probes:
                PARAMETERS["template_negated"] = relation["template_negated"]

        PARAMETERS.update(input_param)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        Precision1, ent_result = run_evaluation(args, shuffle_data=False, model=model)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)
        for k, v in ent_result.items():
            if k not in all_ent_results:
                all_ent_results[k] = v
            else:
                all_ent_results[k][0] += v[0]
                all_ent_results[k][1] += v[1]

        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ {} - mean P@1: {}".format(input_param["label"], mean_p1))
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    # record the evaluation in entity level
    # pickle.dump(all_ent_results, open('pelt_ent_results.pkl', "wb"))
    # pickle.dump(all_ent_results, open('roberta_ent_results.pkl', "wb"))
    # pickle.dump(all_ent_results, open('kepler_ent_results.pkl', "wb"))

    return mean_p1, all_Precision1


def get_TREx_parameters():
    data_path_pre = working_dir+'data/'
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    global suffix
    data_path_pre += "TREx"+suffix+"/" 
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    global suffix
    data_path_pre = working_dir+"data/Google_RE"+suffix+"/" 
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_path_pre='data/'):
    relations = [{"relation": "test"}]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_path_pre='data/'):
    relations = [{"relation": "test"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters):
    for ip in LMs:
        print(ip["label"])
        mean_p1, all_Precision1 = run_experiments(*parameters, input_param=ip, use_negated_probes=False)

        return mean_p1
        
if __name__ == "__main__":

    print("1. Google-RE")
    parameters = get_GoogleRE_parameters()
    mean_p1_G = run_all_LMs(parameters)

    print("2. T-REx")
    parameters = get_TREx_parameters()
    mean_p1_T = run_all_LMs(parameters)

    print (mean_p1_G, mean_p1_T)


