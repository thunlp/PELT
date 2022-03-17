from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  BertConfig, BertForMaskedLM, BertTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                                  RobertaForRep, BertForRep)


import json
import numpy as np
import pickle
import h5py

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForRep, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForRep, RobertaTokenizer),
}


class TextDataset(Dataset):
    def __init__(self, tokenizer, args):

        self.max_seq_length = args.max_seq_length
        self.tokenizer = tokenizer

        self.args = args
        logger.info("Loading features from cached file %s", args.data_dir)

        f1 = h5py.File(os.path.join(args.data_dir, 'input_ids'+args.suffix+'.h5'), 'r')  
        self.input_ids = f1['input_ids'] 

        f2 = h5py.File(os.path.join(args.data_dir, args.datasetname+'_all_instances_'+str(args.K)+args.suffix+'.h5'), 'r')  
        self.cur_samples = f2['samples_'+str(args.sample_id)] 

        self.pageid2embedid = pickle.load(open(os.path.join(args.data_dir, args.datasetname+'_pageid2embedid'+args.suffix+'.pkl'), 'rb'))

        self.num_example  = self.cur_samples.shape[0]

    def __len__(self):
        return self.num_example
    

    def __getitem__(self, index):
        idx, s, e, embed_id = self.cur_samples[index]
        assert (s < e)
        input_ids = self.input_ids[idx].tolist()
        left = input_ids[:s]
        right = input_ids[e:]
        entity_name = input_ids[s:e]

        input_ids = left 
        mask_position = len(input_ids)
        input_ids += [self.tokenizer.mask_token_id] + right #+ [self.tokenizer.sep_token_id]

        pad_len = self.max_seq_length-len(input_ids)

        input_ids = input_ids[:self.max_seq_length]
        input_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len

        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        input_mask = torch.tensor(input_mask, dtype=torch.int64)

        return [input_ids, input_mask, mask_position, embed_id]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(args, model, tokenizer, prefix=''):
    """ Train the model """
    eval_dataset = TextDataset(tokenizer, args)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1, pin_memory=True)

    if args.fp16:
        model.half()

    model.eval()

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    num_ent = len(eval_dataset.pageid2embedid)
    logger.info("  Num Ent = %d", num_ent)
    all_embeds = np.zeros((num_ent, args.hidden_size),  dtype=np.float32)

    epoch_iterator = tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        embed_ids = batch[-1]
        batch = tuple(t.to(args.device) for t in batch[:-1])

        with torch.no_grad():

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'mask_position': batch[2],
                    }

            outputs = model(**inputs)

        rep = outputs[0].detach().cpu().numpy()


        for i in range(len(embed_ids)):
            embed_id = embed_ids[i]
            if embed_id>=0:
                all_embeds[embed_id] += rep[i]

        outputs = None

    np.save(os.path.join(args.data_dir, args.datasetname+'_entity_embed_'+str(args.K)+'.npy'), all_embeds)

    

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input training data file (a text file).")

    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")


    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)") 

    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--epoch", default=0, type=int,  help="")

    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=10000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=2,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
 
    parser.add_argument("--sample_id", default=0, type=int,  help="")
    parser.add_argument('--num_l_prompt', type=int, default=1)
    parser.add_argument('--num_m_prompt', type=int, default=2)
    parser.add_argument('--num_r_prompt', type=int, default=1)
    parser.add_argument('--lminit', action='store_true')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--K', type=int, default=256)
    parser.add_argument('--datasetname', type=str, default='wiki')



    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)


    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

    args.hidden_size = config.hidden_size
    args.vocab_size  = config.vocab_size

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache


    if args.local_rank == 0:
        torch.distributed.barrier()

    evaluate(args, model, tokenizer)


    if args.local_rank == 0:
        torch.distributed.barrier()

    # Evaluation
    results = {}

    return results


if __name__ == "__main__":
    main()
    
    