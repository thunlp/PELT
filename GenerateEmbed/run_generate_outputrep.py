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
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                                  DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer, 
                                  RobertaForRep, BertForRep)


import json
import numpy as np
import pickle
import h5py

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'bertrep': (BertConfig, BertForRep, BertTokenizer),
    'robertarep': (RobertaConfig, RobertaForRep, RobertaTokenizer),
}


class TextDataset(Dataset):
    def __init__(self, tokenizer, working_dir='train', max_seq_length=64, args=None):

        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

        self.args = args
        logger.info("Loading features from cached file %s", working_dir)

        f = h5py.File(os.path.join(working_dir, 'input_ids.h5'), 'r')  
        self.input_ids = f['input_ids'] 


        f = h5py.File(os.path.join(working_dir, 'all_instances_'+args.sample_id+'.h5'), 'r')  
        self.cur_samples = f['samples_'+str(args.process_idx)] 

        self.num_example  = self.cur_samples.shape[0]
            

    def __len__(self):
        return self.num_example
    
    def process(self, item):
        idx, s, e, _ = item
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.int64)


        position_ids = torch.ones((self.max_seq_length, ), dtype=torch.int64)
        position_ids[0] = 0
        entity_l = e-s
        position_ids[e] += -entity_l+1
        position_ids = torch.cumsum(position_ids, dim=0)

        input_ids[s] = self.tokenizer.mask_token_id
        if s+1<e:
            input_ids[s+1:e] = self.tokenizer.pad_token_id


        input_mask = (input_ids!=self.tokenizer.pad_token_id).long()
        return input_ids, input_mask, torch.tensor([s,e], dtype=torch.int64), position_ids

    def __getitem__(self, idx):
        cur_item = self.cur_samples[idx]
        cur_input_ids, cur_input_mask, boundary, cur_postion_ids = self.process(cur_item)

        return (cur_input_ids, cur_input_mask, boundary, cur_postion_ids)



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

def evaluate(args, eval_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) #if args.local_rank == -1 else DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size, num_workers=8,  pin_memory=True)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model  = amp.initialize(model, opt_level=args.fp16_opt_level)
    
    model.eval()
    num_example = len(eval_dataset) 

    all_embeds = np.memmap(filename= os.path.join(args.data_dir, 'samples_maskembed_'+str(args.process_idx)+'.memmap'), mode='w+', dtype=np.float16, shape=(num_example, 768))

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    cnt = 0
    for step, batch in enumerate(epoch_iterator):
        input_ids, input_mask, boundary, position_ids =  batch 
        bsz = input_ids.shape[0]

        with torch.no_grad():
            outputs = model(input_ids=input_ids.to(args.device), attention_mask=input_mask.to(args.device), position_ids=position_ids.to(args.device))

        rep = outputs[0].detach()

        for i in range(rep.shape[0]):
            s, e = boundary[i]
            embed = rep[i, s].cpu().numpy()
            all_embeds[cnt+i] = embed

        outputs = None
        cnt += bsz

    print ('closed')

    

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--model_type", default="deskepeler", type=str,
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

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
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
 
    parser.add_argument("--process_idx", type=int, default=-1, help="")
    parser.add_argument("--sample_id", default='wiki', type=str,  help="")



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
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

    eval_dataset = TextDataset(tokenizer, working_dir=args.data_dir, max_seq_length=args.max_seq_length, args=args)
    evaluate(args, eval_dataset, model, tokenizer)


if __name__ == "__main__":
    main()