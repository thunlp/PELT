# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
from collections import defaultdict
import re
import shutil

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaTokenizer,
                                  RobertaEntForMarkerSequenceClassification,
                                  get_linear_schedule_with_warmup,
                                  AdamW
                                  )


from transformers import AutoTokenizer

from torch.utils.data import TensorDataset, Dataset
import json
import pickle
import numpy as np
import unicodedata

import time

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,  RobertaConfig)), ())

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaEntForMarkerSequenceClassification, RobertaTokenizer),
}



def _is_subword(token, tokenizer):
    if isinstance(tokenizer, RobertaTokenizer):
        token = tokenizer.convert_tokens_to_string(token)
        if not token.startswith(" ") and not _is_punctuation(token[0]):
            return True
    elif token.startswith("##"):
        return True

    return False


def _is_punctuation(char):
    # obtained from:
    # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def _normalize_mention(text):
    return " ".join(text.split(" ")).strip()

class Wiki80Dataset(Dataset):
    def __init__(self, tokenizer, args=None, evaluate=False, do_test=False):
        if evaluate:
            if do_test:
                file_path = 'test.jsonl'
            else:
                file_path = 'dev.jsonl'
        else:
            file_path = args.train_file

        file_path = os.path.join(args.data_dir, file_path)
        assert os.path.isfile(file_path)

        self.data_json = []
        with open(file_path, "r", encoding='utf-8') as f:
            print('reading file:', file_path)
            for line in f:
                self.data_json.append(json.loads(line))
            print('done reading')
        self.label2id = json.load(open(os.path.join(args.data_dir, 'label2id.json')))

        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size 
        self.max_seq_length = args.max_seq_length
        self.max_entity_length = 2
        self.args = args
        self.m1 = self.args.m1
        self.m2 = self.args.m2
        self.m3 = self.args.m3
        self.m4 = self.args.m4
        self.mask_id = self.tokenizer.mask_token_id
        self.l_bracket_id = self.args.l_bracket_id 
        self.r_bracket_id = self.args.r_bracket_id 

        self.mask_emb = args.mask_emb

        self.postion_plus = int(args.model_type.find('roberta')!=-1) * 2
        if not args.noentembed:
            embed_path = args.embed_path
            self.qid2id = pickle.load(open(os.path.join(embed_path, "qid2embedid.pkl"), 'rb'),)
            self.tot_entity_embed = np.load(os.path.join(embed_path, 'wiki_entity_embed_256.npy'))
            L = np.linalg.norm(self.tot_entity_embed, axis=-1)
            self.tot_entity_embed = self.tot_entity_embed / np.expand_dims(L, -1) * self.args.modL
        else:
            self.qid2id = {}

        print ('Loaded data')

    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        entry = self.data_json[idx]
        item = self.one_example_to_tensors(entry)
        return item

    def tokenize(self, words, pos_head, pos_tail):
  
        subwords_idx = [0]
        tokens = []
        for i in range(len(words)):
            word = words[i]
            if i>0 and self.args.model_type.find('roberta')!=-1:
                word = ' ' + word
            tokens.extend(self.tokenizer.tokenize(word))
            subwords_idx.append(len(tokens))

        hiL = subwords_idx[pos_head[0]]
        hiR = subwords_idx[pos_head[1]] 

        tiL = subwords_idx[pos_tail[0]]
        tiR = subwords_idx[pos_tail[1]]

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        ins = [(hiL, [self.m1]), (hiR, [self.l_bracket_id, self.mask_id, self.r_bracket_id, self.m2]), (tiL, [self.m3]), (tiR, [self.l_bracket_id, self.mask_id, self.r_bracket_id, self.m4])]

        ins = sorted(ins)
        h_pos = [-1, -1]
        t_pos = [-1, -1]
        num_inserted = 0
        for i in range(0, 4):
            insert_pos = ins[i][0] + num_inserted
            indexed_tokens = indexed_tokens[:insert_pos] + ins[i][1] + indexed_tokens[insert_pos:]
            if ins[i][0]==hiL:
                h_pos[0] = insert_pos
            if ins[i][0]==hiR:
                h_pos[1] = insert_pos

            if ins[i][0]==tiL:
                t_pos[0] = insert_pos
            if ins[i][0]==tiR:
                t_pos[1] = insert_pos

            num_inserted += len(ins[i][1])
        assert(h_pos[0]>=0 and h_pos[1]>=0 and t_pos[0]>=0 and t_pos[1]>=0)
        h_pos[1] += 3
        t_pos[1] += 3
        
        return indexed_tokens, h_pos, t_pos

    def one_example_to_tensors(self, example):
        max_length = self.max_seq_length
        input_ids, h_pos, t_pos = self.tokenize(example['token'], example['h']['pos'], example['t']['pos'])

        h_id = example['h']['id']
        t_id = example['t']['id']
        h_pos[0] += 1
        h_pos[1] += 1
        t_pos[0] += 1
        t_pos[1] += 1

        input_ids = input_ids[ : max_length - 2]
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id] 
        # Zero-pad up to the sequence length.
        L = len(input_ids)
        padding_length = max_length - L

        input_ids += [self.tokenizer.pad_token_id] * padding_length

        attention_mask = [1] * L + [0] * padding_length
        attention_mask += [1]* self.max_entity_length

        e1_pos = h_pos[1] - 2
        e2_pos = t_pos[1] - 2
        if (h_id in self.qid2id and e1_pos < max_length-1) and (not self.args.noentembed):
            h_id = self.qid2id[h_id]
            h_embed = torch.tensor(np.array(self.tot_entity_embed[h_id]), dtype=torch.float32)
            attention_mask[e1_pos] = 0
        else:
            h_embed = self.args.mask_emb
            attention_mask[max_length] = 0

        if (t_id in self.qid2id and e2_pos < max_length-1) and (not self.args.noentembed):
            t_id = self.qid2id[t_id]
            t_embed = torch.tensor(np.array(self.tot_entity_embed[t_id]), dtype=torch.float32)
            attention_mask[e2_pos] = 0
        else:
            t_embed = self.args.mask_emb 
            attention_mask[max_length+1] = 0

        entity_embeddings = torch.stack([h_embed, t_embed], dim=0)
        entity_position_ids = torch.tensor([min(e1_pos, max_length-1), min(e2_pos, max_length-1)])

        entity_position_ids += self.postion_plus

        label = self.label2id[example['relation']]
        item = [torch.tensor(input_ids),  
                torch.tensor(attention_mask), 
                torch.tensor([h_pos[0], t_pos[0]]),
                entity_embeddings, 
                entity_position_ids,
                torch.tensor(label),
                ]

        return item
 



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def train(args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/wiki80_log/"+args.output_dir[args.output_dir.rfind('/'):])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = Wiki80Dataset(tokenizer=tokenizer, args=args)
                 
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=int(args.output_dir.find('test')==-1)*4)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_f1 = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'ht_position':       batch[2], 
                      'entity_embeddings':       batch[3], 
                      'entity_position_ids':    batch[4],
                      'labels':         batch[5]
                    }
 

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)

                        f1 = results['f1']
                        if f1 > best_f1:
                            checkpoint_prefix = 'checkpoint'

                            best_f1 = f1
                            print ('Best F1', best_f1)
                            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

                            model_to_save.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            logger.info("Saving model checkpoint to %s", output_dir)
                            _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_f1


def evaluate(args, model, tokenizer, prefix="", do_test=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    results = {}

    eval_dataset = Wiki80Dataset(tokenizer=tokenizer, args=args, evaluate=True, do_test=do_test)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=int(args.output_dir.find('test')==-1)*4)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    preds = []
    out_label_ids = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'ht_position':       batch[2], 
                      'entity_embeddings':       batch[3], 
                      'entity_position_ids':    batch[4],
                    }
 
            outputs = model(**inputs)
            logits = outputs[0]
            pred = torch.argmax(logits, axis=1)
            preds.append(pred.detach().cpu().numpy())
            out_label_ids.append(batch[5].detach().cpu().numpy())


    preds = np.concatenate(preds, axis=0)
    out_label_ids = np.concatenate(out_label_ids, axis=0)

    acc = np.sum(preds==out_label_ids) / preds.shape[0]
    results = {'f1':  acc}

    return results




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
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
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
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
    parser.add_argument('--save_total_limit', type=int, default=1,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument( "--train_file",  default="train.jsonl", type=str)
    parser.add_argument( "--dev_file",  default="dev.jsonl", type=str)
    parser.add_argument( "--test_file",  default="test.jsonl", type=str)
    parser.add_argument( "--embed_path",  default="../wiki_roberta_fewrel/", type=str)

    parser.add_argument('--noentembed', action='store_true')
    parser.add_argument("--output_dropout_prob", default=0.1, type=float)
    parser.add_argument("--modL", default=7, type=float)

    args = parser.parse_args()
    args.crossatt = True

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

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
    num_labels = 80

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels)

    config.output_dropout_prob = args.output_dropout_prob

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    args.hidden_size = config.hidden_size
    logger.info("Training/evaluation parameters %s", args)

    if args.model_type.startswith('roberta'):
        spliter_id = tokenizer.encode(' /', add_special_tokens=False)
        assert(len(spliter_id)==1)
        args.spliter_id = spliter_id[0]
        
        l_bracket_id = tokenizer.encode(' (', add_special_tokens=False)
        assert(len(l_bracket_id)==1)
        args.l_bracket_id = l_bracket_id[0]

        r_bracket_id = tokenizer.encode(' )', add_special_tokens=False)
        assert(len(r_bracket_id)==1)
        args.r_bracket_id = r_bracket_id[0]

        mask_id = tokenizer.encode('<mask>', add_special_tokens=False)
        assert(len(mask_id)==1)
        mask_id = mask_id[0]

        word_embeddings = model.roberta.embeddings.word_embeddings.weight.data

        args.mask_id = mask_id
        args.mask_emb = word_embeddings[mask_id].clone()
    else:
        assert (False)



    if args.model_type.startswith('roberta') and tokenizer.convert_tokens_to_ids('[unused0]')==tokenizer.unk_token_id:
        special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.roberta.resize_token_embeddings(len(tokenizer))
        args.m1, args.m2, args.m3, args.m4 = tokenizer.convert_tokens_to_ids(['[unused0]', '[unused1]', '[unused2]', '[unused3]'])

    if args.model_type.startswith('roberta'):
        model.roberta.entity_embeddings.token_type_embeddings.weight.data.copy_(model.roberta.embeddings.token_type_embeddings.weight.data)
        model.roberta.entity_embeddings.LayerNorm.weight.data.copy_(model.roberta.embeddings.LayerNorm.weight.data)
        model.roberta.entity_embeddings.LayerNorm.bias.data.copy_(model.roberta.embeddings.LayerNorm.bias.data)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    best_f1 = 0
    # Training
    if args.do_train:

        global_step, tr_loss, best_f1 = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            results = evaluate(args, model, tokenizer)
            f1 = results['f1']
            if f1 > best_f1:
                best_f1 = f1
                print ('Best F1', best_f1)
                checkpoint_prefix = 'checkpoint'
                output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

                model_to_save.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)
                _rotate_checkpoints(args, checkpoint_prefix)

        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


    # Evaluation
    results = {'dev_best_f1': best_f1}
    if args.do_eval and args.local_rank in [-1, 0]:


        WEIGHTS_NAME = 'pytorch_model.bin'

        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

            model = model_class.from_pretrained(checkpoint)

            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step, do_test=True)

            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "results.json")
        json.dump(results, open(output_eval_file, "w"))

        print (results)

if __name__ == "__main__":
    main()
