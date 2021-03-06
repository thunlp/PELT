from fewshot_re_kit.data_loader import get_loader, get_loader_pair, get_loader_unsupervised
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import RobertaSentenceEncoder, RobertaEntSentenceEncoder, BERTSentenceEncoder
# from fewshot_re_kit.sentence_encoder import LukePAIRSentenceEncoder, RobertaEntPAIRSentenceEncoder, LukeSentenceEncoder, LukeBERTSentenceEncoder
import models
from models.proto import Proto
from models.gnn import GNN
from models.snail import SNAIL
from models.metanet import MetaNet
from models.siamese import Siamese
from models.pair import Pair
from models.d import Discriminator
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import logging
import random


logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_wiki',
            help='train file')
    parser.add_argument('--val', default='val_wiki',
            help='val file')
    parser.add_argument('--test', default='test_wiki',
            help='test file')
    parser.add_argument('--adv', default=None,
            help='adv file')
    parser.add_argument('--trainN', default=10, type=int,
            help='N in train')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=5, type=int,
            help='K shot')
    parser.add_argument('--Q', default=5, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=200, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=2000, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='proto',
            help='model name')
    parser.add_argument('--encoder', default='cnn',
            help='encoder: cnn or bert or roberta')
    parser.add_argument('--max_length', default=128, type=int,
           help='max length')
    parser.add_argument('--lr', default=1e-1, type=float,
           help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
           help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
           help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int,
           help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='sgd',
           help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=230, type=int,
           help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
           help='only test')

    # only for bert / roberta
    parser.add_argument('--pair', action='store_true',
           help='use pair model')
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true',
           help='concatenate entity representation as sentence rep')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', 
           help='use dot instead of L2 distance for proto')
    
    # experiment
    parser.add_argument('--mask_entity', action='store_true',
           help='mask entity names')

    parser.add_argument('--med_fewrel', action='store_true')
    parser.add_argument('--seed', default=42,type=int, help='seed')
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    parser.add_argument('--modL', default=7,type=int, help='modL')


    opt = parser.parse_args()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    if encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        if opt.pair:
            sentence_encoder = BERTPAIRSentenceEncoder(
                    pretrain_ckpt,
                    max_length)
        else:
            sentence_encoder = BERTSentenceEncoder(
                    pretrain_ckpt,
                    max_length,
                    cat_entity_rep=opt.cat_entity_rep,
                    mask_entity=opt.mask_entity)
    elif encoder_name == 'roberta':
        pretrain_ckpt = opt.pretrain_ckpt or 'roberta-base'
        if opt.pair:
            sentence_encoder = RobertaPAIRSentenceEncoder(
                    pretrain_ckpt,
                    max_length)
        else:
            sentence_encoder = RobertaSentenceEncoder(
                    pretrain_ckpt,
                    max_length,
                    cat_entity_rep=opt.cat_entity_rep)

    # elif encoder_name == 'luke':
    #     pretrain_ckpt = opt.pretrain_ckpt
    #     if opt.pair:
    #         sentence_encoder = LukePAIRSentenceEncoder(
    #                 pretrain_ckpt,
    #                 max_length)
    #     else:
    #         sentence_encoder = LukeSentenceEncoder(
    #                 pretrain_ckpt,
    #                 max_length,
    #                 cat_entity_rep=opt.cat_entity_rep)
    elif encoder_name == 'robertaent':
        pretrain_ckpt = opt.pretrain_ckpt

        sentence_encoder = RobertaEntSentenceEncoder(
                pretrain_ckpt,
                max_length,
                cat_entity_rep=opt.cat_entity_rep,
                )
    else:
        raise NotImplementedError
    
    if opt.pair:
        train_data_loader = get_loader_pair(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
        val_data_loader = get_loader_pair(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name, med_fewrel=opt.med_fewrel)
        test_data_loader = get_loader_pair(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name, med_fewrel=opt.med_fewrel)
    else:
        train_data_loader = get_loader(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
        val_data_loader = get_loader(opt.val, sentence_encoder,
                N=N, K=K, Q=1, na_rate=opt.na_rate, batch_size=32, encoder_name=encoder_name, med_fewrel=opt.med_fewrel, modL=opt.modL)
        test_data_loader = get_loader(opt.test, sentence_encoder,
                N=N, K=K, Q=1, na_rate=opt.na_rate, batch_size=32, encoder_name=encoder_name, med_fewrel=opt.med_fewrel, modL=opt.modL)
        if opt.adv:
           adv_data_loader = get_loader_unsupervised(opt.adv, sentence_encoder,
                N=trainN, K=K, Q=1, na_rate=opt.na_rate, batch_size=32)
   
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    if opt.adv:
        d = Discriminator(opt.hidden_size)
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, adv_data_loader, adv=opt.adv, d=d)
    else:
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
        
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    if opt.adv is not None:
        prefix += '-adv_' + opt.adv
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    if opt.dot:
        prefix += '-dot'
    if opt.cat_entity_rep:
        prefix += '-catentity'

    prefix += '-' + str(opt.seed)
    if model_name == 'proto':
        model = Proto(sentence_encoder, dot=opt.dot)
    elif model_name == 'gnn':
        model = GNN(sentence_encoder, N, hidden_size=opt.hidden_size)
    elif model_name == 'snail':
        model = SNAIL(sentence_encoder, N, K, hidden_size=opt.hidden_size)
    elif model_name == 'metanet':
        model = MetaNet(N, K, sentence_encoder.embedding, max_length)
    elif model_name == 'siamese':
        model = Siamese(sentence_encoder, hidden_size=opt.hidden_size, dropout=opt.dropout)
    elif model_name == 'pair':
        model = Pair(sentence_encoder, hidden_size=opt.hidden_size)
    else:
        raise NotImplementedError
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        bert_optim = True

        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, pair=opt.pair, 
                train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim, grad_iter=opt.grad_iter)
    else:
        # ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'


    acc = framework.eval(model, 32, N, K, 1, 2000, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair)
    # acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair)
    print (N, K, opt.test)
    print("RESULT: %.2f" % (acc * 100))

    path = opt.pretrain_ckpt
    path = path[path.rfind('/')+1:]

    path += '-'+encoder_name
    if opt.dot:
        w = open('results/'+path+ '_' +str(opt.test) + '_' +str(N)+'_'+str(K)+'_'+str(Q) +'_'+str(opt.seed)+'_'+str(opt.train_iter)+'_'+str(opt.modL)+'_dot.txt', 'w')
    else:
        w = open('results/'+path+ '_' +str(opt.test) + '_' +str(N)+'_'+str(K)+'_'+str(Q) +'_'+str(opt.seed)+'_'+str(opt.train_iter)+'_'+str(opt.modL)+'.txt', 'w')

    w.write(str(acc * 100))

if __name__ == "__main__":
    main()
