# coding=UTF-8
import os
import argparse
from tqdm import tqdm
import torch
import pdb
import random
import pickle
import numpy as np

from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from plm_score.plm_trainer import Trainer
from plm_score.plm_dataloader import DataLoader
from plm_score.plm_model import LMKE

# argparser
LM_parser = argparse.ArgumentParser()
LM_parser.add_argument('--seed', type=int, default=42)

LM_parser.add_argument('--bert_lr', type=float, default=1e-5)
LM_parser.add_argument('--model_lr', type=float, default=5e-4)
LM_parser.add_argument('--batch_size', type=int, default=512)
LM_parser.add_argument('--epoch', type=int, default=20)
LM_parser.add_argument('--weight_decay', type=float, default=1e-7)

LM_parser.add_argument('--data', type=str, default='wn18rr')
LM_parser.add_argument('--plm', type=str, default='bert')
LM_parser.add_argument('--description', type=str, default='desc')

LM_parser.add_argument('--load_path', type=str, default="")
LM_parser.add_argument('--load_epoch', type=int, default=-1)
LM_parser.add_argument('--load_metric', type=str, default='hits1')

LM_parser.add_argument('--max_desc_length', type=int, default=512)

# directly run test
LM_parser.add_argument('--link_prediction', default=True, action='store_true')
LM_parser.add_argument('--triple_classification', default=False, action='store_true')

LM_parser.add_argument('--add_tokens', default=False, action='store_true',
                       help='add entity and relation tokens into the vocabulary')
LM_parser.add_argument('--p_tuning', default=False, action='store_true', help='add learnable soft prompts')
LM_parser.add_argument('--prefix_tuning', default=False, action='store_true',
                       help='fix language models and only tune added components')
LM_parser.add_argument('--rdrop', default=False, action='store_true')
LM_parser.add_argument('--self_adversarial', default=True, action='store_true',
                       help='self adversarial negative sampling')
LM_parser.add_argument('--no_use_lm', default=False, action='store_true')
LM_parser.add_argument('--use_structure', default=False, action='store_true')
LM_parser.add_argument('--contrastive', default=True, action='store_true')
LM_parser.add_argument('--wandb', default=False, action='store_true')
LM_parser.add_argument('--task', default='LP')

LM_arg = LM_parser.parse_args()

neg_rate = 0
identifier = '{}-{}-{}-batch_size={}-prefix_tuning={}-max_desc_length={}'.format(LM_arg.data, LM_arg.plm,
                                                                                 LM_arg.description,
                                                                                 LM_arg.batch_size,
                                                                                 LM_arg.prefix_tuning,
                                                                                 LM_arg.max_desc_length)

# Set random seed
random.seed(LM_arg.seed)
np.random.seed(LM_arg.seed)
torch.manual_seed(LM_arg.seed)

device = torch.device('cuda')

plm_name = "bert-base-uncased"
t_model = 'bert'
if LM_arg.data == 'fb15k-237':
    in_paths = {
        'dataset': LM_arg.data,
        'train': './plm_score/data/fb15k-237/train.tsv',
        'valid': './plm_score/data/fb15k-237/dev.tsv',
        'test': './plm_score/data/fb15k-237/test.tsv',
        'text': ['./plm_score/data/fb15k-237/FB15k_mid2description.txt',
                 './plm_score/data/fb15k-237/relation2text.txt']
    }
elif LM_arg.data == 'wn18rr':
    in_paths = {
        'dataset': LM_arg.data,
        'train': './plm_score/data/WN18RR/train.tsv',
        'valid': './plm_score/data/WN18RR/dev.tsv',
        'test': './plm_score/data/WN18RR/test.tsv',
        'text': ['./plm_score/data/WN18RR/my_entity2text.txt',
                 './plm_score/data/WN18RR/relation2text.txt']
    }

lm_config = AutoConfig.from_pretrained(plm_name, cache_dir='./plm_score/cached_model')
lm_tokenizer = AutoTokenizer.from_pretrained(plm_name, do_basic_tokenize=False,
                                             cache_dir='./plm_score/cached_model')
lm_model = AutoModel.from_pretrained(plm_name, config=lm_config, cache_dir='./plm_score/cached_model')

# pdb.set_trace()
data_loader = DataLoader(in_paths, lm_tokenizer, batch_size=LM_arg.batch_size, neg_rate=neg_rate,
                         max_desc_length=LM_arg.max_desc_length,
                         add_tokens=LM_arg.add_tokens, p_tuning=LM_arg.p_tuning, rdrop=LM_arg.rdrop, model=t_model)
# print(data_loader.train_set)
if LM_arg.add_tokens:
    data_loader.adding_tokens()
    lm_model.resize_token_embeddings(len(lm_tokenizer))

LM_model = LMKE(lm_model, n_ent=len(data_loader.ent2id), n_rel=len(data_loader.rel2id), add_tokens=LM_arg.add_tokens,
                contrastive=LM_arg.contrastive)

LM_no_decay = ["bias", "LayerNorm.weight"]
param_group = [
    {'lr': LM_arg.model_lr, 'params': [p for n, p in LM_model.named_parameters()
                                       if ('lm_model' not in n) and
                                       (not any(nd in n for nd in LM_no_decay))],
     'weight_decay': LM_arg.weight_decay},
    {'lr': LM_arg.model_lr, 'params': [p for n, p in LM_model.named_parameters()
                                       if ('lm_model' not in n) and
                                       (any(nd in n for nd in LM_no_decay))],
     'weight_decay': 0.0},
]

if not LM_arg.prefix_tuning:
    param_group += [
        {'lr': LM_arg.bert_lr, 'params': [p for n, p in LM_model.named_parameters()
                                          if ('lm_model' in n) and
                                          (not any(nd in n for nd in LM_no_decay))],  # name中不包含bias和LayerNorm.weight
         'weight_decay': LM_arg.weight_decay},
        {'lr': LM_arg.bert_lr, 'params': [p for n, p in LM_model.named_parameters()
                                          if ('lm_model' in n) and
                                          (any(nd in n for nd in LM_no_decay))],
         'weight_decay': 0.0},
    ]

LM_optimizer = AdamW(param_group)  # transformer AdamW

scheduler = get_constant_schedule_with_warmup(LM_optimizer, num_warmup_steps=data_loader.step_per_epc)

hyperparams = {
    'batch_size': LM_arg.batch_size,
    'epoch': LM_arg.epoch,
    'identifier': identifier,
    'load_path': LM_arg.load_path,
    'evaluate_every': 1,
    'update_every': 1,
    'load_epoch': LM_arg.load_epoch,
    'load_metric': LM_arg.load_metric,
    'prefix_tuning': LM_arg.prefix_tuning,
    'plm': LM_arg.plm,
    'description': LM_arg.description,
    'neg_rate': neg_rate,
    'add_tokens': LM_arg.add_tokens,
    'max_desc_length': LM_arg.max_desc_length,
    'p_tuning': LM_arg.p_tuning,
    'rdrop': LM_arg.rdrop,
    'use_structure': LM_arg.use_structure,
    'self_adversarial': LM_arg.self_adversarial,
    'no_use_lm': LM_arg.no_use_lm,
    'contrastive': LM_arg.contrastive,
    'task': LM_arg.task,
    'wandb': LM_arg.wandb
}

LM_trainer = Trainer(data_loader, LM_model, lm_tokenizer, LM_optimizer, scheduler, device, hyperparams)
