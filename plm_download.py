# coding=UTF-8
import os
import argparse
from transformers import AutoTokenizer, AutoModel, AutoConfig

if __name__ == '__main__':
    plm_name = "bert-base-uncased"
    lm_config = AutoConfig.from_pretrained(plm_name, cache_dir='./plm_score/cached_model')
    lm_tokenizer = AutoTokenizer.from_pretrained(plm_name, do_basic_tokenize=False,
                                                 cache_dir='./plm_score/cached_model')
    lm_model = AutoModel.from_pretrained(plm_name, config=lm_config, cache_dir='./plm_score/cached_model')
    print("download bert finished")
