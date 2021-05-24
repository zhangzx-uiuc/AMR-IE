import os
import json
import glob
import tqdm
import traceback
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig

from model import OneIE
from config import Config
from util import save_result
from data import IEDatasetEval
from convert import json_to_cs

cur_dir = os.path.dirname(os.path.realpath(__file__))
format_ext_mapping = {'txt': 'txt', 'ltf': 'ltf.xml', 'json': 'json',
                      'json_single': 'json'}

def load_model(model_path, bert_model, device=0, gpu=False, beam_size=5):
    print('Loading the model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    # print(config)
    if type(config) is dict:
        config = Config.from_dict(config)
    config.bert_cache_dir = os.path.join(cur_dir, 'bert')
    vocabs = state['vocabs']
    valid_patterns = state['valid']

    # recover the model
    model = OneIE(config, vocabs, valid_patterns)
    model.load_bert(bert_model)
    model.load_state_dict(state['model'])
    model.beam_size = beam_size
    if gpu:
        model.cuda(device)

    # tokenizer = RobertaTokenizer.from_pretrained(config.bert_model_name,
                                                # do_lower_case=False)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name,
                                                do_lower_case=False)

    return model, tokenizer, config, vocabs


def predict_document(path, model, tokenizer, config, batch_size=20, 
                     max_length=128, gpu=False, input_format='txt',
                     language='english'):
    """
    :param path (str): path to the input file.
    :param model (OneIE): pre-trained model object.
    :param tokenizer (BertTokenizer): BERT tokenizer.
    :param config (Config): configuration object.
    :param batch_size (int): Batch size (default=20).
    :param max_length (int): Max word piece number (default=128).
    :param gpu (bool): Use GPU or not (default=False).
    :param input_format (str): Input file format (txt or ltf, default='txt).
    :param langauge (str): Input document language (default='english').
    """
    test_set = IEDatasetEval(path, max_length=max_length, gpu=gpu,
                             input_format=input_format, language=language)
    test_set.numberize(tokenizer)
    # document info
    info = {
        'doc_id': test_set.doc_id,
        'ori_sent_num': test_set.ori_sent_num,
        'sent_num': len(test_set)
    }
    # prediction result
    result = []
    for batch in DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            collate_fn=test_set.collate_fn):
        graphs = model.predict(batch)
        for graph, tokens, sent_id, token_ids in zip(graphs, batch.tokens,
                                                     batch.sent_ids,
                                                     batch.token_ids):
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
            result.append((sent_id, token_ids, tokens, graph))
    return result, info