import os
import json
import time
import dgl
from argparse import ArgumentParser

import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (RobertaTokenizer, RobertaConfig, AdamW,
                          get_linear_schedule_with_warmup)
from transformers import (BertTokenizer, BertConfig, AdamW,
                          get_linear_schedule_with_warmup)
from model import OneIE
from graph import Graph
from config import Config
from data import IEDataset
from scorer import score_graphs
from util import generate_vocabs, load_valid_patterns, save_result, best_score_by_task
from predict import load_model

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('--model_path', default='./models/default/best.role.mdl')
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()
config = Config.from_json_file(args.config)
config.gpu_device = args.gpu
print("Run training on GPU " + str(config.gpu_device))

# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# datasets
model_name = config.bert_model_name
model_path = args.model_path

org_test_graphs, test_align, test_exist = torch.load(config.test_amr)

if config.use_gpu:
    test_graphs = []
    for g in org_test_graphs:
        g_device = g.to(config.gpu_device)
        test_graphs.append(g_device)
else:
    test_graphs = org_test_graphs


test_set = IEDataset(config.test_file, test_graphs, test_align, test_exist, gpu=use_gpu,
                     relation_mask_self=config.relation_mask_self,
                     relation_directional=config.relation_directional,
                     symmetric_relations=config.symmetric_relations)

model, tokenizer, config, vocabs = load_model(model_path, model_name, device=args.gpu, gpu=True, beam_size=5)

test_set.numberize(tokenizer, vocabs)

valid_patterns = load_valid_patterns(config.valid_pattern_path, vocabs)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

print("message passing level: ", config.lamda)

tasks = ['trigger', 'relation', 'role']
best_dev = {k: 0 for k in tasks}

progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                         desc='Test')
test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []
for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False,
                        collate_fn=test_set.collate_fn):
    progress.update(1)
    graphs = model.predict(batch, 0)
    if config.ignore_first_header:
        for inst_idx, sent_id in enumerate(batch.sent_ids):
            if int(sent_id.split('-')[-1]) < 4:
                graphs[inst_idx] = Graph.empty_graph(vocabs)
    for graph in graphs:
        graph.clean(relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations)
    test_gold_graphs.extend(batch.graphs)
    test_pred_graphs.extend(graphs)
    test_sent_ids.extend(batch.sent_ids)
    test_tokens.extend(batch.tokens)
    
progress.close()
test_scores = score_graphs(test_gold_graphs, test_pred_graphs,
                            relation_directional=config.relation_directional)
