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


# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('-n', '--name', default='experiment')
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()
config = Config.from_json_file(args.config)
config.gpu_device = args.gpu
print("Run training on GPU " + str(config.gpu_device))

# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# output
output_dir = os.path.join(config.log_path, args.name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
log_file = os.path.join(output_dir, 'log.txt')
with open(log_file, 'w', encoding='utf-8') as w:
    w.write(json.dumps(config.to_dict()) + '\n')
    print('Log file: {}'.format(log_file))
best_role_model = os.path.join(output_dir, 'best.role.mdl')
best_event_model = os.path.join(output_dir, 'best.event.mdl')
best_relation_model = os.path.join(output_dir, 'best.relation.mdl')
final_model = os.path.join(output_dir, 'final.mdl')

dev_role_result_file = os.path.join(output_dir, 'role.dev.json')
dev_event_result_file = os.path.join(output_dir, 'event.dev.json')
dev_relation_result_file = os.path.join(output_dir, 'relation.dev.json')

test_role_result_file = os.path.join(output_dir, 'role.test.json')
test_event_result_file = os.path.join(output_dir, 'event.test.json')
test_relation_result_file = os.path.join(output_dir, 'relation.test.json')

# datasets
model_name = config.bert_model_name

org_train_graphs, train_align, train_exist = torch.load(config.train_amr)
org_dev_graphs, dev_align, dev_exist = torch.load(config.dev_amr)
org_test_graphs, test_align, test_exist = torch.load(config.test_amr)
if config.use_gpu:
    train_graphs, dev_graphs, test_graphs = [], [], []
    for g in org_train_graphs:
        g_device = g.to(config.gpu_device)
        train_graphs.append(g_device)

    for g in org_dev_graphs:
        g_device = g.to(config.gpu_device)
        dev_graphs.append(g_device)

    for g in org_test_graphs:
        g_device = g.to(config.gpu_device)
        test_graphs.append(g_device)
else:
    train_graphs, dev_graphs, test_graphs = org_train_graphs, org_dev_graphs, org_test_graphs

if config.bert_model_name.startswith("roberta"):
    tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=False)
else:
    tokenizer = BertTokenizer.from_pretrained(model_name,
                                              cache_dir=config.bert_cache_dir,
                                              do_lower_case=False)

# print(len(test_graphs))
train_set = IEDataset(config.train_file, train_graphs, train_align, train_exist, gpu=use_gpu,
                      relation_mask_self=config.relation_mask_self,
                      relation_directional=config.relation_directional,
                      symmetric_relations=config.symmetric_relations,
                      ignore_title=config.ignore_title)
dev_set = IEDataset(config.dev_file, dev_graphs, dev_align, dev_exist, gpu=use_gpu,
                    relation_mask_self=config.relation_mask_self,
                    relation_directional=config.relation_directional,
                    symmetric_relations=config.symmetric_relations)
test_set = IEDataset(config.test_file, test_graphs, test_align, test_exist, gpu=use_gpu,
                     relation_mask_self=config.relation_mask_self,
                     relation_directional=config.relation_directional,
                     symmetric_relations=config.symmetric_relations)
  
vocabs = generate_vocabs([train_set, dev_set, test_set])

train_set.numberize(tokenizer, vocabs)
dev_set.numberize(tokenizer, vocabs)
test_set.numberize(tokenizer, vocabs)

valid_patterns = load_valid_patterns(config.valid_pattern_path, vocabs)

batch_num = len(train_set) // config.batch_size
dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = OneIE(config, vocabs, valid_patterns)
model.load_bert(model_name, cache_dir=config.bert_cache_dir)
if use_gpu:
    model.cuda(device=config.gpu_device)

if config.use_graph_encoder:
    graph_params = []
    for k in range(config.gnn_layers):
        for para in model.graph_encoder.gnn.gats[k].fc.parameters():
            graph_params.append(para)
        for para in model.graph_encoder.gnn.gats[k].attn_fc.parameters():
            graph_params.append(para) 
        for para in model.graph_encoder.gnn.gats[k].edge_fc.parameters():
            graph_params.append(para) 

    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
            'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                    and 'crf' not in n and 'global_feature' not in n],
            'lr': config.learning_rate, 'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                    and ('crf' in n or 'global_feature' in n)],
            'lr': config.learning_rate, 'weight_decay': 0
        },
        {
            'params': graph_params, 'lr': config.learning_rate, 'weight_decay': 0
        }
    ]

else:

    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
            'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                    and 'crf' not in n and 'global_feature' not in n],
            'lr': config.learning_rate, 'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                    and ('crf' in n or 'global_feature' in n)],
            'lr': config.learning_rate, 'weight_decay': 0
        }
    ]

optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * config.warmup_epoch,
                                           num_training_steps=batch_num * config.max_epoch)

# model state
state = dict(model=model.state_dict(),
             config=config.to_dict(),
             vocabs=vocabs,
             valid=valid_patterns)

global_step = 0
global_feature_max_step = int(config.global_warmup * batch_num) + 1
print('global feature max step:', global_feature_max_step)
print("message passing level: ", config.lamda)

tasks = ['trigger', 'relation', 'role']
best_dev = {k: 0 for k in tasks}

for epoch in range(config.max_epoch):
    print('Epoch: {}'.format(epoch))

    # training set
    progress = tqdm.tqdm(total=batch_num, ncols=75,
                         desc='Train {}'.format(epoch))
    optimizer.zero_grad()
    train_loss = 0.0
    for batch_idx, batch in enumerate(DataLoader(
            train_set, batch_size=config.batch_size // config.accumulate_step,
            shuffle=True, drop_last=True, collate_fn=train_set.collate_fn)):

        loss = model(batch, epoch)
        loss = loss * (1 / config.accumulate_step)
        loss.backward()
        train_loss += loss.item()

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            global_step += 1
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    progress.close()
    print("Training Loss: --- " + str(train_loss / (batch_idx + 1)))

    # dev set
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                         desc='Dev {}'.format(epoch))
    best_dev_role_model = False
    best_dev_event_model = False
    best_dev_relation_model = False
    dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
    for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=dev_set.collate_fn):
        progress.update(1)
        graphs = model.predict(batch, epoch)
        if config.ignore_first_header:
            for inst_idx, sent_id in enumerate(batch.sent_ids):
                if int(sent_id.split('-')[-1]) < 4:
                    graphs[inst_idx] = Graph.empty_graph(vocabs)
        for graph in graphs:
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
        dev_gold_graphs.extend(batch.graphs)
        dev_pred_graphs.extend(graphs)
        dev_sent_ids.extend(batch.sent_ids)
        dev_tokens.extend(batch.tokens)
    progress.close()
    dev_scores = score_graphs(dev_gold_graphs, dev_pred_graphs,
                              relation_directional=config.relation_directional)
    for task in tasks:
        if dev_scores[task]['f'] > best_dev[task]:
            best_dev[task] = dev_scores[task]['f']
            if task == 'role':
                print('Saving best role model')
                torch.save(state, best_role_model)
                best_dev_role_model = True
                save_result(dev_role_result_file,
                            dev_gold_graphs, dev_pred_graphs, dev_sent_ids,
                            dev_tokens)
            if task == 'trigger':
                print('Saving best event model')
                torch.save(state, best_event_model)
                best_dev_event_model = True
                save_result(dev_event_result_file,
                            dev_gold_graphs, dev_pred_graphs, dev_sent_ids,
                            dev_tokens)
            if task == 'relation':
                print('Saving best relation model')
                torch.save(state, best_relation_model)
                best_dev_relation_model = True
                save_result(dev_relation_result_file,
                            dev_gold_graphs, dev_pred_graphs, dev_sent_ids,
                            dev_tokens)

    # test set
    progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                         desc='Test {}'.format(epoch))
    test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []
    for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False,
                            collate_fn=test_set.collate_fn):
        progress.update(1)
        graphs = model.predict(batch, epoch)
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

    if best_dev_role_model:
        save_result(test_role_result_file, test_gold_graphs, test_pred_graphs,
                    test_sent_ids, test_tokens)
    if best_dev_relation_model:
        save_result(test_relation_result_file, test_gold_graphs, test_pred_graphs,
                    test_sent_ids, test_tokens)
    if best_dev_event_model:
        save_result(test_event_result_file, test_gold_graphs, test_pred_graphs,
                    test_sent_ids, test_tokens)

    result = json.dumps(
        {'epoch': epoch, 'loss': train_loss / (batch_idx + 1), 'dev': dev_scores, 'test': dev_scores})
    with open(log_file, 'a', encoding='utf-8') as w:
        w.write(result + '\n')
    print('Log file', log_file)

torch.save(state, final_model)
print("Final model saved!")

print("================== Best Role Model ==================")
best_score_by_task(log_file, 'role')
print("================== Best Trigger Model ==================")
best_score_by_task(log_file, 'trigger')
print("================== Best Relation Model ==================")
best_score_by_task(log_file, 'relation')
