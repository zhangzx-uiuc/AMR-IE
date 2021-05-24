import torch
import dgl
import json
# import networkx as nx
# import matplotlib.pyplot as plt

from transition_amr_parser.stack_transformer_amr_parser import AMRParser

def get_edge_type_idx(edge_type_str, if_dep):
    if if_dep:
        return 0
    elif edge_type_str in ['location', 'destination', 'path']:
        return 1
    elif edge_type_str in ['year', 'time', 'duration', 'decade', 'weekday']:
        return 2
    elif edge_type_str in ['instrument', 'manner', 'poss']:
        return 3
    elif edge_type_str.startswith('prep-'):
        return 4
    elif edge_type_str.startswith('op') and edge_type_str[-1].isdigit():
        return 5
    elif edge_type_str == 'ARG0':
        return 6
    elif edge_type_str == 'ARG1':
        return 7
    elif edge_type_str == 'ARG2':
        return 8
    elif edge_type_str == 'ARG3':
        return 9
    else:
        return 10

def get_amr_edge_idx(edge_type_str):
    if edge_type_str in ['location', 'destination', 'path']:
        return 0
    elif edge_type_str in ['year', 'time', 'duration', 'decade', 'weekday']:
        return 1
    elif edge_type_str in ['instrument', 'manner', 'poss', 'topic', 'medium', 'duration']:
        return 2
    elif edge_type_str in ['mod']:
        return 3
    elif edge_type_str.startswith('prep-'):
        return 4
    elif edge_type_str.startswith('op') and edge_type_str[-1].isdigit():
        return 5
    elif edge_type_str == 'ARG0':
        return 6
    elif edge_type_str == 'ARG1':
        return 7
    elif edge_type_str == 'ARG2':
        return 8
    elif edge_type_str == 'ARG3':
        return 9
    elif edge_type_str == 'ARG4':
        return 10
    else:
        return 11


def get_amr_edge_idx_sparse(edge_type_str):
    if edge_type_str in ['location', 'destination', 'path']:
        return 0
    elif edge_type_str in ['year', 'time', 'duration', 'decade', 'weekday']:
        return 1
    elif edge_type_str in ['instrument']:
        return 2
    elif edge_type_str.startswith('prep-'):
        return 3
    elif edge_type_str.startswith('ARG'):
        return 4
    else:
        return 5


def get_amr_edge_idx_richer(edge_type_str):
    valid_edges = {
        'ARG0':0, 
        'ARG1':1, 
        'ARG2':2, 
        'ARG3':3, 
        'ARG4':4,
        'op1':5, 
        'op2':6, 
        'op3':7,
        'op4':8,
        'mod':9,
        'location':10,
        'instrument':11,
        'poss':12,
        'manner':13,
        'topic':14,
        'medium':15,
        'year':16,
        'duration':17,
        'decade':18,
        'weekday':19,
        'time':20,
        'destination':21,
        'path':22,
        'source':23,
        'accompanier':24,
        'purpose':25,
        'cause':26,
        'concession':27,
        'condition':28,
        'part':29,
        'subevent':30,
        'consist-of':31,
        'example':32,
        'direction':33,
        'prep-against':34,
        'prep-along-with':35, 
        'prep-amid':36,
        'prep-among':37,
        'prep-as':38,
        'prep-at':39,
        'prep-by':40, 
        'prep-concerning':41,
        'prep-considering':42, 
        'prep-despite':43, 
        'prep-except':44,
        'prep-excluding':45, 
        'prep-following':46,
        'prep-for':47,
        'prep-from':48,
        'prep-in':49,
        'prep-in-addition-to':50,
        'prep-in-spite-of':51,
        'prep-into':52, 
        'prep-like':53,
        'prep-on':54,
        'prep-on-behalf-of':55,
        'prep-opposite':56, 
        'prep-per':57,
        'prep-regarding':58, 
        'prep-save':59,
        'prep-such-as':60, 
        'prep-through':61,
        'prep-to':62,
        'prep-toward':63,
        'prep-under':64,
        'prep-unlike':65,
        'prep-versus':66,
        'prep-with':67,
        'prep-within':68,
        'prep-without':69,
        'null_edge':70,
        'domain':71
    }
    if edge_type_str in valid_edges:
        return valid_edges[edge_type_str]
    else:
        return 72

# Clostridium spp. and Eubacterium spp. have the enzymatic capacity to modify bile acids (41) .

def amr_parse(tokens_list, output_dir):
    # parser = AMRParser.from_checkpoint('/home/zixuan11/oneie_v0.4.7/amr/checkpoint_best.pt')
    # parser = AMRParser.from_checkpoint('/shared/nas/data/m1/nnp2/bioamr/checkpoint_best.pt')
#     parser = AMRParser.from_checkpoint('/home/featurize/amr_ibm/checkpoint_best.pt')
    parser = AMRParser.from_checkpoint('/home/featurize/bioamr/checkpoint_best.pt')
    amr_list = parser.parse_sentences(tokens_list)
    torch.save(amr_list, output_dir)

def processing_amr(amr_dir, tokens_list):
    amr_list = torch.load(amr_dir)
    # tokens_list = [['Orders', 'went', 'out', 'the', 'day', 'before', 'yesterday', 'to', 'deploy', '17000', 'U.S.', 'Army', 'soldiers', 'in', 'the', 'Persian', 'Gulf', 'region', 'for', 'security', 'reasons', '.'], ['Orders', 'went', 'out', 'the', 'day', 'before', 'yesterday', 'to', 'deploy', '17000', 'U.S.', 'Army', 'soldiers', 'in', 'the', 'Persian', 'Gulf', 'region', 'for', 'security', 'reasons', '.']]
    # amr_list = ['# ::tok Orders went out the day before yesterday to deploy 17000 U.S. Army soldiers in the Persian Gulf region for security reasons . <ROOT>\n# ::node\t1\torder-01\t0-1\n# ::node\t5\tday\t4-5\n# ::node\t6\tbefore\t5-6\n# ::node\t7\tyesterday\t6-7\n# ::node\t9\tdeploy-01\t8-9\n# ::node\t10\t17000\t9-10\n# ::node\t12\tmilitary\t10-12\n# ::node\t13\tsoldier\t12-13\n# ::node\t17\tworld-region\t15-17\n# ::node\t19\tcause-01\t18-19\n# ::node\t20\tsecurity\t19-20\n# ::node\t21\treason\t20-21\n# ::node\t27\tname\t10-12\n# ::node\t28\t"U.S."\t10-12\n# ::node\t29\t"Army"\t10-12\n# ::node\t30\tname\t15-17\n# ::node\t32\t"Persian"\t15-17\n# ::node\t33\t"Gulf"\t15-17\n# ::root\t1\torder-01\n# ::edge\tbefore\tquant\tday\t6\t5\t\n# ::edge\torder-01\ttime\tbefore\t1\t6\t\n# ::edge\tbefore\top1\tyesterday\t6\t7\t\n# ::edge\torder-01\tARG1\tdeploy-01\t1\t9\t\n# ::edge\tsoldier\tmod\tmilitary\t13\t12\t\n# ::edge\tsoldier\tquant\t17000\t13\t10\t\n# ::edge\tdeploy-01\tARG1\tsoldier\t9\t13\t\n# ::edge\tdeploy-01\tlocation\tworld-region\t9\t17\t\n# ::edge\tdeploy-01\tARG1-of\tcause-01\t9\t19\t\n# ::edge\treason\tmod\tsecurity\t21\t20\t\n# ::edge\tcause-01\tARG0\treason\t19\t21\t\n# ::edge\tmilitary\tname\tname\t12\t27\t\n# ::edge\tname\top1\t"U.S."\t27\t28\t\n# ::edge\tname\top2\t"Army"\t27\t29\t\n# ::edge\tworld-region\tname\tname\t17\t30\t\n# ::edge\tname\top1\t"Persian"\t30\t32\t\n# ::edge\tname\top2\t"Gulf"\t30\t33\t\n(o / order-01\n      :ARG1 (d2 / deploy-01\n            :ARG1 (s / soldier\n                  :mod (m / military\n                        :name (n / name\n                              :op1 "U.S."\n                              :op2 "Army"))\n                  :quant 17000)\n            :ARG1-of (c / cause-01\n                  :ARG0 (r / reason\n                        :mod (s2 / security)))\n            :location (w / world-region\n                  :name (n2 / name\n                        :op1 "Persian"\n                        :op2 "Gulf")))\n      :time (b / before\n            :op1 (y / yesterday)\n            :quant (d / day)))\n\n', '# ::tok Orders went out the day before yesterday to deploy 17000 U.S. Army soldiers in the Persian Gulf region for security reasons . <ROOT>\n# ::node\t1\torder-01\t0-1\n# ::node\t5\tday\t4-5\n# ::node\t6\tbefore\t5-6\n# ::node\t7\tyesterday\t6-7\n# ::node\t9\tdeploy-01\t8-9\n# ::node\t10\t17000\t9-10\n# ::node\t12\tmilitary\t10-12\n# ::node\t13\tsoldier\t12-13\n# ::node\t17\tworld-region\t15-17\n# ::node\t19\tcause-01\t18-19\n# ::node\t20\tsecurity\t19-20\n# ::node\t21\treason\t20-21\n# ::node\t27\tname\t10-12\n# ::node\t28\t"U.S."\t10-12\n# ::node\t29\t"Army"\t10-12\n# ::node\t30\tname\t15-17\n# ::node\t32\t"Persian"\t15-17\n# ::node\t33\t"Gulf"\t15-17\n# ::root\t1\torder-01\n# ::edge\tbefore\tquant\tday\t6\t5\t\n# ::edge\torder-01\ttime\tbefore\t1\t6\t\n# ::edge\tbefore\top1\tyesterday\t6\t7\t\n# ::edge\torder-01\tARG1\tdeploy-01\t1\t9\t\n# ::edge\tsoldier\tmod\tmilitary\t13\t12\t\n# ::edge\tsoldier\tquant\t17000\t13\t10\t\n# ::edge\tdeploy-01\tARG1\tsoldier\t9\t13\t\n# ::edge\tdeploy-01\tlocation\tworld-region\t9\t17\t\n# ::edge\tdeploy-01\tARG1-of\tcause-01\t9\t19\t\n# ::edge\treason\tmod\tsecurity\t21\t20\t\n# ::edge\tcause-01\tARG0\treason\t19\t21\t\n# ::edge\tmilitary\tname\tname\t12\t27\t\n# ::edge\tname\top1\t"U.S."\t27\t28\t\n# ::edge\tname\top2\t"Army"\t27\t29\t\n# ::edge\tworld-region\tname\tname\t17\t30\t\n# ::edge\tname\top1\t"Persian"\t30\t32\t\n# ::edge\tname\top2\t"Gulf"\t30\t33\t\n(o / order-01\n      :ARG1 (d2 / deploy-01\n            :ARG1 (s / soldier\n                  :mod (m / military\n                        :name (n / name\n                              :op1 "U.S."\n                              :op2 "Army"))\n                  :quant 17000)\n            :ARG1-of (c / cause-01\n                  :ARG0 (r / reason\n                        :mod (s2 / security)))\n            :location (w / world-region\n                  :name (n2 / name\n                        :op1 "Persian"\n                        :op2 "Gulf")))\n      :time (b / before\n            :op1 (y / yesterday)\n            :quant (d / day)))\n\n']
    # print(amr_list[0])
    # print(tokens_list[0])
    # amr_list = ['# ::tok Orders went out today to deploy 17,000 U.S. Army soldiers in the Persian Gulf region . <ROOT>\n# ::node\t1\torder-01\t0-1\n# ::node\t2\tgo-out-17\t1-2\n# ::node\t4\ttoday\t3-4\n# ::node\t6\tdeploy-01\t5-6\n# ::node\t9\tmilitary\t7-9\n# ::node\t10\tsoldier\t9-10\n# ::node\t14\tworld-region\t12-14\n# ::node\t21\tname\t7-9\n# ::node\t22\t"U.S."\t7-9\n# ::node\t23\t"Army"\t7-9\n# ::node\t24\tname\t12-14\n# ::node\t26\t"Persian"\t12-14\n# ::node\t27\t"Gulf"\t12-14\n# ::root\t2\tgo-out-17\n# ::edge\tgo-out-17\tARG0\torder-01\t2\t1\t\n# ::edge\tgo-out-17\ttime\ttoday\t2\t4\t\n# ::edge\tgo-out-17\tARG1\tdeploy-01\t2\t6\t\n# ::edge\tdeploy-01\tARG0\torder-01\t6\t1\t\n# ::edge\tsoldier\tmod\tmilitary\t10\t9\t\n# ::edge\tdeploy-01\tARG1\tsoldier\t6\t10\t\n# ::edge\tdeploy-01\tARG2\tworld-region\t6\t14\t\n# ::edge\tmilitary\tname\tname\t9\t21\t\n# ::edge\tname\top1\t"U.S."\t21\t22\t\n# ::edge\tname\top2\t"Army"\t21\t23\t\n# ::edge\tworld-region\tname\tname\t14\t24\t\n# ::edge\tname\top1\t"Persian"\t24\t26\t\n# ::edge\tname\top2\t"Gulf"\t24\t27\t\n(g / go-out-17\n      :ARG0 (o / order-01)\n      :ARG1 (d / deploy-01\n            :ARG0 o\n            :ARG1 (s / soldier\n                  :mod (m / military\n                        :name (n / name\n                              :op1 "U.S."\n                              :op2 "Army")))\n            :ARG2 (w / world-region\n                  :name (n2 / name\n                        :op1 "Persian"\n                        :op2 "Gulf")))\n      :time (t / today))\n\n', '# ::tok Orders went out today to deploy 17,000 U.S. Army soldiers in the Persian Gulf region . <ROOT>\n# ::node\t1\torder-01\t0-1\n# ::node\t2\tgo-out-17\t1-2\n# ::node\t4\ttoday\t3-4\n# ::node\t6\tdeploy-01\t5-6\n# ::node\t9\tmilitary\t7-9\n# ::node\t10\tsoldier\t9-10\n# ::node\t14\tworld-region\t12-14\n# ::node\t21\tname\t7-9\n# ::node\t22\t"U.S."\t7-9\n# ::node\t23\t"Army"\t7-9\n# ::node\t24\tname\t12-14\n# ::node\t26\t"Persian"\t12-14\n# ::node\t27\t"Gulf"\t12-14\n# ::root\t2\tgo-out-17\n# ::edge\tgo-out-17\tARG0\torder-01\t2\t1\t\n# ::edge\tgo-out-17\ttime\ttoday\t2\t4\t\n# ::edge\tgo-out-17\tARG1\tdeploy-01\t2\t6\t\n# ::edge\tdeploy-01\tARG0\torder-01\t6\t1\t\n# ::edge\tsoldier\tmod\tmilitary\t10\t9\t\n# ::edge\tdeploy-01\tARG1\tsoldier\t6\t10\t\n# ::edge\tdeploy-01\tARG2\tworld-region\t6\t14\t\n# ::edge\tmilitary\tname\tname\t9\t21\t\n# ::edge\tname\top1\t"U.S."\t21\t22\t\n# ::edge\tname\top2\t"Army"\t21\t23\t\n# ::edge\tworld-region\tname\tname\t14\t24\t\n# ::edge\tname\top1\t"Persian"\t24\t26\t\n# ::edge\tname\top2\t"Gulf"\t24\t27\t\n(g / go-out-17\n      :ARG0 (o / order-01)\n      :ARG1 (d / deploy-01\n            :ARG0 o\n            :ARG1 (s / soldier\n                  :mod (m / military\n                        :name (n / name\n                              :op1 "U.S."\n                              :op2 "Army")))\n            :ARG2 (w / world-region\n                  :name (n2 / name\n                        :op1 "Persian"\n                        :op2 "Gulf")))\n      :time (t / today))\n\n']
    # amr_list = ['# ::tok Orders went out the day before yesterday to deploy 17000 U.S. Army soldiers in the Persian Gulf region for security reasons . <ROOT>\n# ::node\t1\torder-01\t0-1\n# ::node\t5\tday\t4-5\n# ::node\t6\tbefore\t5-6\n# ::node\t7\tyesterday\t6-7\n# ::node\t9\tdeploy-01\t8-9\n# ::node\t10\t17000\t9-10\n# ::node\t12\tmilitary\t10-12\n# ::node\t13\tsoldier\t12-13\n# ::node\t17\tworld-region\t15-17\n# ::node\t19\tcause-01\t18-19\n# ::node\t20\tsecurity\t19-20\n# ::node\t21\treason\t20-21\n# ::node\t27\tname\t10-12\n# ::node\t28\t"U.S."\t10-12\n# ::node\t29\t"Army"\t10-12\n# ::node\t30\tname\t15-17\n# ::node\t32\t"Persian"\t15-17\n# ::node\t33\t"Gulf"\t15-17\n# ::root\t1\torder-01\n# ::edge\tbefore\tquant\tday\t6\t5\t\n# ::edge\torder-01\ttime\tbefore\t1\t6\t\n# ::edge\tbefore\top1\tyesterday\t6\t7\t\n# ::edge\torder-01\tARG1\tdeploy-01\t1\t9\t\n# ::edge\tsoldier\tmod\tmilitary\t13\t12\t\n# ::edge\tsoldier\tquant\t17000\t13\t10\t\n# ::edge\tdeploy-01\tARG1\tsoldier\t9\t13\t\n# ::edge\tdeploy-01\tlocation\tworld-region\t9\t17\t\n# ::edge\tdeploy-01\tARG1-of\tcause-01\t9\t19\t\n# ::edge\treason\tmod\tsecurity\t21\t20\t\n# ::edge\tcause-01\tARG0\treason\t19\t21\t\n# ::edge\tmilitary\tname\tname\t12\t27\t\n# ::edge\tname\top1\t"U.S."\t27\t28\t\n# ::edge\tname\top2\t"Army"\t27\t29\t\n# ::edge\tworld-region\tname\tname\t17\t30\t\n# ::edge\tname\top1\t"Persian"\t30\t32\t\n# ::edge\tname\top2\t"Gulf"\t30\t33\t\n(o / order-01\n      :ARG1 (d2 / deploy-01\n            :ARG1 (s / soldier\n                  :mod (m / military\n                        :name (n / name\n                              :op1 "U.S."\n                              :op2 "Army"))\n                  :quant 17000)\n            :ARG1-of (c / cause-01\n                  :ARG0 (r / reason\n                        :mod (s2 / security)))\n            :location (w / world-region\n                  :name (n2 / name\n                        :op1 "Persian"\n                        :op2 "Gulf")))\n      :time (b / before\n            :op1 (y / yesterday)\n            :quant (d / day)))\n\n', '# ::tok Orders went out the day before yesterday to deploy 17000 U.S. Army soldiers in the Persian Gulf region for security reasons . <ROOT>\n# ::node\t1\torder-01\t0-1\n# ::node\t5\tday\t4-5\n# ::node\t6\tbefore\t5-6\n# ::node\t7\tyesterday\t6-7\n# ::node\t9\tdeploy-01\t8-9\n# ::node\t10\t17000\t9-10\n# ::node\t12\tmilitary\t10-12\n# ::node\t13\tsoldier\t12-13\n# ::node\t17\tworld-region\t15-17\n# ::node\t19\tcause-01\t18-19\n# ::node\t20\tsecurity\t19-20\n# ::node\t21\treason\t20-21\n# ::node\t27\tname\t10-12\n# ::node\t28\t"U.S."\t10-12\n# ::node\t29\t"Army"\t10-12\n# ::node\t30\tname\t15-17\n# ::node\t32\t"Persian"\t15-17\n# ::node\t33\t"Gulf"\t15-17\n# ::root\t1\torder-01\n# ::edge\tbefore\tquant\tday\t6\t5\t\n# ::edge\torder-01\ttime\tbefore\t1\t6\t\n# ::edge\tbefore\top1\tyesterday\t6\t7\t\n# ::edge\torder-01\tARG1\tdeploy-01\t1\t9\t\n# ::edge\tsoldier\tmod\tmilitary\t13\t12\t\n# ::edge\tsoldier\tquant\t17000\t13\t10\t\n# ::edge\tdeploy-01\tARG1\tsoldier\t9\t13\t\n# ::edge\tdeploy-01\tlocation\tworld-region\t9\t17\t\n# ::edge\tdeploy-01\tARG1-of\tcause-01\t9\t19\t\n# ::edge\treason\tmod\tsecurity\t21\t20\t\n# ::edge\tcause-01\tARG0\treason\t19\t21\t\n# ::edge\tmilitary\tname\tname\t12\t27\t\n# ::edge\tname\top1\t"U.S."\t27\t28\t\n# ::edge\tname\top2\t"Army"\t27\t29\t\n# ::edge\tworld-region\tname\tname\t17\t30\t\n# ::edge\tname\top1\t"Persian"\t30\t32\t\n# ::edge\tname\top2\t"Gulf"\t30\t33\t\n(o / order-01\n      :ARG1 (d2 / deploy-01\n            :ARG1 (s / soldier\n                  :mod (m / military\n                        :name (n / name\n                              :op1 "U.S."\n                              :op2 "Army"))\n                  :quant 17000)\n            :ARG1-of (c / cause-01\n                  :ARG0 (r / reason\n                        :mod (s2 / security)))\n            :location (w / world-region\n                  :name (n2 / name\n                        :op1 "Persian"\n                        :op2 "Gulf")))\n      :time (b / before\n            :op1 (y / yesterday)\n            :quant (d / day)))\n\n']
    # print(amr_list[0])

    node_idx_list, edge_type_list, node_idx_offset_list, node_idx_offset_whole = [], [], [], []
    list_of_align_dict = []
    list_of_exist_dict = []

    total_edge_num = 0
    covered_edge_num = 0
    order_list = []
    for i, amr in enumerate(amr_list):
        amr_split_list = amr.split('\n')
        # print(amr_split_list)
        node_to_idx, node_to_offset, node_to_offset_whole = {}, {}, {}
        node_num = 0
        # first to fill in the node list
        for line in amr_split_list:
            if line.startswith('# ::node'):
                node_split = line.split('\t')
                # print(node_split)
                if len(node_split) != 4:
                    # check if the alignment text spans exist
                    continue
                else:
                    align_span = node_split[3].split('-')
                    if not align_span[0].isdigit():
                        continue
                    head_word_idx = int(align_span[1]) - 1
                    try:
                        start = int(align_span[0])
                    except:
                        # print(amr_list[i])
                        raise ValueError
                    end = int(align_span[1])
                    if (start, end) not in list(node_to_offset_whole.values()):
                        node_to_offset.update({node_split[1]: head_word_idx})
                        node_to_offset_whole.update({node_split[1]: (start, end)})
                        node_to_idx.update({node_split[1]: node_num})
                        node_num += 1
            else:
                continue
        # print(node_to_idx)
        # print(node_to_offset)
        node_idx_list.append(node_to_idx)
        # print(node_idx_list)
        # change str2offset to idx2offset
        node_idx_to_offset = {}
        for key in node_to_idx.keys():
            node_idx_to_offset.update({node_to_idx[key]: node_to_offset[key]})

        node_idx_to_offset_whole = {}
        for key in node_to_idx.keys():
            node_idx_to_offset_whole.update({node_to_idx[key]: node_to_offset_whole[key]})
        # print(node_idx_to_offset)
        node_idx_offset_list.append(node_idx_to_offset)
        node_idx_offset_whole.append(node_idx_to_offset_whole)

        # print(node_to_idx)
        # print(node_idx_to_offset)
        # print(node_idx_to_offset_whole)
        # second we go through the edges
        edge_type_dict = {}

        for line in amr_split_list:
            if line.startswith('# ::root'):
                root_split = line.split('\t')
                root = root_split[1]
        prior_dict = {root:[]}

        start_list = []
        end_list = []

        for line in amr_split_list:
            if line.startswith('# ::edge'):
                edge_split = line.split('\t')
                amr_edge_type = edge_split[2]
                edge_start = edge_split[4]
                edge_end = edge_split[5]
                # check if the start and end nodes exist
                if (edge_start in node_to_idx) and (edge_end in node_to_idx):
                    # check if the edge type is "ARGx-of", if so, reverse the direction of the edge
                    if amr_edge_type.startswith("ARG") and amr_edge_type.endswith("-of"):
                        edge_start, edge_end = edge_end, edge_start
                        amr_edge_type = amr_edge_type[0:4]
                    # deal with this edge here
                    edge_idx = get_amr_edge_idx(amr_edge_type)
                    total_edge_num += 1
                    if edge_idx == 11:
                        covered_edge_num += 1
                    start_idx = node_to_idx[edge_start]
                    end_idx = node_to_idx[edge_end]
                    edge_type_dict.update({(start_idx, end_idx): edge_idx})
                
                else:
                    continue
                # print(edge_start, edge_end)
                if edge_end != root and (not ((edge_start in end_list) and (edge_end in start_list))):
                    start_list.append(edge_start)
                    end_list.append(edge_end)
                if edge_start not in prior_dict:
                    prior_dict.update({edge_start:[edge_end]})
                else:
                    prior_dict[edge_start].append(edge_end)
            else:
                continue
        # print(edge_type_dict)
        # print(prior_dict)
        edge_type_list.append(edge_type_dict)
        # generating priority list for decoding
        # print(start_list)
        # print(end_list)
        final_order_list = []
        # output orders
        candidate_nodes = node_to_idx.copy()
        # print(node_to_idx)
        # print(start_list)
        # print(end_list)
        # print(i)
        while len(candidate_nodes) != 0:
            # print(candidate_nodes)
            current_level_nodes = []
            for key in candidate_nodes:
                if key not in end_list:
                    final_order_list.append(candidate_nodes[key])
                    current_level_nodes.append(key)
            # print(current_level_nodes)
            # Remove current level nodes from the dictionary
            for node in current_level_nodes:
                candidate_nodes.pop(node)
            
            # deleting from start lists the current level nodes
            for node in current_level_nodes:
                indices_list = [i for i, x in enumerate(start_list) if x == node]
                start_list = [x for x in start_list if x != node]
                new_end_list = []
                for i in range(len(end_list)):
                    if i not in indices_list:
                        new_end_list.append(end_list[i])
                end_list = new_end_list
        # print(final_order_list)
        order_list.append(final_order_list.copy())
    # print(prior_dict)
    # while len(prior_list) != num_nodes:
    #     childs = prior_dict[root]
    #     for i in range(len(childs)):
    #         prior_list.append(childs[i])
    # print(final_order_list)
    # feed into dgl graphs
    graphs_list = []
    # print(node_idx_list)
    # print(node_idx_offset_list)
    # print(covered_edge_num / total_edge_num)
    # print(edge_type_list)
    for i in range(len(node_idx_list)):
        graph_i = dgl.DGLGraph()

        edge2type = edge_type_list[i]
        node2offset = node_idx_offset_list[i]
        node2offset_whole = node_idx_offset_whole[i]

        nodes_num = len(node2offset)

        graph_i.add_nodes(nodes_num)
        graph_i.ndata['token_pos'] = torch.zeros(nodes_num, 1, dtype=torch.long)
        graph_i.ndata['token_span'] = torch.zeros(nodes_num, 2, dtype=torch.long)

        # fill in token positions
        for key in node2offset:
            graph_i.ndata['token_pos'][key][0] = node2offset[key]
        for key in node2offset:
            graph_i.ndata['token_span'][key][0] = node2offset_whole[key][0]
            graph_i.ndata['token_span'][key][1] = node2offset_whole[key][1]
        # add nodes priorities
        node_prior_tensor = torch.zeros(nodes_num, 1, dtype=torch.long)
        for j in range(nodes_num):
            node_prior_tensor[j][0] = order_list[i].index(j)
        # print(node_prior_tensor)
        graph_i.ndata['priority'] = node_prior_tensor
        # print(node_prior_tensor)
        # add edges
        edge_num = len(edge2type)
        

        edge_iter = 0

        # ''' uni-directional edges '''
        # edge_type_tensor = torch.zeros(edge_num, 1, dtype=torch.long)
        # for key in edge2type:
        #     graph_i.add_edges(key[0], key[1])
        #     edge_type_tensor[edge_iter][0] = edge2type[key]
        #     edge_iter += 1

        # ''' inverse-directional edges '''
        # edge_type_tensor = torch.zeros(edge_num, 1, dtype=torch.long)
        # for key in edge2type:
        #     graph_i.add_edges(key[1], key[0])
        #     edge_type_tensor[edge_iter][0] = edge2type[key]
        #     edge_iter += 1
        
        ''' bi-directional edges '''
        edge_type_tensor = torch.zeros(2 * edge_num, 1, dtype=torch.long)
        for key in edge2type:
            graph_i.add_edges(key[0], key[1])
            edge_type_tensor[edge_iter][0] = edge2type[key]
            edge_iter += 1

        for key in edge2type:
            graph_i.add_edges(key[1], key[0])
            edge_type_tensor[edge_iter][0] = edge2type[key]
            edge_iter += 1
        
        graph_i.edata['type'] = edge_type_tensor
        # print(edge2type)
        # print(graph_i.edges())
        # print(graph_i.edata['type'])
        # print(graph_i)
        # print(graph_i.edges())
        # print(graph_i.edata['type'])
        # print(graph_i.ndata['token_pos'])
        # print(graph_i.ndata['token_span'])
        # nx_G = graph_i.to_networkx()
        # pos = nx.kamada_kawai_layout(nx_G)
        # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
        # plt.show()
        graphs_list.append(graph_i)

        align_dict = {}
        exist_dict = {}
        # print(tokens_list[i])
        # print(graph_i.ndata["token_span"])

        span_list = graph_i.ndata["token_span"].tolist()
        # print(span_list)

        for p in range(len(tokens_list[i])):
            min_dis = 2 * len(tokens_list[i])
            min_dis_idx = -1

            if_found = 0

            for q in range(len(span_list)):
                if p >= span_list[q][0] and p < span_list[q][1]:
                    if_found = 1
                    align_dict.update({p: q})
                    exist_dict.update({p: 1})
                    break
                else:
                    new_dis_1 = abs(p - span_list[q][0])
                    new_dis_2 = abs(p - (span_list[q][1] - 1))
                    new_dis = min(new_dis_1, new_dis_2)
                    if new_dis < min_dis:
                        min_dis = new_dis
                        min_dis_idx = q
            
            if not if_found:
                align_dict.update({p: min_dis_idx})
                exist_dict.update({p: 0})

        # print(tokens_list[i])
        # print(align_dict)
        # print(exist_dict)
        # print(graph_i)
        # if graph_i.num_nodes() == 0:
        #     print(tokens_list[i])

        list_of_align_dict.append(align_dict)
        list_of_exist_dict.append(exist_dict)
    # print(graphs_list[0].ndata["token_span"])
    # print(graphs_list)
    # print(list_of_align_dict)
    return graphs_list, list_of_align_dict, list_of_exist_dict


def get_amr_data(json_path, graph_pkl_path, amr_path):
    print(json_path)
    with open(json_path, "r", encoding='utf-8') as f:
        sents = []
        done = 0
        sents = []
        
        while not done:
            line = f.readline()
            if line != '':
                data_dict = json.loads(line)
                sents.append(data_dict['tokens'])
            else:
                done = 1
    amr_parse(sents, amr_path)
    graphs, align, exist = processing_amr(amr_path, sents)
    torch.save((graphs, align, exist), graph_pkl_path)
    # return graphs


if __name__ == "__main__":
    # load_amr_to_dgl_graph(1)
    # processing_amr(1, 1)
    # get_amr_data("new.json", "new.pkl", "amr/train.pkl")
    # get_amr_data("data/try.json", "data/try_amrs_ordered.pkl", "data/try.pkl")


#     get_amr_data("./data/ace_bert/train.oneie.json", "./data/ace_bert/train_graphs.pkl", "./data/ace_bert/train_amrs.pkl")
#     get_amr_data("./data/ace_bert/dev.oneie.json", "./data/ace_bert/dev_graphs.pkl", "./data/ace_bert/dev_amrs.pkl")
#     get_amr_data("./data/ace_bert/test.oneie.json", "./data/ace_bert/test_graphs.pkl", "./data/ace_bert/test_amrs.pkl")

#     get_amr_data("./data/ace_roberta/train.oneie.json", "./data/ace_roberta/train_graphs.pkl", "./data/ace_roberta/train_amrs.pkl")
#     get_amr_data("./data/ace_roberta/dev.oneie.json", "./data/ace_roberta/dev_graphs.pkl", "./data/ace_roberta/dev_amrs.pkl")
#     get_amr_data("./data/ace_roberta/test.oneie.json", "./data/ace_roberta/test_graphs.pkl", "./data/ace_roberta/test_amrs.pkl")

#     get_amr_data("./data/ace_roberta_correct/train.oneie.json", "./data/ace_roberta_correct/train_graphs.pkl", "./data/ace_roberta_correct/train_amrs.pkl")
#     get_amr_data("./data/ace_roberta_correct/dev.oneie.json", "./data/ace_roberta_correct/dev_graphs.pkl", "./data/ace_roberta_correct/dev_amrs.pkl")
#     get_amr_data("./data/ace_roberta_correct/test.oneie.json", "./data/ace_roberta_correct/test_graphs.pkl", "./data/ace_roberta_correct/test_amrs.pkl")
    
#     get_amr_data("./data/ace_roberta_real/train.oneie.json", "./data/ace_roberta_real/train_graphs.pkl", "./data/ace_roberta_real/train_amrs.pkl")
#     get_amr_data("./data/ace_roberta_real/dev.oneie.json", "./data/ace_roberta_real/dev_graphs.pkl", "./data/ace_roberta_real/dev_amrs.pkl")
#     get_amr_data("./data/ace_roberta_real/test.oneie.json", "./data/ace_roberta_real/test_graphs.pkl", "./data/ace_roberta_real/test_amrs.pkl")

#     get_amr_data("./data/ere_bert/train.oneie.json", "./data/ere_bert/train_graphs.pkl", "./data/ere_bert/train_amrs.pkl")
#     get_amr_data("./data/ere_bert/dev.oneie.json", "./data/ere_bert/dev_graphs.pkl", "./data/ere_bert/dev_amrs.pkl")
#     get_amr_data("./data/ere_bert/test.oneie.json", "./data/ere_bert/test_graphs.pkl", "./data/ere_bert/test_amrs.pkl")

#     get_amr_data("./data/ere_roberta/train.oneie.json", "./data/ere_roberta/train_graphs.pkl", "./data/ere_roberta/train_amrs.pkl")
#     get_amr_data("./data/ere_roberta/dev.oneie.json", "./data/ere_roberta/dev_graphs.pkl", "./data/ere_roberta/dev_amrs.pkl")
#     get_amr_data("./data/ere_roberta/test.oneie.json", "./data/ere_roberta/test_graphs.pkl", "./data/ere_roberta/test_amrs.pkl")

#     get_amr_data("./data/ere_roberta_correct/train.oneie.json", "./data/ere_roberta_correct/train_graphs.pkl", "./data/ere_roberta_correct/train_amrs.pkl")
#     get_amr_data("./data/ere_roberta_correct/dev.oneie.json", "./data/ere_roberta_correct/dev_graphs.pkl", "./data/ere_roberta_correct/dev_amrs.pkl")
#     get_amr_data("./data/ere_roberta_correct/test.oneie.json", "./data/ere_roberta_correct/test_graphs.pkl", "./data/ere_roberta_correct/test_amrs.pkl")
    
#     get_amr_data("./data/ere_roberta_real/train.oneie.json", "./data/ere_roberta_real/train_graphs.pkl", "./data/ere_roberta_real/train_amrs.pkl")
#     get_amr_data("./data/ere_roberta_real/dev.oneie.json", "./data/ere_roberta_real/dev_graphs.pkl", "./data/ere_roberta_correct/dev_amrs.pkl")
#     get_amr_data("./data/ere_roberta_real/test.oneie.json", "./data/ere_roberta_real/test_graphs.pkl", "./data/ere_roberta_real/test_amrs.pkl")
    
    get_amr_data("./data/genia_2011_roberta/train.oneie.json", "./data/genia_2011_roberta/train_graphs.pkl", "./data/genia_2011_roberta/train_amrs.pkl")
    get_amr_data("./data/genia_2011_roberta/dev.oneie.json", "./data/genia_2011_roberta/dev_graphs.pkl", "./data/genia_2011_roberta/dev_amrs.pkl")
    get_amr_data("./data/genia_2011_roberta/test.oneie.json", "./data/genia_2011_roberta/test_graphs.pkl", "./data/genia_2011_roberta/test_amrs.pkl")
    
    get_amr_data("./data/genia_2011_bert/train.oneie.json", "./data/genia_2011_bert/train_graphs.pkl", "./data/genia_2011_bert/train_amrs.pkl")
    get_amr_data("./data/genia_2011_bert/dev.oneie.json", "./data/genia_2011_bert/dev_graphs.pkl", "./data/genia_2011_bert/dev_amrs.pkl")
    get_amr_data("./data/genia_2011_bert/test.oneie.json", "./data/genia_2011_bert/test_graphs.pkl", "./data/genia_2011_bert/test_amrs.pkl")
    
    get_amr_data("./data/genia_2013_bert/train.oneie.json", "./data/genia_2013_bert/train_graphs.pkl", "./data/genia_2013_bert/train_amrs.pkl")
    get_amr_data("./data/genia_2013_bert/dev.oneie.json", "./data/genia_2013_bert/dev_graphs.pkl", "./data/genia_2013_bert/dev_amrs.pkl")
    get_amr_data("./data/genia_2013_bert/test.oneie.json", "./data/genia_2013_bert/test_graphs.pkl", "./data/genia_2013_bert/test_amrs.pkl")
    
    get_amr_data("./data/genia_2013_roberta/train.oneie.json", "./data/genia_2013_roberta/train_graphs.pkl", "./data/genia_2013_roberta/train_amrs.pkl")
    get_amr_data("./data/genia_2013_roberta/dev.oneie.json", "./data/genia_2013_roberta/dev_graphs.pkl", "./data/genia_2013_roberta/dev_amrs.pkl")
    get_amr_data("./data/genia_2013_roberta/test.oneie.json", "./data/genia_2013_roberta/test_graphs.pkl", "./data/genia_2013_roberta/test_amrs.pkl")
