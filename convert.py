import os
import glob
import json

cur_dir = os.path.dirname(os.path.realpath(__file__))

entity_type_mapping_file = os.path.join(cur_dir, 'resource', 'ace_to_aida_entity.tsv')
event_type_mapping_file = os.path.join(cur_dir, 'resource', 'ace_to_aida_event.tsv')
role_type_mapping_file = os.path.join(cur_dir, 'resource', 'ace_to_aida_role.tsv')
relation_type_mapping_file = os.path.join(cur_dir, 'resource', 'ace_to_aida_relation.tsv')

def load_mapping(mapping_file):
    mapping = {}
    with open(mapping_file, 'r', encoding='utf-8') as r:
        for line in r:
            from_type, to_type = line.strip().split('\t')
            mapping[from_type] = to_type
    return mapping


def get_span_mention_text(tokens, token_ids, start, end):
    if start + 1 == end:
        return tokens[start], token_ids[start]

    start_token = tokens[start]
    end_token = tokens[end - 1]
    start_char = int(token_ids[start].split(':')[1].split('-')[0])
    end_char = int(token_ids[end - 1].split(':')[1].split('-')[1])
    text = ' ' * (end_char - start_char + 1)
    for token, token_id in zip(tokens[start:end], token_ids[start:end]):
        token_start, token_end = token_id.split(':')[1].split('-')
        token_start, token_end = int(token_start), int(token_end)
        token_start -= start_char
        token_end -= start_char
        assert len(text[:token_start] + token + text[token_end + 1:]) == len(text)
        text = text[:token_start] + token + text[token_end + 1:]
    return text, '{}:{}-{}'.format(token_ids[start].split(':')[0],
                                   start_char, end_char)


def json_to_cs(input_dir, output_dir):
    # TODO: add the first cs line
    entity_type_mapping = load_mapping(entity_type_mapping_file)
    relation_type_mapping = load_mapping(relation_type_mapping_file)
    event_type_mapping = load_mapping(event_type_mapping_file)
    role_type_mapping = load_mapping(role_type_mapping_file)

    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    # convert entities
    print('Converting entity mentions and generate entity cs file')
    entity_mapping = {}
    entity_id_mapping = {}
    entity_cs_file = os.path.join(output_dir, 'entity.cs')
    with open(entity_cs_file, 'w', encoding='utf-8') as w:
        for f in json_files:
            with open(f, 'r', encoding='utf-8') as r:
                for line in r:
                    result = json.loads(line)
                    doc_id = result['doc_id']
                    sent_id = result['sent_id']
                    tokens, token_ids = result['tokens'], result['token_ids']
                    for i, (start, end, enttype, mentype, _) in enumerate(result['graph']['entities']):
                        entity_text, entity_span = get_span_mention_text(
                            tokens, token_ids, start, end)
                        entity_id = 'Entity_EDL_{:07d}'.format(len(entity_mapping) + 1)
                        entity_mapping[(sent_id, i)] = (entity_text, entity_id, entity_span, enttype, mentype)
                        entity_id_mapping[entity_id] = (sent_id, i)
                        enttype_mapped = entity_type_mapping[enttype]
                        w.write(':{}\ttype\t{}\t1.000000\n'.format(entity_id, enttype_mapped))
                        w.write(':{}\tcanonical_mention\t"{}"\t{}\t0.000\n'.format(
                            entity_id, entity_text, entity_span))
                        w.write(':{}\tmention\t"{}"\t{}\t0.000\n'.format(
                            entity_id, entity_text, entity_span))
                        # skip the link line
    
    # converting relations and events
    print('Converting relations and events')
    event_count = 0
    relation_cs_file = os.path.join(output_dir, 'relation.cs')
    event_cs_file = os.path.join(output_dir, 'event.cs')
    with open(relation_cs_file, 'w', encoding='utf-8') as rel_w, \
        open(event_cs_file, 'w', encoding='utf-8') as evt_w:
        for f in json_files:
            with open(f, 'r', encoding='utf-8') as r:
                for line in r:
                    result = json.loads(line)
                    sent_id = result['sent_id']
                    tokens, token_ids = result['tokens'], result['token_ids']
                    relations = result['graph']['relations']
                    triggers = result['graph']['triggers']
                    roles = result['graph']['roles']
                    # sentence span
                    sent_span = '{}:{}-{}'.format(
                        token_ids[0].split(':')[0],
                        token_ids[0].split(':')[1].split('-')[0],
                        token_ids[-1].split(':')[1].split('-')[1])
                    # convert relations
                    for arg1, arg2, reltype, _ in relations:
                        if reltype == 'ART':
                            continue
                        entity_id_1 = entity_mapping[(sent_id, arg1)][1]
                        entity_id_2 = entity_mapping[(sent_id, arg2)][1]
                        reltype_mapped = relation_type_mapping[reltype]
                        rel_w.write(':{}\t{}\t:{}\t{}\t1.000\n'.format(
                            entity_id_1, reltype_mapped, entity_id_2, sent_span
                        ))
                    # convert events
                    for cur_trigger_idx, (start, end, eventtype, _) in enumerate(triggers):
                        event_count += 1
                        event_id = 'Event_{:06d}'.format(event_count)
                        trigger_text, trigger_span = get_span_mention_text(
                            tokens, token_ids, start, end)
                        eventtype_mapped = event_type_mapping[eventtype]
                        evt_w.write(':{}\ttype\t{}\n'.format(event_id, eventtype_mapped))
                        evt_w.write(':{}\tmention.actual\t"{}"\t{}\t1.000\n'.format(
                            event_id, trigger_text, trigger_span))
                        evt_w.write(':{}\tcanonical_mention.actual\t"{}"\t{}\t1.000\n'.format(
                            event_id, trigger_text, trigger_span))
                        for trigger_idx, entity_idx, role, _ in roles:
                            if cur_trigger_idx == trigger_idx:
                                role_mapped = role_type_mapping['{}:{}'.format(eventtype, role).lower()]
                                _, entity_id, entity_span, _, _ = entity_mapping[(sent_id, entity_idx)]
                                evt_w.write(':{}\t{}.actual\t{}\t{}\t1.000\n'.format(
                                    event_id, role_mapped, entity_id, entity_span))