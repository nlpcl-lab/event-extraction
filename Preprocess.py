import os
import xml.etree.ElementTree as ET
import pickle
from Config import MyConfig
import pprint
from bs4 import BeautifulSoup
import json

pp = pprint.PrettyPrinter(indent=4)


class PreprocessManager():
    def __init__(self):
        self.dir_list = MyConfig.raw_dir_list
        self.dir_path = MyConfig.raw_data_path
        self.dataset = []
        self.tri_task_format_data = []
        self.arg_task_format_data = []

    def preprocess(self):
        '''
        Overall Iterator for whole dataset
        '''
        fnames = self.fname_search()
        print('Total XML file: {}'.format(len(fnames)))
        total_res = []
        for fname in fnames:
            total_res.append(self.process_one_file(fname))
        print('total_event: {}개'.format(len(total_res)))

        for doc in total_res:
            self.dataset += self.process_sentencewise(doc)
        print("END PREPROCESSING")
        print('TOTAL DATA :  {}'.format(len(self.dataset)))
        self.format_to_trigger()
        self.format_to_argument()

        print('TRIGGER DATASET: {}\nARGUMENT DATASET: {}\n'.format(len(self.tri_task_format_data),
                                                                   len(self.arg_task_format_data)))

    def format_to_argument(self):
        for item in self.dataset:
            d = item[0]
            fname = item[1]
            generated_candi = self.generate_argument_candidate_pos_list(d['argument_position'], d['entity_position'],
                                                                        d['trigger_position'])
            # TODO: get SeunWon's variable and change 100.
            if len(d['sentence'])>100:continue

            trigger_cnt = 0
            for m in d['trigger_position']:
                if m=='T':trigger_cnt+=1
            if trigger_cnt>1:continue

            for candi in generated_candi:
                self.arg_task_format_data.append([d['sentence']]+candi+[fname])

    def generate_argument_candidate_pos_list(self, arg_pos, enti_pos, trigger_pos):
        cand_list = []
        Entity_as_candidate_only = True  # Entity만 Candidates로 사용

        assert len(arg_pos)==len(enti_pos)==len(trigger_pos)
        for idx,el in enumerate(arg_pos):
            if Entity_as_candidate_only:
                if enti_pos[idx]!='E': continue
            if trigger_pos[idx]!='*': continue

            tri_idx_list = []
            for j,a in enumerate(trigger_pos):
                if a != '*': tri_idx_list.append(j)

            marks = ['A' for i in range(len(arg_pos))]
            marks[idx]='B'
            for i in tri_idx_list:
                marks[i]='T'
            label = 'None' if arg_pos[idx]=='*' else arg_pos[idx]

            '''
            Time-After,  Time-At-End,  Time-Ending , Time-Holds , Time-At-Beginning, Time-Before,  Time-Within, Time-Starting to Time
            '''
            if 'Time-' in label: label = 'Time'
            cand_list.append([marks,label])
        return cand_list

    def format_to_trigger(self):
        for item in self.dataset:
            d = item[0]
            fname = item[1]
            generated_candi = self.generate_trigger_candidate_pos_list(d['trigger_position'])
            if len(d['sentence'])>100:continue
            for candi in generated_candi:
                self.tri_task_format_data.append([d['sentence']]+candi+[fname])

    def generate_trigger_candidate_pos_list(self, trigger_pos):
        cand_list = []
        idx_list = []
        for idx,el in enumerate(trigger_pos):
            if el!='*': idx_list.append((idx,el))

        for idx in range(len(trigger_pos)):
            marks = ['A' for i in range(len(trigger_pos))]
            marks[idx]='B'
            label = 'None'
            for i in idx_list:
                if idx == i[0]: label = i[1]
            cand_list.append([marks,label])
        return cand_list

    def process_sentencewise(self, doc):
        entities, val_timexs, events, xml_fname = doc
        datas = []
        for event in events:
            for e_mention in event['event_mention']:
                tmp = {'TYPE': event['TYPE'], 'SUBTYPE': event['SUBTYPE']}
                tmp['raw_sent'] = e_mention['ldc_scope']['text']
                sent_pos = [int(i) for i in e_mention['ldc_scope']['position']]
                entities_in_sent = self.search_entity_in_sentence(entities, sent_pos)
                val_timexs_in_sent = self.search_valtimex_in_sentence(val_timexs, sent_pos)
                e_mention = self.get_argument_head(entities_in_sent, e_mention)
                res = self.packing_sentence(e_mention, tmp, sent_pos, entities_in_sent, val_timexs_in_sent)
                if res!=1: datas.append([res,xml_fname])
        return datas

    @staticmethod
    def get_argument_head(entities, e_mention):
        for idx, arg in enumerate(e_mention['argument']):
            arg_refID = arg['REFID']
            for entity in entities:
                if entity['ID'] == arg_refID:
                    e_mention['argument'][idx]['position_head'] = entity['head']['position']
                    e_mention['argument'][idx]['text_head'] = entity['head']['text']
        return e_mention

    def packing_sentence(self, e_mention, tmp, sent_pos, entities, valtimexes):
        packed_data = {
            'sentence': [],
            'EVENT_TYPE' : tmp['TYPE'],
            'EVENT_SUBTYPE' : tmp['SUBTYPE'],
            'entity_position' : [],
        }
        # Each Entity, value, timex2 overlap check
        assert self.check_entity_overlap(entities, valtimexes)
        raw_sent = e_mention['ldc_scope']['text']

        idx_list = [0 for i in range(len(e_mention['ldc_scope']['text']))]
        if not (len(idx_list) == (int(e_mention['ldc_scope']['position'][1])-int(e_mention['ldc_scope']['position'][0])+1)):
            return 1
        sent_start_idx = int(e_mention['ldc_scope']['position'][0])

        # Mark Entity position
        for ent in entities:
            ent_start_idx = int(ent['head']['position'][0])
            for i in range(int(ent['head']['position'][1]) - int(ent['head']['position'][0]) + 1):
                if idx_list[ent_start_idx + i - sent_start_idx]==1: raise ValueError('까율~~~~~~~~~~~~~~~~~~')
                idx_list[ent_start_idx + i - sent_start_idx] = 1  # entity mark

        # Mark Value&Timex2 position
        for val in valtimexes:
            ent_start_idx = int(val['position'][0])
            for i in range(int(val['position'][1]) - int(val['position'][0]) + 1):
                idx_list[ent_start_idx + i - sent_start_idx] = 1  # entity mark

        token_list = []
        entity_mark_list = []
        curr_token = ''
        # TODO: save each mark as variable, not to type 'N' or 'E' each time.
        for idx, el in enumerate(e_mention['ldc_scope']['text']):
            if idx==0:
                curr_token += el
                continue
            if idx_list[idx]!=idx_list[idx-1]:
                if idx_list[idx-1]==1: entity_mark_list.append('E')
                else: entity_mark_list.append('*')
                token_list.append(curr_token)
                curr_token = el
                continue
            curr_token += el
            if idx == len(e_mention['ldc_scope']['text'])-1:
                if idx_list[idx]==1: entity_mark_list.append('E')
                else: entity_mark_list.append('*')
                token_list.append(curr_token)

        assert len(token_list)==len(entity_mark_list)
        good_token_list = []  # TODO: The better name....
        good_entity_mark_list = []

        for tok, mark in zip(token_list, entity_mark_list):
            if mark == '*':
                splitted_tok = tok.split()
                good_token_list += splitted_tok
                good_entity_mark_list += ['*' for i in range(len(splitted_tok))]
            if mark == 'E':
                good_token_list.append(tok.strip())
                good_entity_mark_list.append('E')
        assert len(good_entity_mark_list)==len(good_token_list)

        argument_role_label = ['*' for i in range(len(good_entity_mark_list))]
        for arg in e_mention['argument']:
            if 'text_head' in arg:
                arg_text,arg_role = arg['text_head'],arg['ROLE']
            else:
                arg_text,arg_role = arg['text'],arg['ROLE']

            #assert good_token_list.count(arg_text) in [0,1,2]  # 단 한번!
            # TODO: Move this part to up
            arg_idx = None
            if arg_text not in good_token_list:
                for idx,el in enumerate(good_token_list):
                    if arg_text in el:
                        arg_idx = idx
                        break
            else:
                arg_idx = good_token_list.index(arg_text)
            if arg_idx==None:
                print('Exception')
                return 1
            argument_role_label[arg_idx] = arg_role

        trigger_idx = None
        trigger_type_label = ['*' for i in range(len(good_entity_mark_list))]
        if e_mention['anchor']['text'] in good_token_list:
            trigger_idx = [good_token_list.index(e_mention['anchor']['text'])]
        else:
            for idx,tok in enumerate(token_list):
                if e_mention['anchor']['text'] in tok:
                    trigger_idx = [idx]
        if trigger_idx==None:  # multiple trigger like 'blew him up'
            triggers = e_mention['anchor']['text'].split()
            trigger_idx = []
            for tmp_t in triggers:
                if tmp_t in token_list: trigger_idx.append(token_list.index(tmp_t))
                else:
                    for idx, tok in enumerate(token_list):
                        if tmp_t in tok:
                            trigger_idx.append(idx)

        for el in trigger_idx:
            trigger_type_label[el] = tmp['TYPE']+'/'+tmp['SUBTYPE']

        assert len(good_entity_mark_list)==len(good_token_list)==len(trigger_type_label)==len(argument_role_label)
        packed_data['sentence'] = good_token_list
        packed_data['trigger_position'] = trigger_type_label
        packed_data['entity_position'] = good_entity_mark_list
        packed_data['argument_position'] = argument_role_label
        return packed_data

    @staticmethod
    def check_entity_overlap(entities, valtimexes):
        ranges = []
        # TODO: Implement this later
        for ent in entities:
            ranges.append(None)
        return True

    @staticmethod
    def search_entity_in_sentence(entities, sent_pos):
        headVSextent = 'head'
        entities_in_sent = list()
        check = dict()
        for entity in entities:
            for mention in entity['mention']:
                if sent_pos[0] <= int(mention[headVSextent]['position'][0]) and int(mention[headVSextent]['position'][1]) <= sent_pos[1]:
                    if mention[headVSextent]['position'][0] in check:  # duplicate entity in one word.
                        #print('으악!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        #raise ValueError
                        continue
                    check[mention[headVSextent]['position'][0]] = 1
                    entities_in_sent.append(mention)
        return entities_in_sent

    @staticmethod
    def search_valtimex_in_sentence(valtimex, sent_pos):
        valtimex_in_sent = list()
        for item in valtimex:
            for mention in item['mention']:
                if sent_pos[0] <= int(mention['position'][0]) and sent_pos[1] >= int(mention['position'][1]):
                    valtimex_in_sent.append(mention)
        return valtimex_in_sent

    def fname_search(self):
        '''
        Search dataset directory & Return list of (sgm fname, apf.xml fname)
        '''
        fname_list = list()
        for dir in self.dir_list:
            # To exclude hidden files
            if len(dir) and dir[0] == '.': continue
            full_path = self.dir_path.format(dir)
            flist = os.listdir(full_path)
            for fname in flist:
                if '.sgm' not in fname: continue
                raw = fname.split('.sgm')[0]
                fname_list.append((self.dir_path.format(dir) + raw + '.sgm', self.dir_path.format(dir) + raw + '.apf.xml'))
        return fname_list

    def process_one_file(self, fname):
        # args fname = (sgm fname(full path), xml fname(full path))
        # return some multiple [ sentence, entities, event mention(trigger + argument's information]
        xml_ent_res, xml_valtimex_res, xml_event_res = self.parse_one_xml(fname[1])
        # sgm_ent_res, sgm_event_res = self.parse_one_sgm(fname[0])
        # TODO : merge xml and sgm file together if need.
        return xml_ent_res, xml_valtimex_res, xml_event_res, fname[1]

    def parse_one_xml(self, fname):
        tree = ET.parse(fname)
        root = tree.getroot()
        entities, val_timex, events = [], [], []

        for child in root[0]:
            if child.tag == 'entity':
                entities.append(self.xml_entity_parse(child, fname))
            if child.tag in ['value', 'timex2']:
                val_timex.append(self.xml_value_timex_parse(child, fname))
            if child.tag == 'event':
                events.append(self.xml_event_parse(child, fname))
        return entities, val_timex, events

    def xml_value_timex_parse(self, item, fname):
        child = item.attrib
        child['fname'] = fname
        child['mention'] = []
        for sub in item:
            mention = sub.attrib
            mention['position'] = [sub[0][0].attrib['START'], sub[0][0].attrib['END']]
            mention['text'] = sub[0][0].text
            child['mention'].append(mention)
        return child

    def xml_entity_parse(self, item, fname):
        entity = item.attrib
        entity['fname'] = fname
        entity['mention'] = []
        entity['attribute'] = []  # What is this exactly?
        for sub in item:
            if sub.tag != 'entity_mention': continue
            mention = sub.attrib
            for el in sub:  # charseq and head
                mention[el.tag] = dict()
                mention[el.tag]['position'] = [el[0].attrib['START'], el[0].attrib['END']]
                mention[el.tag]['text'] = el[0].text
            entity['mention'].append(mention)
        return entity

    def xml_event_parse(self, item, fname):
        #  event: one event item
        event = item.attrib
        event['fname'] = fname
        event['argument'] = []
        event['event_mention'] = []
        for sub in item:
            if sub.tag == 'event_argument':
                tmp = sub.attrib
                event['argument'].append(tmp)
                continue
            if sub.tag == 'event_mention':
                mention = sub.attrib  # init dict with mention ID
                mention['argument'] = []
                for el in sub:
                    if el.tag == 'event_mention_argument':
                        one_arg = el.attrib
                        one_arg['position'] = [el[0][0].attrib['START'], el[0][0].attrib['END']]
                        one_arg['text'] = el[0][0].text
                        mention['argument'].append(one_arg)
                    else:  # [extent, ldc_scope, anchor] case
                        for seq in el:
                            mention[el.tag] = dict()
                            mention[el.tag]['position'] = [seq.attrib['START'], seq.attrib['END']]
                            mention[el.tag]['text'] = seq.text
                event['event_mention'].append(mention)
        return event

    def parse_one_sgm(self, fname):
        print('fname :', fname)
        with open(fname, 'r') as f:
            data = f.read()
            soup = BeautifulSoup(data, features='html.parser')

            doc = soup.find('doc')
            doc_id = doc.docid.text
            doc_type = doc.doctype.text.strip()
            date_time = doc.datetime.text
            headline = doc.headline.text if doc.headline else ''

            body = []

            if doc_type == 'WEB TEXT':
                posts = soup.findAll('post')
                for post in posts:
                    poster = post.poster.text
                    post.poster.extract()
                    post_date = post.postdate.text
                    post.postdate.extract()
                    subject = post.subject.text if post.subject else ''
                    if post.subject: post.subject.extract()
                    text = post.text
                    body.append({
                        'poster': poster,
                        'post_date': post_date,
                        'subject': subject,
                        'text': text,
                    })
            elif doc_type in ['STORY', 'CONVERSATION', 'NEWS STORY']:
                turns = soup.findAll('turn')
                for turn in turns:
                    speaker = turn.speaker.text if turn.speaker else ''
                    if turn.speaker: turn.speaker.extract()
                    text = turn.text
                    body.append({
                        'speaker': speaker,
                        'text': text,
                    })

            result = {
                'doc_id': doc_id,
                'doc_type': doc_type,
                'date_time': date_time,
                'headline': headline,
                'body': body,
            }

            return result

    def Data2Json(self, data):
        pass

    def next_train_data(self):
        pass

    def eval_data(self):
        pass


if __name__ == '__main__':
    man = PreprocessManager()
    man.preprocess()

    # Example
    trigger_classification_data = man.tri_task_format_data
    argument_classification_data = man.arg_task_format_data
    # print('\n\n')


    all_labels = set()
    total = 0
    for data in argument_classification_data:
        total += 1
        all_labels.add(data[2])

    print('total :', total)
    print('label len:', len(all_labels))


