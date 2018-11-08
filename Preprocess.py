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
        self.vocab, self.all_labels_event, self.all_labels_role = [set() for i in range(3)]

    def preprocess(self):
        '''
        Overall Iterator for whole dataset
        '''
        fnames = self.fname_search()  # list of tuple (sgm file, apf.xml file)
        instances, words, pos_tags, marks, label_event, label_role = [[] for i in range(6)]
        total_res = []
        for fname in fnames:
            res = self.process_one_file(fname)  # Do something....
            total_res += res
        json_result = self.Data2Json(total_res)
        with open('./data/dump.txt', 'wb') as f:
            pickle.dump(json_result, f)

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
        # Do something....
        self.parse_one_xml(fname[1])
        self.parse_one_sgm(fname[0])

    def parse_one_xml(self, fname):
        tree = ET.parse(fname)
        root = tree.getroot()

        entities, events = [], []

        for child in root[0]:
            if child.tag == 'entity':
                entities.append(self.xml_entity_parse(child))

            if child.tag == 'event':
                events.append(self.xml_event_parse(child))

    def xml_entity_parse(self, item):
        entity = item.attrib
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

    def xml_event_parse(self, item):
        #  event: one event item
        event = item.attrib
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
            headline = doc.headline.text

            body = []

            if doc_type == 'WEB TEXT':
                posts = soup.findAll('post')
                for post in posts:
                    poster = post.poster.text
                    post.poster.extract()
                    post_date = post.postdate.text
                    post.postdate.extract()
                    subject = post.subject.text
                    post.subject.extract()
                    text = post.text
                    body.append({
                        'poster': poster,
                        'post_date': post_date,
                        'subject': subject,
                        'text': text,
                    })
            elif doc_type == 'STORY':
                turns = soup.findAll('turn')
                for turn in turns:
                    speaker = turn.speaker.text
                    turn.speaker.extract()
                    text = turn.text
                    body.append({
                        'speaker': speaker,
                        'text': text,
                    })
            else:
                print('another type! :', doc_type)

    def Data2Json(self, data):
        pass

    def next_train_data(self):
        pass

    def eval_data(self):
        pass


if __name__ == '__main__':
    man = PreprocessManager()
    # man.preprocess()

    man.parse_one_sgm('./data/ace_2005_td_v7/data/English/bc/adj/CNN_CF_20030303.1900.00.sgm')
