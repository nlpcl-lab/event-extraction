import os
from xml.etree.ElementTree import parse
import pickle
from Config import MyConfig


class PreprocessManager():
    def __init__(self):
        self.dir_list = MyConfig.raw_dir_list
        self.dir_path = MyConfig.raw_data_path
        self.vocab, self.all_labels_event, self.all_labels_role = [set() for i in range(3)]

    def preprocess(self):
        '''
        Overall Iterator for whole dataset
        '''
        fnames = self.fname_search() #list of tuple (sgm file, apf.xml file)

        instances, words, pos_tags, marks, label_event, label_role = [[] for i in range(6)]

        total_res = []
        for fname in fnames:
            res = self.process_one_file(fname) # Do something....
            total_res.append(res)
        json_result = self.Data2Json(total_res)

        with open('./data/dump.txt','wb') as f:
            pickle.dump(json_result,f)

    def fname_search(self):
        '''
        Search dataset directory & Return list of (sgm fname, apf.xml fname)
        '''
        fname_list = list()
        for dir in self.dir_list:
            full_path = self.dir_path.format(dir)
            flist = os.listdir(full_path)
            for fname in flist:
                if '.sgm' not in fname: continue
                raw = fname.split('.sgm')[0]
                fname_list.append((self.dir_path.format(dir)+raw+'.sgm',self.dir_path.format(dir)+raw+'apf.xml'))
        return fname_list

    def process_one_file(self, fname):
        pass

    def parse_one_xml(self, fname):
        pass

    def parse_one_sgm(self, fname):
        pass

    def Data2Json(self, data):
        pass

    def next_train_data(self):
        pass

    def eval_data(self):
        pass


if __name__ == '__main__':
    man = PreprocessManager()
    man.preprocess()

