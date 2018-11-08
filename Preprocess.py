import os
from xml.etree.ElementTree import parse
from Config import MyConfig

class PreprocessManager():
    def __init__(self):
        self.dir_list = MyConfig.raw_dir_list
        self.dir_path = MyConfig.raw_data_path

    def Preprocess(self,doc):
        pass



    def process_one_doc(self,path,docname):
        pass

    def parse_one_xml(self,path,docname):
        pass

    def parse_one_sgm(self,path,docname):
        pass

    def train_data(self):
        pass

    def test_data(self):
        pass


if __name__ == '__main__':
    a = PreprocessManager()

