import os
from xml.etree.ElementTree import parse
from Config import MyConfig

class Preprocess:
    def __init__(self):
        self.dir_list = MyConfig.raw_dir_list
        self.dir_path = MyConfig.raw_data_path

    def train_data(self):
        pass

    def test_data(self):
        pass


if __name__ == '__main__':
    a = Preprocess()

