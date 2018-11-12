import os

class MyConfig:
    raw_data_path = './data/ace_2005_td_v7/data/English/{}/adj/'
    raw_dir_list = os.listdir('./data/ace_2005_td_v7/data/English/')


class HyperParams:
    batch_size = 30
    max_sequence_length = 80
    windows = 3
    word_embedding_size = 100
    pos_embedding_size = 10

    filter_sizes = [3, 4, 5]
    filter_num = 100

    num_epochs = 20
