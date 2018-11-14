import random
import numpy as np
from Util import one_hot, find_candidates
from Config import MyConfig,HyperParams_Tri_classification

class Dataset_Trigger:
    def __init__(self,
                 data_path='',
                 batch_size=30,
                 max_sequence_length=30,
                 windows=3,
                 eval_num=30,
                 dtype=None):
        assert dtype in ['IDENTIFICATION','CLASSIFICATION']

        self.windows = windows
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.eval_num = eval_num
        self.dtype = dtype

        self.all_words = list()
        self.all_pos_taggings = list()
        self.all_marks = list()
        self.all_labels = list()
        self.instances = list()

        self.word_id = dict()
        self.pos_taggings_id = dict()
        self.mark_id = dict()
        self.label_id = dict()

        print('read data...',end=' ')
        self.read_dataset()
        print('complete')

        self.word_embed = self.embed_manager()

        self.eval_instances, self.train_instances = [],[]
        self.divide_train_eval_data()
        self.batch_nums = len(self.train_instances) // self.batch_size
        self.index = np.arange(len(self.train_instances))
        self.point = 0
        print('all label for dataset: {}'.format(len(self.all_labels)))

    def embed_manager(self):
        matrx = np.zeros([len(self.all_words), HyperParams_Tri_classification.word_embedding_size])

        for idx,word in enumerate(self.all_words):


    def load_glove(self):
        with open(MyConfig.glove_txt_path,'r',encoding='utf8') as f:
            ls = f.readlines()
            print(ls[0])


    def divide_train_eval_data(self):
        testset_fname = []

        # select test set randomly
        random.shuffle(self.instances)

        for ins in self.instances:
            #if 'nw/adj' not in ins['fname']: self.train_instances.append(ins)
            if ins['fname'] in testset_fname: self.eval_instances.append(ins)
            elif len(testset_fname)>45: self.train_instances.append(ins)
            else:
                testset_fname.append(ins['fname'])
                self.eval_instances.append(ins)

        print('TRAIN: {} TEST: {}'.format(len(self.train_instances),len(self.eval_instances)))
        random.shuffle(self.train_instances)
        assert len(self.instances)==(len(self.train_instances)+len(self.eval_instances))


    def read_dataset(self):
        all_words, all_pos_taggings, all_labels, all_marks = [set() for _ in range(4)]

        def read_one(words, marks, label, fname):
            # TODO: remove comments mark when use POS tag info for model. `nltk.pos_tag()` method too slow.
            #pos_taggings = nltk.pos_tag(words)
            #pos_taggings = [pos_tagging[1] for pos_tagging in pos_taggings]
            pos_taggings = [None for i in range(1)]
            for word in words: all_words.add(word)
            for mark in marks: all_marks.add(mark)
            for pos_tag in pos_taggings: all_pos_taggings.add(pos_tag)
            all_labels.add(label)

            if len(words) >80:
                #print('len(word) > 80, Goodbye! ', len(words), words)
                return

            self.instances.append({
                'words': words,
                'pos_taggings': pos_taggings,
                'marks': marks,
                'label': label,
                'fname':fname
            })

        from Preprocess import PreprocessManager
        man = PreprocessManager()
        man.preprocess(tasktype='TRIGGER',subtasktype=self.dtype)
        tri_classification_data = man.tri_task_format_data
        for data in tri_classification_data:
            read_one(words=data[0], marks=data[1], label=data[2], fname=data[3])

        all_words.add('<eos>')
        all_words.add('<unk>')
        all_pos_taggings.add('*')

        self.word_id = dict(zip(all_words, range(len(all_words))))
        self.pos_taggings_id = dict(zip(all_pos_taggings, range(len(all_pos_taggings))))
        self.mark_id = dict(zip(all_marks, range(len(all_marks))))
        self.label_id = dict(zip(all_labels, range(len(all_labels))))

        self.all_words = list(all_words)
        self.all_pos_taggings = list(all_pos_taggings)
        self.all_labels = list(all_labels)
        self.all_marks = list(all_marks)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > len(self.train_instances):
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.train_instances[x], self.index[start:end])
        return batch_instances

    def next_train_data(self):
        batch_instances = self.next_batch()
        pos_tag, y, x, c, pos_c = [list() for _ in range(5)]

        for instance in batch_instances:
            words = instance['words']
            pos_taggings = instance['pos_taggings']
            marks = instance['marks']
            label = instance['label']

            index_candidates = find_candidates(marks, ['B'])
            assert (len(index_candidates)) == 1

            y.append(label)
            marks = marks + ['A'] * (self.max_sequence_length - len(marks))
            words = words + ['<eos>'] * (self.max_sequence_length - len(words))
            pos_taggings = pos_taggings + ['*'] * (self.max_sequence_length - len(pos_taggings))
            pos_taggings = list(map(lambda x: self.pos_taggings_id[x], pos_taggings))
            pos_tag.append(pos_taggings)
            index_words = list(map(lambda x: self.word_id[x], words))
            x.append(index_words)
            pos_candidate = [i for i in range(-index_candidates[0], 0)] + [i for i in range(0, self.max_sequence_length - index_candidates[0])]
            pos_c.append(pos_candidate)
            c.append([index_words[index_candidates[0]]] * self.max_sequence_length)
            assert len(words) == len(marks) == len(pos_taggings) == len(index_words) == len(pos_candidate)

        assert len(y) == len(x) == len(c) == len(pos_c) == len(pos_tag)
        return x, c, one_hot(y, self.label_id, len(self.all_labels)), pos_c, pos_tag

    def next_eval_data(self):
        batch_instances = self.eval_instances
        pos_tag, y, x, c, pos_c= [list() for _ in range(5)]

        for instance in batch_instances:
            words = instance['words']
            pos_taggings = instance['pos_taggings']
            marks = instance['marks']
            label = instance['label']

            index_candidates = find_candidates(marks, ['B'])
            assert (len(index_candidates)) == 1

            y.append(label)
            marks = marks + ['A'] * (self.max_sequence_length - len(marks))
            words = words + ['<eos>'] * (self.max_sequence_length - len(words))
            pos_taggings = pos_taggings + ['*'] * (self.max_sequence_length - len(pos_taggings))
            pos_taggings = list(map(lambda x: self.pos_taggings_id[x], pos_taggings))
            pos_tag.append(pos_taggings)
            index_words = list(map(lambda x: self.word_id[x], words))
            x.append(index_words)
            pos_candidate = [i for i in range(-index_candidates[0], 0)] + [i for i in range(0,
                                                                                            self.max_sequence_length -
                                                                                            index_candidates[0])]
            pos_c.append(pos_candidate)
            c.append([index_words[index_candidates[0]]] * self.max_sequence_length)
            assert len(words) == len(marks) == len(pos_taggings) == len(index_words) == len(pos_candidate)
        assert len(y) == len(x) == len(c) == len(pos_c) == len(pos_tag)
        return x, c, one_hot(y, self.label_id, len(self.all_labels)), pos_c, pos_tag

if __name__ == '__main__':
    D = Dataset_Trigger()
    a = D.next_train_data()
