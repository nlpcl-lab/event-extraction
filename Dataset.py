import numpy as np
from collections import namedtuple

def find_candidates(items1, items2):
    result = []
    for i in range(len(items1)):
        if items1[i] in items2:
            result.append(i)
    return result

def one_hot(labels, label_num):
    result = []
    for i in range(len(labels)):
        one_hot_vec = [0] * label_num
        one_hot_vec[labels[i][0]] = 1
        result.append(one_hot_vec)
    return result

class Dataset:
    def __init__(self,
                 data_path='',
                 batch_size=5,
                 max_sequence_length=20,
                 windows=3,
                 eval_num=50):
        all_words, all_pos_taggings, all_labels, all_marks = [set() for _ in range(4)]

        instances = []
        words = []
        marks = []
        label = []

        # data_model = namedtuple(('data'), ['words', 'pos_taggings', 'marks', 'label'])
        # instances.append(data_model(words=words, pos_taggings=pos_taggings, marks=marks, label=label))

        all_words.add('<eos>')
        all_pos_taggings.add('*')
        words_size = len(all_words)
        word_id = dict(zip(all_words, range(words_size)))
        pos_taggings_size = len(all_pos_taggings)
        pos_taggings_id = dict(zip(all_pos_taggings, range(pos_taggings_size)))

        labels_size = len(all_labels)
        mark_size = len(all_marks)
        mark_id = dict(zip(all_marks, range(mark_size)))

        self.windows = windows
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length

        self.all_words = list(all_words)
        self.all_pos_taggings = list(all_pos_taggings)
        self.all_marks = list(all_marks)
        self.all_labels = list(all_labels)

        self.words_size = words_size
        self.pos_taggings_size = pos_taggings_size
        self.labels_size = labels_size
        self.mark_size = mark_size

        self.word_id = word_id
        self.pos_taggings_id = pos_taggings_id
        self.mark_id = mark_id

        self.eval_num = eval_num
        self.eval_instances = instances[-eval_num:]

        instances = instances[0:-eval_num]
        self.instances_size = len(instances)
        self.instances = instances
        self.batch_nums = self.instances_size // self.batch_size
        self.index = np.arange(self.instances_size)
        self.point = 0

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.instances_size:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])
        return batch_instances

    def next_train_data(self):
        batch_instances = self.next_batch()
        pos_tag = []
        y = []
        x = []
        t = []
        c = []
        pos_c = []
        pos_t = []

        c_context = []
        t_context = []

        for instance in batch_instances:
            words = instance.words
            pos_taggings = instance.pos_taggings
            marks = instance.marks
            label = instance.label

            index_candidates = find_candidates(marks, ['B'])
            assert (len(index_candidates)) == 1
            index_triggers = find_candidates(marks, ['T'])
            assert (len(index_triggers)) == 1
            y.append(label)
            marks = marks + ['A'] * (self.max_sequence_length - len(marks))
            words = words + ['<eos>'] * (self.max_sequence_length - len(words))
            pos_taggings = pos_taggings + ['*'] * (self.max_sequence_length - len(pos_taggings))
            pos_taggings = map(lambda x: self.pos_taggings_id[x], pos_taggings)
            pos_tag.append(pos_taggings)
            index_words = map(lambda x: self.word_id[x], words)
            x.append(index_words)

            pos_candidate = range(-index_candidates[0], 0) + range(0, self.max_sequence_length - index_candidates[0])
            pos_c.append(pos_candidate)
            pos_trigger = range(-index_triggers[0], 0) + range(0, self.max_sequence_length - index_triggers[0])
            pos_t.append(pos_trigger)

            t.append([index_words[index_triggers[0]]] * self.max_sequence_length)
            c.append([index_words[index_candidates[0]]] * self.max_sequence_length)

            assert len(words) == len(marks) == len(pos_taggings) == len(index_words) == len(pos_candidate) == len(pos_trigger)

        assert len(y) == len(x) == len(t) == len(c) == len(pos_c) == len(pos_t) == len(
            pos_tag)

        return x, t, c, one_hot(y, self.labels_size), pos_c, pos_t, c_context, t_context, pos_tag
