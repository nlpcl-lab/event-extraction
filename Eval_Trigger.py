import datetime, os, time
import numpy as np
import tensorflow as tf
from Dataset_Trigger import Dataset_Trigger as TRIGGER_DATASET
from Config import HyperParams_Tri_classification as hp
import nltk

def get_batch(sentence, word_id, max_sequence_length):
    words = [word for word in nltk.word_tokenize(sentence)]
    words = words + ['<eos>'] * (max_sequence_length - len(words))

    word_ids = []
    for word in words:
        if word in word_id:
            word_ids.append(word_id[word])
        else:
            word_ids.append(word_id['<unk>'])

    print('word_ids :', word_ids)
    size = len(word_ids)

    x_batch = []
    x_pos_batch = []
    for i in range(size):
        x_batch.append(word_ids)
        x_pos_batch.append([j - i for j in range(size)])

    return x_batch, x_pos_batch

if __name__ == '__main__':
    dataset = TRIGGER_DATASET(batch_size=hp.batch_size, max_sequence_length=hp.max_sequence_length,
                              windows=hp.windows, dtype='IDENTIFICATION')

    x_batch, x_pos_batch = get_batch(sentence = 'It could swell to as much as $500 billion if we go to war in Iraq',
                                     word_id = dataset.word_id, max_sequence_length=hp.max_sequence_length)

    print('x_batch :', x_batch)
    print('x_pos_batch :', x_pos_batch)

    checkpoint_dir = './runs/1542830074/checkpoints'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_c_pos = graph.get_operation_by_name("input_c_pos").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predicts").outputs[0]

            feed_dict = {
                input_x: x_batch,
                input_c_pos: x_pos_batch,
                dropout_keep_prob: 1.0,
            }

            pred = sess.run(predictions, feed_dict)
            print('pred :', pred)
