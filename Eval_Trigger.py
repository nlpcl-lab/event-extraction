import datetime, os, time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
from Dataset_Trigger import Dataset_Trigger as TRIGGER_DATASET
from Config import HyperParams_Tri_classification as hp
from Model_Trigger import Model

import Visualize

if __name__ == '__main__':
    dataset = TRIGGER_DATASET(batch_size=hp.batch_size, max_sequence_length=hp.max_sequence_length,
                              windows=hp.windows, dtype='IDENTIFICATION')

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
                'input_x': [],
                'input_c_pos': [],
                'dropout_keep_prob': 1.0,
            }

            pred = sess.run(predictions, feed_dict)
            print('pred :', pred)

