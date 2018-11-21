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

    checkpoint_dir = './runs/1542825629/checkpoints'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    with tf.Graph().as_default():
        sess = tf.Session()
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        with sess.as_default():
            model = Model(sentence_length=hp.max_sequence_length,
                          num_labels=len(dataset.all_labels),
                          vocab_size=len(dataset.all_words),
                          word_embedding_size=hp.word_embedding_size,
                          pos_embedding_size=hp.pos_embedding_size,
                          filter_sizes=hp.filter_sizes,
                          pos_tag_max_size=len(dataset.all_pos_taggings),
                          filter_num=hp.filter_num,
                          batch_size=hp.batch_size,
                          embed_matrx=dataset.word_embed)

            optimizer = tf.train.AdamOptimizer(hp.lr)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars)

            def trigger_eval_step(input_x, input_y, input_c_pos, input_pos_tag, dropout_keep_prob):
                feed_dict = {
                    model.input_x: input_x,
                    model.input_y: input_y,
                    model.input_c_pos: input_c_pos,
                    # model.input_pos_tag: input_pos_tag,
                    model.dropout_keep_prob: dropout_keep_prob,
                }
                accuracy, predicts = sess.run([model.accuracy, model.predicts], feed_dict)
                print("eval accuracy:{}".format(accuracy))

                print(classification_report([np.argmax(item) for item in input_y], predicts, target_names=dataset.all_labels))
                average_policy = 'macro'
                pre, rec, acc = precision_score([np.argmax(item) for item in input_y], predicts,
                                                average=average_policy), recall_score([np.argmax(item) for item in input_y],
                                                                                      predicts, average=average_policy), \
                                accuracy_score([np.argmax(item) for item in input_y], predicts)
                print("[{}]\nPrecision: {}\nRecall: {}\nAccuracy  :  {}\n".format(average_policy, pre, rec, acc))
                return predicts

            print("----test results---------------------------------------------------------------------")
            x, c, y, pos_c, pos_tag = dataset.next_eval_data()
            predicts = trigger_eval_step(input_x=x, input_y=y, input_c=c, input_c_pos=pos_c, input_pos_tag=pos_tag, dropout_keep_prob=1.0)

