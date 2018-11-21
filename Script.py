import datetime, os, time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
from Util import train_parser
from Dataset import Dataset as ARGUMENT_DATASET
from Dataset_Trigger import Dataset_Trigger as TRIGGER_DATASET
from Config import HyperParams_Tri_classification as hp_trigger, HyperParams as hp_argument

import Visualize

if __name__ == '__main__':
    task, subtask = train_parser()
    subtask_type = 'IDENTIFICATION' if subtask == 1 else 'CLASSIFICATION'
    hp, dataset, Model = [None for _ in range(3)]

    if task == 1:
        hp = hp_trigger
        dataset = TRIGGER_DATASET(batch_size=hp.batch_size, max_sequence_length=hp.max_sequence_length,
                                  windows=hp.windows, dtype=subtask_type)
        for label in dataset.all_labels:
            print(label + ' ' + str(dataset.label_id[label]))

        from Model_Trigger import Model

        print("\n\nTrigger {} start.\n\n".format(subtask_type))
    if task == 2:
        hp = hp_argument
        dataset = ARGUMENT_DATASET(batch_size=hp.batch_size, max_sequence_length=hp.max_sequence_length,
                                   windows=hp.windows, dtype=subtask_type)
        from Model import Model

        print("\n\nArgument {} start.\n\n".format(subtask_type))

    with tf.Graph().as_default():
        sess = tf.Session()
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

            # TODO: after train, do save
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
            sess.run(tf.initialize_all_variables())

            def trigger_train_step(input_x, input_y, input_c, input_c_pos, input_pos_tag, dropout_keep_prob):
                feed_dict = {
                    model.input_x: input_x,
                    model.input_y: input_y,
                    model.input_c_pos: input_c_pos,
                    model.input_pos_tag: input_pos_tag,
                    model.dropout_keep_prob: dropout_keep_prob,
                }
                _, loss, accuracy = sess.run([train_op, model.loss, model.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                # print("{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))


            def trigger_eval_step(input_x, input_y, input_c, input_c_pos, input_pos_tag, dropout_keep_prob):
                feed_dict = {
                    model.input_x: input_x,
                    model.input_y: input_y,
                    model.input_c_pos: input_c_pos,
                    model.input_pos_tag: input_pos_tag,
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

                Visualize.draw(
                    epoch=epoch,
                    input_x=input_x,
                    input_y=[np.argmax(item) for item in input_y],
                    predicts=predicts,
                    input_c_pos=input_c_pos,
                    id2label = dataset.id2label,
                    id2word=dataset.id2word,
                )
                return predicts


            def argument_train_step(input_x, input_y, input_t, input_c, input_t_pos, input_c_pos, dropout_keep_prob):
                feed_dict = {
                    model.input_x: input_x,
                    model.input_y: input_y,
                    # model.input_t:input_t,
                    # model.input_c:input_c,
                    model.input_t_pos: input_t_pos,
                    model.input_c_pos: input_c_pos,
                    model.dropout_keep_prob: dropout_keep_prob,
                }
                _, loss, accuracy = sess.run([train_op, model.loss, model.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                # print("{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))


            def argument_eval_step(input_x, input_y, input_t, input_c, input_t_pos, input_c_pos, dropout_keep_prob):
                feed_dict = {
                    model.input_x: input_x,
                    model.input_y: input_y,
                    # model.input_t:input_t,
                    # model.input_c:input_c,
                    model.input_t_pos: input_t_pos,
                    model.input_c_pos: input_c_pos,
                    model.dropout_keep_prob: dropout_keep_prob,
                }
                accuracy, predicts = sess.run([model.accuracy, model.predicts], feed_dict)
                from sklearn.metrics import classification_report
                print("eval accuracy:{}".format(accuracy))
                # print("input_y : ", [np.argmax(item) for item in input_y], ', predicts :', predicts)
                print(classification_report([np.argmax(item) for item in input_y], predicts,
                                            target_names=dataset.all_labels))
                return predicts


            print("TRAIN START")
            for epoch in range(hp.num_epochs):
                print('epoch: {}/{}'.format(epoch + 1, hp.num_epochs))
                for j in range(len(dataset.train_instances) // hp.batch_size):
                    if task == 1:
                        x, c, y, pos_c, pos_tag = dataset.next_train_data()
                        trigger_train_step(input_x=x, input_y=y, input_c=c, input_c_pos=pos_c, input_pos_tag=pos_tag,
                                           dropout_keep_prob=0.5)

                        saver.save(sess, checkpoint_prefix + "-trigger-identification")
                    if task == 2:
                        x, t, c, y, pos_c, pos_t, _ = dataset.next_train_data()
                        argument_train_step(input_x=x, input_y=y, input_t=t, input_c=c, input_c_pos=pos_c,
                                            input_t_pos=pos_t,
                                            dropout_keep_prob=0.5)

                if epoch % 5 == 0:
                    if task == 1:
                        x, c, y, pos_c, pos_tag = dataset.next_eval_data()
                        trigger_eval_step(input_x=x, input_y=y, input_c=c, input_c_pos=pos_c, input_pos_tag=pos_tag,
                                          dropout_keep_prob=1.0)
                    if task == 2:
                        x, t, c, y, pos_c, pos_t, _ = dataset.eval_data()
                        argument_eval_step(input_x=x, input_y=y, input_t=t, input_c=c, input_c_pos=pos_c,
                                           input_t_pos=pos_t,
                                           dropout_keep_prob=1.0)

            print("----test results---------------------------------------------------------------------")
            if task == 1:
                x, c, y, pos_c, pos_tag = dataset.next_eval_data()
                predicts = trigger_eval_step(input_x=x, input_y=y, input_c=c, input_c_pos=pos_c, input_pos_tag=pos_tag, dropout_keep_prob=1.0)
            if task == 2:
                x, t, c, y, pos_c, pos_t, _ = dataset.eval_data()
                predicts = argument_eval_step(input_x=x, input_y=y, input_t=t, input_c=c, input_c_pos=pos_c,
                                              input_t_pos=pos_t,
                                              dropout_keep_prob=1.0)
