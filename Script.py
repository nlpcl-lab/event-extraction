import datetime
import tensorflow as tf
from Util import train_parser
from Dataset import Dataset as ARGUMENT_DATASET
from Dataset_Trigger import Dataset_Trigger as TRIGGER_DATASET
from Config import HyperParams_Tri_classification as hp_trigger, HyperParams as hp_argument

if __name__=='__main__':
    task, subtask = train_parser()
    subtask_type = 'IDENTIFICATION' if subtask==1 else 'CLASSIFICATION'

    if task==1:
        hp = hp_trigger
        dataset = TRIGGER_DATASET(batch_size=hp.batch_size, max_sequence_length=hp.max_sequence_length,
                                  windows=hp.windows, dtype=subtask_type)
        from Model_Trigger import Model
    if task==2:
        hp = hp_argument
        dataset = ARGUMENT_DATASET(batch_size=hp.batch_size, max_sequence_length=hp.max_sequence_length,
                                  windows=hp.windows, dtype=subtask_type)
        from Model import Model

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            model = Model(sentence_length=hp.max_sequence_length,
                          num_labels=len(dataset.all_labels),
                          vocab_size=len(dataset.all_words),
                          word_embedding_size=hp.word_embedding_size,
                          pos_embedding_size=hp.pos_embedding_size,
                          filter_sizes=hp.filter_sizes,
                          filter_num=hp.filter_num,
                          batch_size=hp.batch_size)

            optimizer = tf.train.AdamOptimizer(hp.lr)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars)

            # TODO: after train, do save
            # timestamp = str(int(time.time()))
            # out_dir = os.path.abspath(os.path.join(os.path.curdir, "model_01", timestamp))
            # print("Writing to {}\n".format(out_dir))
            # checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            # if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
            # saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
            sess.run(tf.initialize_all_variables())


            def train_step(input_x, input_y, input_c, input_c_pos, dropout_keep_prob):
                feed_dict = {
                    model.input_x: input_x,
                    model.input_y: input_y,
                    model.input_c_pos: input_c_pos,
                    model.dropout_keep_prob: dropout_keep_prob,
                }
                _, loss, accuracy = sess.run([train_op, model.loss, model.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))


            def eval_step(input_x, input_y, input_c, input_c_pos, dropout_keep_prob):
                feed_dict = {
                    model.input_x: input_x,
                    model.input_y: input_y,
                    model.input_c_pos: input_c_pos,
                    model.dropout_keep_prob: dropout_keep_prob,
                }
                accuracy, predicts = sess.run([model.accuracy, model.predicts], feed_dict)
                print("eval accuracy:{}".format(accuracy))
                print("input_y : ", [np.argmax(item) for item in input_y], ', predicts :', predicts)
                print(classification_report([np.argmax(item) for item in input_y], predicts))#, target_names=dataset.all_labels))
                print("Precision: {}\nRecall: {}\nAccuracy  :  {}".format(
                    precision_score([np.argmax(item) for item in input_y], predicts, average='weighted'),
                    recall_score([np.argmax(item) for item in input_y], predicts, average='weighted'),
                    accuracy_score([np.argmax(item) for item in input_y], predicts)))
                return predicts

            print("TRAIN START")
            for epoch in range(hp.num_epochs):
                print('epoch: {}/{}'.format(epoch + 1, hp.num_epochs))
                for j in range(len(dataset.train_instances) // hp.batch_size):
                    x, c, y, pos_c, _ = dataset.next_train_data()
                    train_step(input_x=x, input_y=y, input_c=c, input_c_pos=pos_c, dropout_keep_prob=0.8)
                if epoch % 3 == 0:
                    x, c, y, pos_c, _ = dataset.next_eval_data()
                    eval_step(input_x=x, input_y=y, input_c=c, input_c_pos=pos_c, dropout_keep_prob=1.0)

            print("----test results---------------------------------------------------------------------")
            x, c, y, pos_c, _ = dataset.next_eval_data()
            predicts = eval_step(input_x=x, input_y=y, input_c=c, input_c_pos=pos_c, dropout_keep_prob=1.0)

