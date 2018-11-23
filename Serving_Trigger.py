import datetime, os, sys, json
import tensorflow as tf
from Dataset_Trigger import Dataset_Trigger as TRIGGER_DATASET
from Config import HyperParams_Tri_classification as hp
import nltk

from flask import Flask, session, g, request, render_template, redirect, Response, jsonify

app = Flask(__name__)

def get_batch(sentence, word_id, max_sequence_length):
    tokens = [word for word in nltk.word_tokenize(sentence)]
    words = []
    for i in range(max_sequence_length):
        if i < len(tokens):
            words.append(tokens[i])
        else:
            words.append('<eos>')

    word_ids = []
    for word in words:
        if word in word_id:
            word_ids.append(word_id[word])
        else:
            word_ids.append(word_id['<unk>'])

    # print('word_ids :', word_ids)
    size = len(word_ids)

    x_batch = []
    x_pos_batch = []
    for i in range(size):
        x_batch.append(word_ids)
        x_pos_batch.append([j - i for j in range(size)])

    return x_batch, x_pos_batch, tokens


dataset = TRIGGER_DATASET(batch_size=hp.batch_size, max_sequence_length=hp.max_sequence_length,
                          windows=hp.windows, dtype='IDENTIFICATION')

checkpoint_dir = './runs/1542973204/checkpoints'
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        print('restore model from {}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        @app.route('/api/event-extraction/trigger/identification', methods=['POST'])
        def serving():
            data = request.get_json()
            sentence = data['sentence']

            x_batch, x_pos_batch, tokens = get_batch(sentence=sentence, word_id=dataset.word_id, max_sequence_length=hp.max_sequence_length)

            print('x_batch :', x_batch)
            print('x_pos_batch :', x_pos_batch)

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

            preds = sess.run(predictions, feed_dict)
            print('id2label : ', dataset.id2label)
            result = ''
            for i in range(len(preds)):
                word = dataset.id2word[x_batch[0][i]]
                if word == '<unk>': word = tokens[i]
                if word == '<eos>': break
                print('word: {}, pred: {}'.format(word, str(preds[i])))
                result += '{}/{} '.format(word, dataset.id2label[preds[i]])

            return Response(json.dumps({'result': result}), status=200, mimetype='application/json')

base_dir = os.path.abspath(os.path.dirname(__file__) + '/')
sys.path.append(base_dir)
FLASK_DEBUG = os.getenv('FLASK_DEBUG', True)
app.run(host='0.0.0.0', debug=FLASK_DEBUG, port=8085)
