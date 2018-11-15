def draw(epoch, input_x, input_y, predicts, input_c_pos, id2label, id2word):
    sents_visual_file = './visualization/{}.html'.format(epoch)

    batch_size = len(input_y)
    with open(sents_visual_file, "w") as html_file:
        html_file.write('<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"/></head>')

        for i in range(batch_size):
            if input_y[i] == predicts[i]: continue

            sent_size = len(input_x[i])
            html_file.write('<div style="padding: 20px 0;">')
            current_pos = 0
            for j in range(sent_size):
                if input_c_pos[j] == 0:
                    current_pos = j
                    break

            sent = ''
            for j in range(sent_size):
                word = id2word[input_x[i][j]]
                if word == '<eos>': continue
                if j == current_pos:
                    sent += '<span style="background: rgba(255, 0, 0, 0.4);>{}</span> '.format(word)
                else:
                    sent += word + ' '

            html_file.write(sent)
            html_file.write('<div>Prediction: {}</div>'.format(id2label[predicts[j]]))
            html_file.write('<div>Answer: {}</div>'.format(id2label[input_y[j]]))

            html_file.write('</div>')

        html_file.write('</html>')


if __name__ == '__main__':
    draw()
