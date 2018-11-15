from bs4 import BeautifulSoup

def draw(epoch, input_x, input_y, predicts, input_c_pos, id2label, id2word):
    sents_visual_file = './visualization/{}.html'.format(epoch)

    size = len(input_y)
    with open(sents_visual_file, "w") as html_file:
        html_file.write('<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"/></head>')

        for i in range(size):
            if input_y[i] == predicts[i]: continue

            html_file.write('<div style="padding: 20px 0;">')
            current_pos = 0
            for j in range(size):
                if input_c_pos[j] == 0:
                    current_pos = j
                    break

            sent = ''
            for j in range(size):
                if j == current_pos:
                    sent += '<span style="background: rgba(255, 0, 0, 0.4);>{}</span> '.format(id2word[input_x[j]])
                else:
                    sent += id2word[input_x[j]] + ' '

            html_file.write(sent)
            html_file.write('<div>Prediction: {}</div>'.format(id2label(predicts[j])))
            html_file.write('<div>Answer: {}</div>'.format(id2label(input_y[j])))

            html_file.write('</div>')

        html_file.write('</html>')


if __name__ == '__main__':
    draw()
