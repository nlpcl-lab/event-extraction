# Event Extraction

Tensorflow Implementation of Deep Learning Approach for Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks.

## Data

[ACE 2005](https://catalog.ldc.upenn.edu/LDC2006T06) 

## Usage

### Train

"[GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/)" is used as pre-trained word2vec model.

## TODO 

- Increase the number of data used for evaluation (currently only 30)
- Add word2vec
- Move hyperparameter into `Config.py`
- Trigger Classification performance
```bash
2018-11-11T18:18:11.830461: loss 0.100064, acc 0.966667
2018-11-11T18:18:11.872351: loss 0.0342995, acc 0.966667
2018-11-11T18:18:11.914238: loss 0.000669132, acc 1
2018-11-11T18:18:11.957123: loss 0.000533539, acc 1
2018-11-11T18:18:11.999011: loss 0.390004, acc 0.933333
2018-11-11T18:18:12.041897: loss 0.145098, acc 0.966667
2018-11-11T18:18:12.085778: loss 0.251621, acc 0.933333
2018-11-11T18:18:12.127666: loss 0.280516, acc 0.933333
2018-11-11T18:18:12.170552: loss 0.0573161, acc 0.966667
2018-11-11T18:18:12.211442: loss 0.00108241, acc 1
2018-11-11T18:18:12.254328: loss 0.18019, acc 0.966667
2018-11-11T18:18:12.298210: loss 0.565784, acc 0.933333
2018-11-11T18:18:12.339103: loss 0.00310437, acc 1
2018-11-11T18:18:12.381988: loss 0.357736, acc 0.966667
2018-11-11T18:18:12.423874: loss 0.158542, acc 0.966667
2018-11-11T18:18:12.466759: loss 0.250013, acc 0.966667
2018-11-11T18:18:12.509646: loss 4.06054e-05, acc 1
2018-11-11T18:18:12.553527: loss 0.00259135, acc 1
2018-11-11T18:18:12.594419: loss 0.206243, acc 0.966667
2018-11-11T18:18:12.638300: loss 0.293289, acc 0.933333
2018-11-11T18:18:12.681185: loss 0.124193, acc 0.966667
2018-11-11T18:18:12.724071: loss 0.0422064, acc 0.966667
2018-11-11T18:18:12.765960: loss 0.00164163, acc 1
2018-11-11T18:18:12.807848: loss 0.106209, acc 0.966667
----test results---------------------------------------------------------------------
eval accuracy:0.9333333373069763
input_y :  [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 2, 20, 20] , predicts : [20 20 20 20 20 20 20 20 20 20 20 20 20 20 20  6 20 20 20 20 20 20 20 20
 20 20 20 20 20 20]
              precision    recall  f1-score   support

           2       0.00      0.00      0.00         1
           6       0.00      0.00      0.00         0
          20       0.97      0.97      0.97        29

   micro avg       0.93      0.93      0.93        30
   macro avg       0.32      0.32      0.32        30
weighted avg       0.93      0.93      0.93        30
```
- Argument classification performance
```bash
2018-11-10T03:38:16.408551: loss 0.00862374, acc 1
2018-11-10T03:38:16.484978: loss 0.427776, acc 0.933333
2018-11-10T03:38:16.556502: loss 0.552501, acc 0.9
2018-11-10T03:38:16.630882: loss 0.290723, acc 0.9
2018-11-10T03:38:16.699259: loss 0.120635, acc 0.966667
2018-11-10T03:38:16.770981: loss 0.172747, acc 0.966667
2018-11-10T03:38:16.845399: loss 0.0313517, acc 0.966667
2018-11-10T03:38:16.912906: loss 0.238546, acc 0.966667
2018-11-10T03:38:16.984497: loss 0.0615598, acc 0.966667
2018-11-10T03:38:17.052549: loss 0.120855, acc 0.966667
2018-11-10T03:38:17.120559: loss 0.213745, acc 0.9
2018-11-10T03:38:17.196295: loss 0.0645926, acc 0.966667
2018-11-10T03:38:17.268672: loss 0.0221864, acc 1
2018-11-10T03:38:17.341155: loss 0.0375219, acc 1
2018-11-10T03:38:17.411111: loss 0.0501572, acc 0.966667
2018-11-10T03:38:17.484186: loss 0.986013, acc 0.9
2018-11-10T03:38:17.554882: loss 1.06596, acc 0.9
2018-11-10T03:38:17.626529: loss 0.120261, acc 0.933333
2018-11-10T03:38:17.699312: loss 1.08561, acc 0.9
2018-11-10T03:38:17.768334: loss 0.0488479, acc 1
2018-11-10T03:38:17.844030: loss 0.00877127, acc 1
2018-11-10T03:38:17.913705: loss 0.0676153, acc 0.966667
2018-11-10T03:38:17.981607: loss 0.652455, acc 0.9
2018-11-10T03:38:18.054250: loss 0.156669, acc 0.966667
2018-11-10T03:38:18.125788: loss 0.288849, acc 0.933333
2018-11-10T03:38:18.199569: loss 0.133604, acc 0.966667
2018-11-10T03:38:18.276992: loss 0.151481, acc 0.966667
2018-11-10T03:38:18.352524: loss 0.439608, acc 0.933333
2018-11-10T03:38:18.423659: loss 0.178131, acc 0.9
2018-11-10T03:38:18.496560: loss 0.286932, acc 0.866667
2018-11-10T03:38:18.569519: loss 0.0904374, acc 0.933333
2018-11-10T03:38:18.642055: loss 0.0318502, acc 1
2018-11-10T03:38:18.712787: loss 0.164931, acc 0.933333
2018-11-10T03:38:18.784207: loss 0.393558, acc 0.9
2018-11-10T03:38:18.854141: loss 0.625278, acc 0.9
2018-11-10T03:38:18.925009: loss 1.23597, acc 0.833333
2018-11-10T03:38:18.996662: loss 0.19571, acc 0.933333
2018-11-10T03:38:19.068105: loss 0.0390664, acc 1
2018-11-10T03:38:19.142528: loss 0.054021, acc 0.966667
2018-11-10T03:38:19.212596: loss 0.137261, acc 0.933333
2018-11-10T03:38:19.285346: loss 0.27786, acc 0.9
----test results---------------------------------------------------------------------
eval accuracy:0.7666666507720947
input_y :  [7, 7, 3, 7, 12, 7, 7, 7, 7, 7, 7, 7, 8, 33, 7, 7, 18, 29, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7]
predicts : [ 7  7  7  7 12  7  7  7  7  7  7  7 18 24  7  7  8  7  7  5  7  7  7  7 7  7  7  7  7  7]
              precision    recall  f1-score   support

           3       0.00      0.00      0.00         1
           5       0.00      0.00      0.00         1
           7       0.88      0.96      0.92        23
           8       0.00      0.00      0.00         1
          12       1.00      1.00      1.00         1
          18       0.00      0.00      0.00         1
          24       0.00      0.00      0.00         0
          29       0.00      0.00      0.00         1
          33       0.00      0.00      0.00         1

   micro avg       0.77      0.77      0.77        30
   macro avg       0.21      0.22      0.21        30
weighted avg       0.71      0.77      0.74        30
```

## Reference

* **Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks** (IJCNLP 2015), Chen, Yubo, et al. [[paper]](http://www.aclweb.org/anthology/C14-1220)
* zhangluoyang's cnn-for-auto-event-extract repository [[github]](https://github.com/zhangluoyang/cnn-for-auto-event-extract)



