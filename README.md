# Event Extraction

Tensorflow Implementation of Deep Learning Approach for  Event Extraction([**ACE 2005**](https://catalog.ldc.upenn.edu/LDC2006T06)) via Dynamic Multi-Pooling Convolutional Neural Networks.

## Usage

### Train

* "[GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/)" is used as a pre-trained word2vec model.

* "[glove.6B](https://nlp.stanford.edu/projects/glove/)" is used as a pre-trained GloVe model.

* Performance (accuracy and f1-socre) outputs during training are **UNOFFICIAL SCORES** of *ACE 2005*. 

##### Train Example:
```bash
$ python Script.py {taskID} {subtaskID}
```
* `taskID`: 1 for Trigger, 2 for Argument

* `subtaskID`: 1 for Identification, 2 for Classification

* After model training, evaluation results will be shown.

```bash
$ python Script.py 1 2  # Script for `Trigger Classification`
```

## Todo 

- Evaluation Script

## Results

### Argument classification performance
```
              precision    recall  f1-score   support

      Seller       0.00      0.00      0.00         4
       Money       0.00      0.00      0.00        13
      Target       0.30      0.16      0.21        67
 Destination       0.45      0.20      0.28        49
      Victim       0.35      0.23      0.28        48
  Instrument       0.25      0.10      0.14        31
       Crime       0.67      0.14      0.23        43
 Adjudicator       0.00      0.00      0.00        20
      Origin       0.00      0.00      0.00        34
        Time       0.46      0.22      0.30       193
       Agent       0.00      0.00      0.00        40
    Position       0.00      0.00      0.00        20
       Giver       0.00      0.00      0.00        16
 Beneficiary       0.00      0.00      0.00         5
         Org       0.00      0.00      0.00         6
    Artifact       0.00      0.00      0.00        14
       Place       0.28      0.21      0.24       149
        None       0.75      0.95      0.84      2593
  Prosecutor       0.00      0.00      0.00         2
      Person       0.25      0.18      0.21       113
    Attacker       0.32      0.11      0.16        75
   Defendant       0.47      0.14      0.21        51
    Sentence       0.67      0.40      0.50        10
   Plaintiff       0.00      0.00      0.00        17
     Vehicle       0.00      0.00      0.00        10
      Entity       0.17      0.02      0.03       110
   Recipient       0.00      0.00      0.00         4
       Price       0.00      0.00      0.00         1
       Buyer       0.00      0.00      0.00         4

   micro avg       0.70      0.70      0.70      3742
   macro avg       0.19      0.11      0.13      3742
weighted avg       0.61      0.70      0.64      3742
```

### Trigger classification performance

```
             precision    recall  f1-score   support

           1       0.40      0.36      0.38        11
           2       0.00      0.00      0.00         7
           3       0.61      0.40      0.48        35
           4       1.00      0.17      0.29         6
           5       0.00      0.00      0.00         1
           6       0.33      0.08      0.13        12
           7       0.50      0.13      0.21        15
           8       0.00      0.00      0.00         4
           9       0.29      0.41      0.34        29
          11       0.00      0.00      0.00         6
          12       0.00      0.00      0.00        10
          13       0.75      0.30      0.43        10
          14       0.43      0.18      0.25        73
          15       0.18      0.10      0.13        20
          16       0.00      0.00      0.00         5
          17       0.50      0.21      0.30        14
          18       0.00      0.00      0.00         5
          19       0.67      0.11      0.18        19
          20       0.45      0.38      0.42        13
          21       0.00      0.00      0.00         1
          23       0.48      0.40      0.44       181
          24       0.00      0.00      0.00         4
          25       0.00      0.00      0.00        10
          27       0.25      0.04      0.06        27
          28       0.00      0.00      0.00         1
          29       0.00      0.00      0.00         3
          30       0.00      0.00      0.00         2
          31       0.00      0.00      0.00         1
          32       0.97      0.99      0.98     12966
          33       0.00      0.00      0.00        13

   micro avg       0.96      0.96      0.96     13504
   macro avg       0.26      0.14      0.17     13504
weighted avg       0.95      0.96      0.95     13504
```

## References

* **Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks** (IJCNLP 2015), Chen, Yubo, et al. [[paper]](https://pdfs.semanticscholar.org/ca70/480f908ec60438e91a914c1075b9954e7834.pdf)
* zhangluoyang's cnn-for-auto-event-extract repository [[github]](https://github.com/zhangluoyang/cnn-for-auto-event-extract)


## Contributors

[ChaeHun](http://nlp.kaist.ac.kr/~ddehun), [Seungwon](http://nlp.kaist.ac.kr/~swyoon)

