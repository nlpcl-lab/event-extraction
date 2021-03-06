# Event Extraction

Tensorflow Implementation of Deep Learning Approach for  Event Extraction([**ACE 2005**](https://catalog.ldc.upenn.edu/LDC2006T06)) via Dynamic Multi-Pooling Convolutional Neural Networks.

# Requirements

* Tensorflow
* Scikit-learn
* NLTK

`pip install -r requirements.txt` may help.

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

- Apply Dymamic Multi-Pooling CNN
- Evaluation Script

## Results

### Trigger identification performance
```
              precision    recall  f1-score   support

     TRIGGER       0.59      0.44      0.50       527
        None       0.97      0.98      0.98      9151

   micro avg       0.95      0.95      0.95      9678
   macro avg       0.78      0.71      0.74      9678
weighted avg       0.95      0.95      0.95      9678

```

### Trigger classification performance

```
             precision    recall  f1-score   support

        Life       0.75      0.70      0.72       114
     Justice       0.78      0.85      0.81       114
    Movement       0.69      0.70      0.69        53
   Personnel       0.68      0.64      0.66        78
    Business       0.75      0.46      0.57        13
    Conflict       0.78      0.83      0.80       247
     Contact       0.79      0.86      0.83        36
 Transaction       1.00      0.48      0.65        27

   micro avg       0.76      0.76      0.76       682
   macro avg       0.78      0.69      0.72       682
weighted avg       0.76      0.76      0.76       682
```

##### with None label
```
              precision    recall  f1-score   support

    Movement       0.47      0.21      0.29        68
    Business       1.00      0.10      0.18        10
     Contact       0.67      0.22      0.33        37
     Justice       0.32      0.17      0.23        63
        None       0.96      0.99      0.98      8348
    Conflict       0.70      0.38      0.50       156
        Life       0.64      0.38      0.48        65
 Transaction       0.75      0.10      0.18        29
   Personnel       0.73      0.24      0.36        46

   micro avg       0.95      0.95      0.95      8822
   macro avg       0.69      0.31      0.39      8822
weighted avg       0.94      0.95      0.94      8822
```

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

## References

* **Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks** (IJCNLP 2015), Chen, Yubo, et al. [[paper]](https://pdfs.semanticscholar.org/ca70/480f908ec60438e91a914c1075b9954e7834.pdf)
* zhangluoyang's cnn-for-auto-event-extract repository [[github]](https://github.com/zhangluoyang/cnn-for-auto-event-extract)
