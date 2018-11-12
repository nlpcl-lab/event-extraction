# Event Extraction

Tensorflow Implementation of Deep Learning Approach for  Event Extraction([**ACE 2005**](https://catalog.ldc.upenn.edu/LDC2006T06)) via Dynamic Multi-Pooling Convolutional Neural Networks.

## Usage

### Train

"[GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/)" is used as pre-trained word2vec model.

## Todo 

- Add word2vec

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

### Trigger Classification performance

```
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

* **Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks** (IJCNLP 2015), Chen, Yubo, et al. [[paper]](https://pdfs.semanticscholar.org/ca70/480f908ec60438e91a914c1075b9954e7834.pdf)
* zhangluoyang's cnn-for-auto-event-extract repository [[github]](https://github.com/zhangluoyang/cnn-for-auto-event-extract)



