# Keras seq2seq word-level model implementation
# Overview
Keras implementation for seq2seq by wanzeyu

In this project I try to implement seq2seq word level model using keras.

Resource Used:

1. MSRP paraphrase corpus

Requirements:
1. Keras
2. Numpy


# The reason I open this repo
* The [official tutorial](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) 
about implementing seq2seq model is character level. 
But word level implementation is common.


# QuickStart
```
python basic_seq2seq.py
```

# Resource
* `test_source.txt` original sentence
* `test_target.txt` translation sentence
* `train_vocab.txt` vocabulary 

# Problems
1. The inference function has some problems

# Progress
- [ ] Write the evaluation code
- [ ] Add Attention function
- [ ] Using Bidirectional LSTM in encoder
- [ ] Implementing *Joint Copying and Restricted Generation for Paraphrase*

# References
- [A ten-minute introduction to sequence-to-sequence learning in Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)
- [Joint Copying and Restricted Generation for Paraphrase](https://arxiv.org/abs/1611.09235)