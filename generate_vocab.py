'''
@author: wanzeyu

@contact: wan.zeyu@outlook.com

@file: generate_vocab.py

@time: 2017/12/12 9:31
'''
import nltk


def get_vocab():
    words = []
    with open("train_source.txt", "r", encoding="utf8") as fp:
        for line in fp:
            res = line.strip().split(" ")
            for i in res:
                words.append(i)
    with open("train_target.txt", "r", encoding="utf8") as fp:
        for line in fp:
            res = line.strip().split(" ")
            for i in res:
                words.append(i)
    freq = nltk.FreqDist(words)
    with open("train_vocab.txt", "w", encoding="utf8") as fp:
        for word in freq.most_common(10052):
            fp.write(word[0])
            fp.write("\n")


if __name__ == "__main__":
    get_vocab()
