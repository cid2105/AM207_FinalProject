import os.path
import numpy as np
import codecs
import nltk
import string
import cPickle as pickle
from nltk.corpus import stopwords
from nltk.stem.porter import *
import codecs
from pre_process import PreProcess


def create_count_matrices(files, save_files=True):
    all_docs, train_docs, test_docs = build_train_test_set(files)
    preprocess = PreProcess()
    preprocess.build_vocab(all_docs)
    preprocessed_train_docs = preprocess.preprocess_docs(train_docs)
    preprocessed_test_docs = preprocess.preprocess_docs(test_docs)
    train_count_mat = preprocess.get_count_matrix(preprocessed_train_docs)
    test_count_mat  = preprocess.get_count_matrix(preprocessed_test_docs)
    if save_files:
        np.save("temp_data/train_count_mat.npy", train_count_mat)
        np.save("temp_data/test_count_mat.npy", test_count_mat)
    return train_count_mat, test_count_mat
        
def build_train_test_set(files):
    train_docs = []
    test_docs = []
    all_docs = []
    for book in books:
        with codecs.open('data/%s'%(book), 'r', encoding='utf-8') as f:
            lines = f.read().splitlines() 
            train_docs.append(" ".join(lines[0:len(lines)/2]))
            test_docs.append(" ".join(lines[0:len(lines)/2]))
            all_docs.append(" ".join(lines))

if __name__ == "__main__":
    books = ["beowulf.txt", "divine_comedy.txt", "dracula.txt", "frankenstein.txt", "huck_finn.txt", "moby_dick.txt", "sherlock_holmes.txt", "tale_of_two_cities.txt", "the_republic.txt", "ulysses.txt"]
    if os.path.isfile("temp_data/train_count_mat.npy") and os.path.isfile("temp_data/test_count_mat.npy"):
        train_count_mat = np.load("temp_data/train_count_mat.npy")
        test_count_mat  = np.load("temp_data/test_count_mat.npy")
    else:
        print "Pre-processing..."
        files = ["data/" + book for book in books]
        train_count_mat, test_count_mat = create_count_matrices(files, save_files=True)
        print "Built training and test count matrices"
    