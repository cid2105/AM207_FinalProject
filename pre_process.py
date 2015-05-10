
import numpy as np
import codecs
import nltk
import string
import cPickle as pickle
from nltk.corpus import stopwords
from nltk.stem.porter import *


class PreProcess(object):
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.vocab_dict = {}
        self.vocab = np.empty(0)
        
    def stem_tokens(self, tokens):
        stemmed = []
        for item in tokens:
            stemmed.append(self.stemmer.stem(item))
        return stemmed

    def preprocess_doc(self, doc):
        doc = doc.lower()
        no_punctuation = re.sub(r'[^a-zA-Z\s]','',doc)
        tokens = nltk.word_tokenize(no_punctuation)
        filtered = [w for w in tokens if not w in stopwords.words('english')]
        stemmed = self.stem_tokens(filtered)
        return stemmed
    
    def preprocess_docs(self, docs):
        processed = np.array(map(self.preprocess_doc, docs))
        return processed 
    
    def freq_map(self, doc):
        out = np.zeros(self.vocab.size, dtype=np.int32)
        for w in doc:
            out[w] += 1
        return out

    '''
        Description: Prepopulates the vocabulary
        In: expects a list of docs
    '''
    def build_vocab(self, docs):  
        processed = self.preprocess_docs(docs)
        self.vocab = np.unique(np.hstack(processed.flat))
        for idx, w in enumerate(self.vocab):
            self.vocab_dict[w] = idx
        return self.vocab
        
    def doc_as_num(self, doc):
        return [self.vocab_dict[w] for w in doc]
    
    ''' 
        Input: processed documents
        Returns: count matrix
    '''
    def get_count_matrix(self, processed_docs):
        if self.vocab.size == 0:
            print "usage: please build vocab before calling get_count_matrix"
            return
        docs_as_nums = map(lambda doc: self.doc_as_num(doc), processed_docs)
        count_mat = np.array([ self.freq_map(doc) for doc in docs_as_nums], dtype=np.int32)
        return count_mat
        
if __name__ == "__main__":
    
    ## TEST PREPROCESS ##
    import codecs
    books = ["beowulf.txt", "divine_comedy.txt", "dracula.txt", "frankenstein.txt", "huck_finn.txt", "moby_dick.txt", "sherlock_holmes.txt", "tale_of_two_cities.txt", "the_republic.txt", "ulysses.txt"]
    train_docs = []
    test_docs = []
    all_docs = []
    for book in books:
        with codecs.open('data/%s'%(book), 'r', encoding='utf-8') as f:
            lines = f.read().splitlines() 
            train_docs.append(" ".join(lines[0:len(lines)/2]))
            test_docs.append(" ".join(lines[0:len(lines)/2]))
            all_docs.append(" ".join(lines))
    pre_process = PreProcess()
    pre_process.build_vocab(all_docs) # GLOBAL VOCAB
    processed_train_docs = pre_process.preprocess_docs(train_docs)
    processed_test_docs = pre_process.preprocess_docs(test_docs)
    train_count_mat = pre_process.get_count_matrix(processed_train_docs)
    test_count_mat = pre_process.get_count_matrix(processed_test_docs)
    np.save("temp_data/train_count_mat.npy", train_count_mat)
    np.save("temp_data/train_count_mat.npy", test_count_mat)
    print "Success"