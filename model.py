from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity

import gensim.downloader as api
import pandas as pd
import numpy as np
import re

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')


model = api.load("glove-wiki-gigaword-50")
model_dimensions = 50
sentence_max_length = 64


def clean_document(paragraph):
    stop_words = stopwords.words('english')

    def clean(sentence):
        clean_sentence = ""
        words = nltk.word_tokenize(sentence)
        for word in words:
            sub_word = re.sub(r'[^a-zA-Z ]', '', word).lower()
            if sub_word not in stop_words:
                clean_sentence += ' '+sub_word

        return clean_sentence.strip()

    clean_doc = [clean(sentence) for sentence in paragraph]

    return clean_doc



def get_embedding_matrix(vocab_size, tokenizer):
    embedding_matrix = np.zeros((vocab_size, model_dimensions))

    for word, i in tokenizer.word_index.items():
        if word in model:
            embedding_matrix[i] = model[word]

    return embedding_matrix


def get_tfidf_vectors(document):
    tfidfvectoriser = TfidfVectorizer(max_features=model_dimensions)
    for para in document:
        tfidfvectoriser.fit(para)

    tfidf_vectors = [tfidfvectoriser.transform(
        para).toarray() for para in document]
    return tfidf_vectors, tfidfvectoriser


def get_weighted_embeddings(padded_doc, tokenizer, embedding_matrix, cleaned_doc):

    tfidf_vectors, tfidf_vectorizer = get_tfidf_vectors(cleaned_doc)
    words_doc = tfidf_vectorizer.get_feature_names_out()

    def get_para_embeddings(padded_para, tfidf):

        para_embeddings = np.zeros((len(padded_para), model_dimensions))

        for i in range(len(padded_para)):
            for j in range(len(words_doc)):
                word = words_doc[j]
                token = tokenizer.word_index[word]
                embedding = embedding_matrix[token]
                tfidf_vector = tfidf[i][j]
                para_embeddings[i] += embedding*tfidf_vector

        return para_embeddings

    embeddings = []
    for i in range(len(padded_doc)):
        doc = padded_doc[i]
        tfidf = tfidf_vectors[i]
        embeddings.append(get_para_embeddings(doc, tfidf))

    return embeddings


def compare_doc(doc1, doc2):

    para_doc1 = list(filter(lambda x: x != '', doc1.split('\n')))
    para_doc1 = [para.strip() for para in para_doc1]

    para_doc2 = list(filter(lambda x: x != '', doc2.split('\n')))
    para_doc2 = [para.strip() for para in para_doc2]

    print("Para doc : ", para_doc1)

    # break the paragraphs to sentences
    sent_doc1 = [nltk.sent_tokenize(para) for para in para_doc1]
    sent_doc2 = [nltk.sent_tokenize(para) for para in para_doc2]

    print("List sentence in para : ", sent_doc1)

    cleaned_doc1 = [clean_document(para) for para in sent_doc1]
    cleaned_doc2 = [clean_document(para) for para in sent_doc2]

    print("Cleaned doc : ", cleaned_doc1)

    # 1 create tokenizer for full corpus
    tokenizer = Tokenizer()
    for para in cleaned_doc1:
        tokenizer.fit_on_texts(para)
    for para in cleaned_doc2:
        tokenizer.fit_on_texts(para)

    # 2 converted text to indexes
    tokenized_doc1 = [tokenizer.texts_to_sequences(
        para) for para in cleaned_doc1]
    tokenized_doc2 = [tokenizer.texts_to_sequences(
        para) for para in cleaned_doc2]

    print("Tokenized doc : ", tokenized_doc1)

    # 3 pad the sentences
    padded_doc1 = [pad_sequences(
        para, maxlen=sentence_max_length, padding='post') for para in tokenized_doc1]
    padded_doc2 = [pad_sequences(para, maxlen=sentence_max_length, padding='post')
                   for para in tokenized_doc2]

    print("Padded doc : ", padded_doc1)

    vocab_size = len(tokenizer.word_index)+1
    embedding_matrix = get_embedding_matrix(
        vocab_size, tokenizer)

    # 4 create tfidf for each document separately
    # reverse
    # changed cleaned doc to padded doc

    # document embeddings
    doc1_embeddings = get_weighted_embeddings(
        padded_doc1, tokenizer, embedding_matrix, cleaned_doc1)
    doc2_embeddings = get_weighted_embeddings(
        padded_doc2, tokenizer, embedding_matrix, cleaned_doc2)

    # print("Doc embeddings : ", doc1_embeddings)

    report = {}
    report["doc_a"] = {}
    # report["doc_b"] = {}
    report["doc_a"]["num_para"] = len(sent_doc1)
    # report["doc_b"]["num_para"] = len(sent_doc2)
    report["doc_a"]["para"] = []

    for i, pd1 in enumerate(sent_doc1):
        report["doc_a"]["para"].append({"num_sent": len(pd1)})
        # report["doc_a"]["para"][i] = {}
        report["doc_a"]["para"][i]["sent"] = []

        for j, sd1 in enumerate(pd1):
            temp = []
            most_similar = -1
            print("Sent : ", sd1)
            for x, pd2 in enumerate(sent_doc2):
                for y, sd2 in enumerate(pd2):
                    e1 = doc1_embeddings[i][j]
                    e2 = doc2_embeddings[x][y]
                    similarity = cosine_similarity([e1], [e2])[0][0]
                    # print("check", i, x, j, y, similarity)
                    if(similarity > most_similar):
                        temp = {"para_b": x+1, "sent_b": y +
                                1, "simi": similarity}
                        most_similar = similarity

            report["doc_a"]["para"][i]["sent"].append(temp)

    for i in range(report["doc_a"]["num_para"]):
        count = 0
        num_sent = report["doc_a"]["para"][i]["num_sent"]
        for j in range(num_sent):
            if(report["doc_a"]["para"][i]["sent"][j]["simi"] > 0.6):
                count = count+1
        if(num_sent == 0):
          report["doc_a"]["para"]["simi"] = 0
        else:
          print(type(num_sent))
          report["doc_a"]["para"][i]["simi"] = count/num_sent

    return report


