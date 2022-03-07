from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity

import gensim.downloader as api
import pandas as pd
import numpy as np
import re


from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


model = api.load("glove-wiki-gigaword-50")


def clean_document(document):
    stop_words = stopwords.words('english')

    cleaned_document = document.apply(lambda x: " ".join(
        re.sub(r'[^a-zA-Z]', ' ', w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]', ' ', w).lower() not in stop_words_l))

    return cleaned_document


def compare_sentence(sentence1, sentence2):
    return 1


def get_embedding_matrix(vocab_size, model_dimensions, tokenizer):
    embedding_matrix = np.zeros((vocab_size, model_dimensions))

    for word, i in tokenizer.word_index.items():
        if word in model:
            embedding_matrix[i] = model[word]

    return embedding_matrix


def get_tfidf_vectors(model_dimensions, document):
    tfidfvectoriser = TfidfVectorizer(max_features=model_dimensions)
    tfidfvectoriser.fit(document)
    tfidf_vectors = tfidfvectoriser.transform(document).toarray()
    return tfidf_vectors


def get_weighted_embeddings(padded_doc, model_dimensions, words_doc, tokenizer, embedding_matrix, tfidf_vectors):
    doc_embeddings = np.zeros((len(padded_doc), model_dimensions))

    for i in range(len(padded_doc)):
        for j in range(len(words_doc)):
            word = words_doc[j]
            token = tokenizer.word_index[word]
            embedding = embedding_matrix[token]
            tfidf_vector = tfidf_vectors[i][j]
            doc_embeddings[i] += embedding*tfidf_vector

    return doc_embeddings



def compare_doc(doc1, doc2):

    doc1_df = pd.DataFrame(doc1, columns=['documents'])
    doc2_df = pd.DataFrame(doc2, columns=['documents'])

    # 0 clean both the documents
    cleaned_doc1_df = clean_document(doc1_df)
    cleaned_doc2_df = clean_document(doc2_df)

    # 1 create tokenizer for full corpus
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(cleaned_doc1_df)
    tokenizer.fit_on_texts(cleaned_doc2_df)

    # 2 converted text to indexes
    tokenized_doc1 = tokenizer.texts_to_sequences(cleaned_doc1_df)
    tokenized_doc2 = tokenizer.texts_to_sequences(cleaned_doc2_df)

    # 3 pad the sentences
    padded_doc1 = pad_sequences(tokenized_doc1, maxlen=64, padding='post')
    padded_doc2 = pad_sequences(tokenized_doc2, maxlen=64, padding='post')


    vocab_size = len(tokenizer.word_index)+1
    model_dimensions = 300
    embedding_matrix = get_embedding_matrix(vocab_size, model_dimensions, tokenizer)

    # 4 create tfidf for each document separately
    tfidf_vectors1 = get_tfidf_vectors(model_dimensions, cleaned_doc1_df)
    tfidf_vectors2 = get_tfidf_vectors(model_dimensions, cleaned_doc2_df)

   
    # document embeddings
    doc1_embeddings = get_weighted_embeddings(
        padded_doc1, model_dimensions, words_doc1, tokenizer, embedding_matrix, tfidf_vectors1)
    doc2_embeddings = get_weighted_embeddings(
        padded_doc2, model_dimensions, words_doc2, tokenizer, embedding_matrix, tfidf_vectors2)


    #docuement wise tifidf features 
    words_doc1 = tfidf_vectors1.get_feature_names_out()
    words_doc2 = tfidf_vectors2.get_feature_names_out()


    pairwise_similarities = cosine_similarity(doc1_embeddings, doc2_embeddings)

    return pairwise_similarities
