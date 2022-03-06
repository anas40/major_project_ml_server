from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity

import traceback
import logging

from flask import Flask, request, Response, abort

import gensim.downloader as api
import pandas as pd
import numpy as np
import re


from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


model = api.load("glove-wiki-gigaword-50")
model_dimensions = 300


app = Flask(__name__)


def compare_doc(doc1, doc2):
    documents = doc1+doc2
    documents_df = pd.DataFrame(documents, columns=['documents'])
    print(documents_df)

    # removing special characters and stop words from the text
    stop_words_l = stopwords.words('english')
    documents_df['documents_cleaned'] = documents_df.documents.apply(lambda x: " ".join(re.sub(
        r'[^a-zA-Z]', ' ', w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]', ' ', w).lower() not in stop_words_l))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(documents_df.documents_cleaned)

    # converted text to indexes
    tokenized_documents = tokenizer.texts_to_sequences(
        documents_df.documents_cleaned)
    tokenized_paded_documents = pad_sequences(
        tokenized_documents, maxlen=64, padding='post')
    vocab_size = len(tokenizer.word_index)+1

    # print("Vocab size : ", vocab_size)
     # six sentences, each of 64 length
    # print("Padded document shape : ", tokenized_paded_documents.shape)
    # print("Padded document : ", tokenized_paded_documents[0])

    # creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index.

    # creating 0 filled matrix of size vocab_sizexmodel_dimensions
    embedding_matrix = np.zeros((vocab_size, model_dimensions))
    print("Embedding matrix dimensions : ", embedding_matrix.shape)

    for word, i in tokenizer.word_index.items():
        if word in model:
            embedding_matrix[i] = model[word]

    # print(len(embedding_matrix[1]),embedding_matrix[1])

    tfidfvectoriser = TfidfVectorizer(max_features=300)
    tfidfvectoriser.fit(documents_df.documents_cleaned)
    tfidf_vectors = tfidfvectoriser.transform(
        documents_df.documents_cleaned).toarray()

    # representing every paragraph with vocab_size (91)
    print(tfidf_vectors[0])
    # pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T).toarray()
    # pairwise_differences=euclidean_distances(tfidf_vectors)

    def most_similar(doc_id, similarity_matrix):
        # print (f'Document: {documents_df.iloc[doc_id]["documents"]}')
        # print ('\n')
        # print ('Similar Documents:')

        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]

        for ix in similar_ix:
            if ix == doc_id:
                continue
            print('\n')
            print(f'Document: {documents_df.iloc[ix]["documents"]}')
            print("Cosine similarity : ", similarity_matrix[doc_id][ix])

    # calculating average of word vectors of a document weighted by tf-idf
    document_embeddings = np.zeros(
        (len(tokenized_paded_documents), model_dimensions))
    words = tfidfvectoriser.get_feature_names_out()

    # print("Document embeddings shape : ", document_embeddings.shape)
    # print("Total words : ", len(words))
    # print("TFIDF vector shape : ", tfidf_vectors.shape)
    # print("Embedding matrix shape : ", embedding_matrix.shape)
    # print("Word : ", words[0], ", Word index : ",
    #       tokenizer.word_index[words[0]])

    print(tfidf_vectors[0])
    # print(embedding_matrix[tokenizer.word_index[words[0]]].shape)
    for i in range(len(tokenized_paded_documents)):
        for j in range(len(words)):
            embedding = embedding_matrix[tokenizer.word_index[words[j]]]
            tfidf_vector = tfidf_vectors[i][j]
            document_embeddings[i] += embedding*tfidf_vector

    # document_embeddings=document_embeddings/np.sum(tfidf_vectors,axis=1).reshape(-1,1)

    document_embeddings.shape

    pairwise_similarities = cosine_similarity(document_embeddings)
    print(pairwise_similarities.shape, pairwise_similarities)

    most_similar(0, pairwise_similarities)

    return pairwise_similarities

@app.route('/compare_data',methods = ['POST'])
def compare_data():
    try:
        docs = request.json

        first = docs.get('First')
        second = docs.get('Second')

        if(first is None or second is None):
            raise Exception("Docs names not correct")
        
        #send data to model
        print("First doc : ", first)

        return "OK"
    except Exception as e:
        logging.error(traceback.format_exc())
        return Response(response="Bad Request", status=400)


   
