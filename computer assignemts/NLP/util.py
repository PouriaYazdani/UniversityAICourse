import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pickle as pk
import lexical


def create_dtm(data: pd.DataFrame, all_attrs):
    all_attrs = list(all_attrs)
    texts = [eval(row) if isinstance(row, str) else row for row in data['text']]
    matrix = np.zeros((len(texts), len(all_attrs)), dtype=int)
    word_idx = {word: i for i, word in enumerate(all_attrs)}
    for i, doc in enumerate(texts):
        for word in doc:
            if word in all_attrs:
                matrix[i, word_idx[word]] += 1
    return matrix


def all_attrs(data: pd.DataFrame):
    texts = [eval(row) if isinstance(row, str) else row for row in data['text']]
    all_attrs = list(set(word for doc in texts for word in doc))
    print(len(all_attrs))
    return all_attrs


def ignore_redundant_attrs(test: pd.DataFrame, all_attrs: list):
    test['text'] = test['text'].apply(lambda words: [word for word in words if word in all_attrs])




def dim_reduction(train_dtm: np.ndarray,test_dtm: np.ndarray,i, k=800):
    pca = PCA(k)
    pca.fit(train_dtm)
    train_reduced = pca.transform(train_dtm)
    test_reduced = pca.transform(test_dtm)
    path_test = r"D:\ca4\10-fold\pcatest{}.pkl".format(i)
    path_train = r"D:\ca4\10-fold\pcatrain{}.pkl".format(i)
    pk.dump(test_reduced, open(path_test, "wb"))
    pk.dump(train_reduced, open(path_train, "wb"))
    return train_reduced,test_reduced


def encode_tags(tags):
    tags_encoded = [1 if (tag == 's' or tag == 'Sport') else 0 for tag in tags]  # s = 1, p = 0
    return tags_encoded


def preprocess(test_df:pd.DataFrame):
    ''' just preprocesses the data
      :returns pandas df with 2 columns text and tag
       '''
    normalized_data = lexical.normalize(test_df)
    tokenized_data = lexical.tokenize(normalized_data)
    removed_redundants = lexical.remove_redundnat(tokenized_data)
    lemmatized = lexical.lemmatize(removed_redundants)
    removed_stopwords = lexical.remove_stopword(lemmatized)

    cleaned = pd.DataFrame({
        'text':removed_stopwords,
        'tag':test_df['Category']
    })
    return cleaned


def evaluate(predicted:list,actual:list):
    TP=TN=FN=FP = 0
    for y_predicted, y_actual in zip(predicted,actual):
        if y_predicted == 1 and y_actual == 1:
            TP += 1
        elif y_predicted == 0 and y_actual == 0:
            TN += 1
        elif y_predicted == 0 and y_actual == 1:
            FN += 1
        else:
            FP += 1

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    f_score = 2 * (precision * recall) / (precision + recall)
    return f_score
