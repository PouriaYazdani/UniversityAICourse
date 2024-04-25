import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pickle as pk

folding = True


def read():
    return pd.read_csv(r'D:\ca4\utility\cleaned.csv', encoding='utf-8')


def doc_term_matrix(train):
    shuffled_data = pd.read_csv(r"D:\ca4\new\train.csv",encoding='utf-8')
    texts = [eval(row) if isinstance(row, str) else row for row in shuffled_data['text']]
    all_attrs =list(pd.read_csv(r"D:\ca4\new\all_attrs.csv",encoding='utf-8')['all_attrs'])
    print(len(all_attrs))
    matrix = np.zeros((len(texts), len(all_attrs)), dtype = np.int16)
    word_idx = {word: i for i, word in enumerate(all_attrs)}
    for i, doc in enumerate(texts):
        for word in doc:
            matrix[i, word_idx[word]] += 1
    return matrix



def dim_reduction(matrix):
    pca = PCA(n_components=800)
    pca.fit(matrix)
    pk.dump(pca,open(r"D:\ca4\new\pca_object.pkl","wb"))
    train_reduced =  pca.transform(matrix)
    pk.dump(train_reduced,open(r"D:\ca4\new\train_reduced.pkl","wb"))


    # pca1 = PCA(0.9)
    # pca2 = PCA(n_components=800)
    # pca3 = PCA(n_components=1000)
    # pca1.fit(matrix)
    # pca2.fit(matrix)
    # pca3.fit(matrix)
    # print(pca1.n_components)
    # print(pca2.explained_variance_)
    # print(pca2.explained_variance_ratio_)
    # print(pca3.explained_variance_)
    # print(pca3.explained_variance_ratio_)
    # pk.dump(pca1,open(r"D:\ca4\utility\pca1.pkl","wb"))
    # pk.dump(pca2, open(r"D:\ca4\utility\pca800.pkl", "wb"))
    # pca = pk.load(open(r"D:\ca4\utility\pca3.pkl", "rb"))
    # transformed = pca2.fit_transform(matrix)
    # df_transformed = pd.DataFrame(transformed, columns=[f"PC{i}" for i in range(1, transformed.shape[1] + 1)])
    # pk.dump(df_transformed,open(r"D:\ca4\utility\train_reduced800.pkl",'wb'))
    # pk.dump(transformed,open(r"D:\ca4\utility\transformed800.pkl",'wb'))


def write(data):
    df = pd.DataFrame(data)
    df.to_csv(r'D:\ca4\all\doc_term_mat.csv', index=False, encoding='utf-8')


def main():
    train = read()
    matrix = doc_term_matrix(train)
    dim_reduction(matrix)


if __name__ == "__main__":
    main()
