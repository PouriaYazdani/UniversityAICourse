import numpy as np
import pickle as pk
import pandas as pd
import util
from sklearn.decomposition import PCA
from tqdm import tqdm


class KNNClassifier():
    def __init__(self, k, train: np.ndarray, test: np.ndarray):
        self.k = k
        self.train = train
        self.test = test

    def predict_all(self):
        result = [self._predict(test_instance) for test_instance in tqdm(self.test)]
        return result

    def _cosine_similarity(self, test_vect, train_vect):
        dot_product = np.dot(test_vect, train_vect)
        norm_a = np.linalg.norm(test_vect)
        norm_b = np.linalg.norm(train_vect)
        similarity = dot_product / (norm_a * norm_b)
        return similarity

    def _predict(self, test):
        k = self.k
        distances = {index: self._cosine_similarity(test, train_instance[:-1])
                     for index, train_instance in enumerate(self.train)}  # HERE ERROR

        sorted_distances = sorted(distances, key=distances.get, reverse=True)
        k_nearest = sorted_distances[:k]
        sport = politics = 0

        for idx in k_nearest:
            if self.train[idx, -1] == 1:
                sport += 1
            else:
                politics += 1
        # Determine the predicted class
        if sport >= politics:
            return 1
        else:
            return 0


def kfold_indices(data: pd.DataFrame, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds

def main():
    #  TEST INPUT
    raw_texts = pd.read_csv(r"D:\ca4\test\nlp_test.csv",encoding="utf-8")
    test_tags = pd.read_csv(r"D:\ca4\test\nlp_test.csv", encoding="utf-8")['Category'].tolist()
    # LOADING PARAMETERS
    all_attrs = pd.read_csv(r"D:\ca4\new\all_attrs.csv", encoding='utf-8')['all_attrs'].tolist()
    train_tags = pd.read_csv(r"D:\ca4\new\train.csv")['tag'].tolist()
    reduced_train = pk.load(open(r"D:\ca4\new\train_reduced.pkl", "rb"))
    reduced_train_labeled = np.column_stack((reduced_train, train_tags))
    pca = pk.load(open(r"D:\ca4\new\pca_object.pkl", "rb"))
    pca: PCA
    # PREPROCESS TEST DATA
    test_tags = util.encode_tags(test_tags)
    texts_cleaned = util.preprocess(raw_texts)
    # texts_cleaned.to_csv(r"D:\ca4\test\test_cleaned.csv",index=False,encoding='utf-8')
    texts_cleaned = pd.read_csv(r"D:\ca4\test\test_cleaned.csv",encoding='utf-8')
    all_attrs = pd.read_csv(r"D:\ca4\new\all_attrs.csv",encoding='utf-8')['all_attrs']
    test_dtm = util.create_dtm(texts_cleaned, all_attrs)
    reduced_test = pca.transform(test_dtm)
    # APPLY THE MODEL ON TEST DATA
    classifier = KNNClassifier(15, reduced_train_labeled, reduced_test)
    results = classifier.predict_all()
    result_df = pd.DataFrame({'prediction' : results})
    result_df.to_csv(r"D:\ca4\test\prediction.csv",index=False,encoding='utf-8')
    f_score = util.evaluate(results, test_tags)
    print(f_score)


if __name__ == "__main__":
    main()