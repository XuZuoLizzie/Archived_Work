"""
@Project : COVID-19 Preprint classification
@File    : Feature engineering
@Time    : 2021/1/7 21:07
@Author  : Xu Zuo
@Class   : BMI 6319 Spring 2021
"""
# import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


def get_features(data):
    """
    Generate features for preprint text
    Encode label_1 (study type: RCT, observational study, other)
    Encode label_2 (center: single center, multicenter)
    :param data: Preprint dataset
    :return: weights, label_1, label_2
    """
    c_vect = CountVectorizer()
    tfidf_trans = TfidfTransformer()

    count = c_vect.fit_transform(data['text'])
    weights = tfidf_trans.fit_transform(count)

    le = LabelEncoder()
    label_1 = le.fit_transform(data['study type'])
    label_2 = le.fit_transform(data['center'])

    return weights, label_1, label_2


def oversampling(X, y):
    """
    The dataset is heavily imbalanced.
    Use SMOTE method to over-sample minority classes
    :param X: tf-idf weights for preprint text
    :param y: preprint labels
    :return: X_over, y_over
    """
    oversample = SMOTE(sampling_strategy='not majority', random_state=27)
    X_over, y_over = oversample.fit_resample(X, y)
    return X_over, y_over


if __name__ == '__main__':
    dataset = '../data/preprint_classification_data.tsv'
    df = pd.read_csv(dataset, sep='\t')

    X, y1, y2 = get_features(df)
    print(y1)
    print(df['study type'])