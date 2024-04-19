#! /usr/bin/python
"""
@Project : COVID-19 Preprint classification
@File    : Classification model - for docker file
@Time    : 2021/1/7 21:07
@Author  : Xu Zuo
@Class   : BMI 6319 Spring 2021
"""
# import libraries
import pandas as pd
from model.feature_generation import get_features, oversampling
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def clf_lr(class_type):
    """
    Generate classification pipeline using logistic regression.

    :param class_type: study type (multi-class), center(binary)
    :return:
    """
    if class_type == 'study type':
        pipe = Pipeline([('sc', StandardScaler(with_mean=False)),
                         ('poly', PolynomialFeatures(degree=1)),
                         ('clf', LogisticRegression(C=10))])
    elif class_type == 'center':
        pipe = Pipeline([('sc', StandardScaler()),
                         ('poly', PolynomialFeatures(degree=1)),
                         ('clf', LogisticRegression(C=1))])
    else:
        print('Please specify the classification task.')
    return pipe


if __name__ == '__main__':
    dataset = 'data/preprint_classification_data.tsv'
    df = pd.read_csv(dataset, sep='\t')

    X, y1, y2 = get_features(df)
    X_over, y1_over = oversampling(X, y1)
    X_test, X_train, y_test, y_train = train_test_split(X_over, y1_over, test_size=0.2, random_state=42)

    model = clf_lr(class_type='study type')
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    results = classification_report(y_test, y_predict, target_names=['RCT', 'observational study', 'other'])
    print(results)