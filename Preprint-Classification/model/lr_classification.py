"""
@Project : COVID-19 Preprint classification
@File    : Classification model - logistic regression
@Time    : 2021/1/7 21:07
@Author  : Xu Zuo
@Class   : BMI 6319 Spring 2021
"""
# import libraries
import pandas as pd
from feature_generation import get_features, oversampling
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def clf_lr(class_type, model_type):
    """
    Generate classification pipeline using logistic regression.

    :param class_type: study type (multi-class), center(binary)
    :return:
    """
    if class_type == 'study type':
        if model_type == 'lr':
            pipe = Pipeline([('sc', StandardScaler(with_mean=False)),
                             ('poly', PolynomialFeatures(degree=1)),
                             ('clf', LogisticRegression(C=10))])
        elif model_type == 'xgb':
            pipe = XGBClassifier(learning_rate=0.3,
                                 n_estimators=100,
                                 max_depth=3,
                                 min_child_weight=1,
                                 gamma=0.1,
                                 subsample=0.89,
                                 colsample_bytree=1,
                                 objective='multi:softprob',
                                 eval_metric='mlogloss',
                                 nthread=4,
                                 seed=27,
                                 use_label_encoder=False)

        elif model_type == 'svm':
            pipe = SVC(C=1, probability=True, class_weight={0: 0.2, 1: 0.8})
        else:
            print('Please specify the model type.')
    elif class_type == 'center':
        pipe = Pipeline([('sc', StandardScaler()),
                         ('poly', PolynomialFeatures(degree=1)),
                         ('clf', LogisticRegression(C=1))])
    else:
        print('Please specify the classification task.')
    return pipe


if __name__ == '__main__':
    dataset = '../data/preprint_classification_data.tsv'
    df = pd.read_csv(dataset, sep='\t')

    X, y1, y2 = get_features(df)
    # X_over, y1_over = oversampling(X, y1)
    X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=42)

    X_smote, y_smote = oversampling(X_train, y_train)
    model = clf_lr(class_type='study type', model_type='lr')
    # model.fit(X_train, y_train)
    model.fit(X_smote, y_smote)
    y_predict = model.predict(X_test)
    results = classification_report(y_test,
                                    y_predict,
                                    zero_division=0,
                                    target_names=['RCT', 'observational study', 'other'])
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    print(results)
    print('Area under ROC curve: {}'.format(auc_score))