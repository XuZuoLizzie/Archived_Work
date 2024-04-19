"""
@Project : COVID-19 Preprint classification
@File    : Convert data for BERT classification input
@Time    : 2021/1/7 21:07
@Author  : Xu Zuo
@Class   : BMI 6319 Spring 2021
"""
# import libraries
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


def load_preprints(data_path):
    df = pd.read_csv(data_path, sep='\t')
    data = df[['text', 'study type']]
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
    return data_train, data_test


def generator_preprints(data):
    for row in data.itertuples(name=None):
        yield row


if __name__ == '__main__':
    dataset = '../data/preprint_classification_data.tsv'
    data_train, data_test = load_preprints(dataset)
    train = generator_preprints(data_train)
    dev = generator_preprints(data_test)

    labels = ['RCT', 'observational study', 'other']
    train_documents, train_labels = [], []
    for _, text, status in train:
        train_documents.append(text)
        label = [0] * len(labels)
        for idx, name in enumerate(labels):
            if name == status:
                label[idx] = 1
        train_labels.append(label)
    # print(train_labels)
    # df = pd.read_csv(dataset, sep='\t')
    # print(df)

    '''
    lb = LabelBinarizer()
    label_1 = lb.fit_transform(df['study type'])
    train_labels = label_1[:8].tolist()
    train_doc = df['text'][:8].tolist()
    dev_labels = label_1[8:10].tolist()
    dev_doc = df['text'][8:10].tolist()
    print(train_labels)
    print(train_doc)
    '''