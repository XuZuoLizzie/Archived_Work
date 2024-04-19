import gzip
import math
import re
import sklearn
from sklearn import svm
from sklearn.cluster import KMeans

import numpy as np
from sklearn import feature_extraction
from sklearn import metrics

import util_5353

# Problem A [0 points]
def read_data(filenames):
  data = None
  # Begin CODE
  article_list = []
  records = []
  for file in filenames:
    with gzip.open(file, 'rt') as f:
      content = f.read()
    article_list.extend(content.split('\n\n'))

  for article in article_list:
    record = {}
    key = ''
    lines = article.split('\n')
    for line in lines:
      line = line.rstrip()
      if line[:6] == "      ":
        if key == 'MH':
          record[key][-1] += line[5:]
        else:
          record[key].append(line[6:])
      elif line:
        key = line[:4].rstrip()
        if key not in record:
          record[key] = []
        record[key].append(line[6:])
    records.append(record)

  for record in records:
    record['PMID'] = "".join(record['PMID'])
    record['AB'] = " ".join(record['AB'])
    record['TI'] = " ".join(record['TI'])
    # print(record['TI'])

  data = records
  # End CODE
  return data

# Problem B [0 points]
tokenizer = re.compile('\w+|[^\s\w]+')
def tokenize(text):
  return tokenizer.findall(text.lower())

# Problem C [0 points]
def pmids(data):
  pmids = []
  # Begin CODE
  for record in data:
    pmids.append(record['PMID'])
  # End CODE
  return pmids

# Problem 1 [10 points]
def unigrams(data, pmid):
  unigrams = {}
  # Begin CODE
  pmid_list = []
  for record in data:
    pmid_list.append(record['PMID'])

  article = data[pmid_list.index(pmid)]
  title = article['TI']
  abstract = article['AB']

  text = title + abstract
  tokens = tokenize(text)
  unitokens = np.unique(np.array(tokens)).tolist()
  values = np.ones(len(unitokens)).tolist()
  unigrams = dict(zip(unitokens, values))
  # End CODE
  return unigrams

# Problem 2 [10 points]
def tfidf(data, pmid):
  tfidf = {}
  # Begin CODE
  pmid_list = []
  for record in data:
    pmid_list.append(record['PMID'])

  article = data[pmid_list.index(pmid)]
  title = article['TI']
  abstract = article['AB']

  text = title + abstract
  tokens = tokenize(text)
  unitokens, freq = np.unique(np.array(tokens), return_counts=True)
  tf_dict = dict(zip(unitokens.tolist(), freq.tolist()))

  corpus = []
  for i, record in enumerate(data):
    search_tokens = tokenize(record['TI'] + record['AB'])
    corpus.append(np.unique(np.array(search_tokens)).tolist())

  num = len(data)
  idf_dict = dict.fromkeys(tf_dict.keys(), 0)
  for token in tf_dict.keys():
    for doc in corpus:
      if token in doc:
        idf_dict[token] += 1

  for token, val in idf_dict.items():
    idf_dict[token] = math.log(num / float(val))

  tfidf_dict = dict.fromkeys(tf_dict.keys(), 0)
  for token, val in tf_dict.items():
    tfidf_dict[token] = val * idf_dict[token]
  tfidf = tfidf_dict
  # End CODE
  return tfidf

# Problem 3 [10 points]
def mesh(data, pmid):
  mesh = []
  # Begin CODE
  pmid_list = []
  for record in data:
    pmid_list.append(record['PMID'])

  article = data[pmid_list.index(pmid)]
  for term in article['MH']:
    base_term = term.split('/')[0]
    base_term = re.sub('\*', '', base_term)
    mesh.append(base_term)
  # End CODE
  return mesh

# Problem 4 [10 points]
def svm_predict_unigram(data, train, test, mesh_list):
  predictions = {m:[] for m in mesh_list}
  # Begin CODE
  train_features = []
  for pmid in train:
    train_features.append(unigrams(data, pmid))
  vec = feature_extraction.DictVectorizer(sparse=False)
  X_train = vec.fit_transform(train_features)

  label_dict = {}
  for m in mesh_list:
    labels = []
    for pmid in train:
      mh_terms = mesh(data, pmid)
      if m in mh_terms:
        label = 1
      else:
        label = 0
      labels.append(label)
    label_np = np.array(labels)
    label_dict.update({m: label_np})

  test_features = []
  for pmid in test:
    test_features.append(unigrams(data, pmid))
  X_test = vec.transform(test_features)

  output = {}
  for m in mesh_list:
    y_train = label_dict[m]
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    output.update({m: y_predict})

  predictions = {}
  for m in mesh_list:
    list_predict = []
    y_predict = output[m]
    index_list = np.where(y_predict == 1)[0]
    for index in index_list:
      list_predict.append(test[index])
    predictions.update({m: list_predict})
  # End CODE
  return predictions

# Problem 5 [10 points]
def svm_predict_tfidf(data, train, test, mesh_list):
  predictions = {m:[] for m in mesh_list}
  # Begin CODE
  train_features = []
  for pmid in train:
    train_features.append(tfidf(data, pmid))
  vec = feature_extraction.DictVectorizer(sparse=False)
  X_train = vec.fit_transform(train_features)

  label_dict = {}
  for m in mesh_list:
    labels = []
    for pmid in train:
      mh_terms = mesh(data, pmid)
      if m in mh_terms:
        label = 1
      else:
        label = 0
      labels.append(label)
    label_np = np.array(labels)
    label_dict.update({m: label_np})
  
  test_features = []
  for pmid in test:
    test_features.append(unigrams(data, pmid))
  X_test = vec.transform(test_features)

  output = {}
  for m in mesh_list:
    y_train = label_dict[m]

    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    output.update({m: y_predict})

  predictions = {}
  for m in mesh_list:
    list_predict = []
    y_predict = output[m]
    index_list = np.where(y_predict == 1)[0]
    for index in index_list:
      list_predict.append(test[index])
    predictions.update({m: list_predict})
  # End CODE
  return predictions

# Problem 6 [10 points]
def kmeans(data, k):
  clusters = {}
  # Begin CODE
  pmid_list = []
  for record in data:
    pmid_list.append(record['PMID'])
  features = []
  for pmid in pmid_list:
    features.append(unigrams(data, pmid))
  vec = feature_extraction.DictVectorizer(sparse=False)
  X = vec.fit_transform(features)
  kmeans = KMeans(n_clusters=k, random_state=0, init='random').fit(X)
  cluster_list = kmeans.labels_.tolist()

  clusters = dict.fromkeys(pmid_list, 0)
  for pmid in pmid_list:
    index_num = pmid_list.index(pmid)
    clusters[pmid] = cluster_list[index_num]
  # End CODE
  return clusters

# Problem 7 [10 points]
def svm_predict_cluster(data, train, test, mesh_list, k):
  predictions = {m:[] for m in mesh_list}
  # Begin CODE
  clusters = kmeans(data, k)
  train_features = []
  for pmid in train:
    train_features.append(clusters[pmid])
  X_train = np.array(train_features).reshape((-1, 1))

  label_dict = {}
  for m in mesh_list:
    labels = []
    for pmid in train:
      mh_terms = mesh(data, pmid)
      if m in mh_terms:
        label = 1
      else:
        label = 0
      labels.append(label)
    label_np = np.array(labels)
    label_dict.update({m: label_np})

  test_features = []
  for pmid in test:
    test_features.append(clusters[pmid])
  X_test = np.array(test_features).reshape((-1, 1))

  output = {}
  for m in mesh_list:
    y_train = label_dict[m]

    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    output.update({m: y_predict})

  predictions = {}
  for m in mesh_list:
    list_predict = []
    y_predict = output[m]
    index_list = np.where(y_predict == 1)[0]
    for index in index_list:
      list_predict.append(test[index])
    predictions.update({m: list_predict})

  # End CODE
  return predictions

# Problem 8 [10 points]
def svm_predict_cluster_unigrams(data, train, test, mesh_list, k):
  predictions = {m:[] for m in mesh_list}
  # Begin CODE
  clusters = kmeans(data, k)
  train_unigrams = []
  train_clusters = []
  for pmid in train:
    train_unigrams.append(unigrams(data, pmid))
    train_clusters.append(clusters[pmid])
  vec = feature_extraction.DictVectorizer(sparse=False)
  train_unigram = vec.fit_transform(train_unigrams)
  train_cluster = np.array(train_clusters).reshape((-1, 1))
  X_train = np.concatenate((train_unigram, train_cluster), axis=1)

  label_dict = {}
  for m in mesh_list:
    labels = []
    for pmid in train:
      mh_terms = mesh(data, pmid)
      if m in mh_terms:
        label = 1
      else:
        label = 0
      labels.append(label)
    label_np = np.array(labels)
    label_dict.update({m: label_np})

  test_unigrams = []
  test_clusters = []
  for pmid in test:
    test_unigrams.append(unigrams(data, pmid))
    test_clusters.append(clusters[pmid])
  test_unigram = vec.transform(test_unigrams)
  test_cluster = np.array(test_clusters).reshape((-1, 1))
  X_test = np.concatenate((test_unigram, test_cluster), axis=1)

  output = {}
  for m in mesh_list:
    y_train = label_dict[m]
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    output.update({m: y_predict})

  predictions = {}
  for m in mesh_list:
    list_predict = []
    y_predict = output[m]
    index_list = np.where(y_predict == 1)[0]
    for index in index_list:
      list_predict.append(test[index])
    predictions.update({m: list_predict})
  # End CODE
  return predictions

# Problem 9 [20 points]
def evaluate(data, test, mesh_predict):
  evaluation = {}
  # Begin CODE
  mesh_list = list(mesh_predict.keys())

  label_test_dict = {}
  for m in mesh_list:
    test_labels = np.zeros(len(test))
    for pmid in test:
      mh_terms = mesh(data, pmid)
      if m in mh_terms:
        test_labels[test.index(pmid)] = 1
    label_test_dict.update({m: test_labels})

  label_predict_dict = {}
  for m in mesh_list:
    predict_labels = np.zeros((len(test)))
    for pmid in test:
      if pmid in mesh_predict[m]:
        predict_labels[test.index(pmid)] = 1
    label_predict_dict.update({m: predict_labels})

  result_dict = {}
  for m in mesh_list:
    y_test = label_test_dict[m]
    y_predict = label_predict_dict[m]
    accuracy = metrics.accuracy_score(y_test, y_predict)
    prf = metrics.precision_recall_fscore_support(y_test, y_predict, average='binary', zero_division=0)
    results = {'accuracy': float(accuracy), 'precision': float(prf[0]), 'recall': float(prf[1]), 'f1': float(prf[2])}
    result_dict.update({m: results})
  evaluation = result_dict
  # End CODE
  return evaluation

# Note: don't mess with this code block!  Your code will be tested by an outside
# program that will not call this __main__ block.  So if you mess with the
# following block of code you might crash the autograder.  You're definitely
# encouraged to look at this code, however, especially if your code crashes.
if __name__ == '__main__':

  # Comment out some file names to speed up the development process, but
  # ultimately you want to uncomment the filenames so you ensure that your code
  # works will all files.  The assertions below assume that medline.0.txt.gz is
  # in the list.
  file_list = []
  file_list.append('medline.0.txt.gz')
  file_list.append('medline.1.txt.gz')
  file_list.append('medline.2.txt.gz')
  file_list.append('medline.3.txt.gz')
  file_list.append('medline.4.txt.gz')
  file_list.append('medline.5.txt.gz')
  file_list.append('medline.6.txt.gz')
  file_list.append('medline.7.txt.gz')
  file_list.append('medline.8.txt.gz')
  file_list.append('medline.9.txt.gz')
  
  pmid_list = ['22999938', '23010078', '23018989']

  print('::: Problem A :::')
  data = read_data(file_list)

  print('::: Problem C :::')
  pmids = pmids(data)
  for pmid in pmid_list:
    if pmid not in pmids:
      util_5353.die('C', 'Assertions assume PMID is present: %s', pmid)

  tts = int(len(pmids) * 0.8)
  train = pmids[:tts]
  test = pmids[tts:]

  print('::: Problem 1 :::')
  one_ret = unigrams(data, pmid_list[0])
  util_5353.assert_dict(one_ret, '1')
  util_5353.assert_int_eq(99, len(one_ret), '1')
  util_5353.assert_float_eq(1.0, one_ret['metastasis'], '1')
  one_ret = unigrams(data, pmid_list[1])
  util_5353.assert_dict(one_ret, '1')
  util_5353.assert_int_eq(95, len(one_ret), '1')
  util_5353.assert_float_eq(1.0, one_ret['destruction'], '1')
  one_ret = unigrams(data, pmid_list[2])
  util_5353.assert_dict(one_ret, '1')
  util_5353.assert_int_eq(133, len(one_ret), '1')
  util_5353.assert_float_eq(1.0, one_ret['concurrent'], '1')

  print('::: Problem 2 :::')
  two_ret = tfidf(data, pmid_list[0])
  util_5353.assert_dict(two_ret, '2')
  util_5353.assert_int_eq(99, len(two_ret), '2')
  util_5353.assert_float_range((1.5, 3.0), two_ret['metastasis'], '2')
  two_ret = tfidf(data, pmid_list[1])
  util_5353.assert_dict(two_ret, '2')
  util_5353.assert_int_eq(95, len(two_ret), '2')
  util_5353.assert_float_range((10.0, 20.0), two_ret['destruction'], '2')
  two_ret = tfidf(data, pmid_list[2])
  util_5353.assert_dict(two_ret, '2')
  util_5353.assert_int_eq(133, len(two_ret), '2')
  util_5353.assert_float_range((7.0, 10.0), two_ret['concurrent'], '2')

  print('::: Problem 3 :::')
  three_ret = mesh(data, pmid_list[0])
  GOLD = ['Animals', 'Breast Neoplasms', 'DNA Methylation', 'DNA, Neoplasm', 'DNA-Binding Proteins', 'Dioxygenases', 'Down-Regulation', 'Female', 'Gene Expression Regulation, Neoplastic', 'Humans', 'Male', 'Mice', 'Mice, Inbred BALB C', 'Mice, Nude', 'Mixed Function Oxygenases', 'Neoplasm Invasiveness', 'Prostatic Neoplasms', 'Proto-Oncogene Proteins', 'Tissue Inhibitor of Metalloproteinase-2', 'Tissue Inhibitor of Metalloproteinase-3', 'Tumor Suppressor Proteins']
  util_5353.assert_list(three_ret, len(GOLD), '3', valid_values=GOLD)
  three_ret = mesh(data, pmid_list[1])
  GOLD = ['Animals', 'Contrast Media', 'Gene Knockdown Techniques', 'Genetic Therapy', 'Mice', 'Mice, Inbred C3H', 'Microbubbles', 'Neoplasms, Squamous Cell', 'RNA, Small Interfering', 'Receptor, Epidermal Growth Factor', 'Sonication', 'Transfection', 'Ultrasonics', 'Ultrasonography']
  util_5353.assert_list(three_ret, len(GOLD), '3', valid_values=GOLD)
  three_ret = mesh(data, pmid_list[2])
  GOLD = ['Adult', 'Aged', 'Chemoradiotherapy', 'Diffusion Magnetic Resonance Imaging', 'Female', 'Humans', 'Medical Oncology', 'Middle Aged', 'Reproducibility of Results', 'Time Factors', 'Treatment Outcome', 'Tumor Burden', 'Uterine Cervical Neoplasms']
  util_5353.assert_list(three_ret, len(GOLD), '3', valid_values=GOLD)

  print('::: Problem 4 :::')
  mesh_list = ['Humans', 'Female', 'Male', 'Animals', 'Treatment Outcome',
               'Neoplasms', 'Prognosis', 'Risk Factors', 'Breast Neoplasms', 'Lung Neoplasms']
  mesh_set = set()
  for pmid in pmids:
    mesh_set.update(mesh(data, pmid))
  for m in mesh_list:
    if m not in mesh_set:
      util_5353.die('4', 'Assertions assume MeSH term is present: %s', m)
  four_ret = svm_predict_unigram(data, train, test, mesh_list)
  util_5353.assert_dict(four_ret, '4')
  for m in mesh_list:
    util_5353.assert_dict_key(four_ret, m, '4')
    util_5353.assert_list(four_ret[m], None, '4', valid_values=pmids)
    util_5353.assert_int_range((0, len(test)), len(four_ret[m]), '4')
  util_5353.assert_int_range((len(test)/2, len(test)), len(four_ret['Humans']), '4')

  print('::: Problem 5 :::')
  five_ret = svm_predict_tfidf(data, train, test, mesh_list)
  util_5353.assert_dict(five_ret, '5')
  for m in mesh_list:
    util_5353.assert_dict_key(five_ret, m, '5')
    util_5353.assert_list(five_ret[m], None, '5', valid_values=pmids)
    util_5353.assert_int_range((0, len(test)), len(five_ret[m]), '5')
  util_5353.assert_int_range((len(test)/2, len(test)), len(five_ret['Humans']), '5')

  print('::: Problem 6 :::')
  K = 10
  six_ret = kmeans(data, K)
  util_5353.assert_dict(six_ret, '6')
  util_5353.assert_int_eq(len(pmids), len(six_ret), '6')
  for pmid in pmids:
    util_5353.assert_dict_key(six_ret, pmid, '6')
    util_5353.assert_int_range((0, K-1), six_ret[pmid], '6')

  print('::: Problem 7 :::')
  seven_ret = svm_predict_cluster(data, train, test, mesh_list, K)
  util_5353.assert_dict(seven_ret, '7')
  for m in mesh_list:
    util_5353.assert_dict_key(seven_ret, m, '7')
    util_5353.assert_list(seven_ret[m], None, '7', valid_values=pmids)
    util_5353.assert_int_range((0, len(test)), len(seven_ret[m]), '7')
  util_5353.assert_int_range((len(test)/2, len(test)), len(seven_ret['Humans']), '7')

  print('::: Problem 8 :::')
  eight_ret = svm_predict_cluster_unigrams(data, train, test, mesh_list, K)
  util_5353.assert_dict(eight_ret, '8')
  for m in mesh_list:
    util_5353.assert_dict_key(eight_ret, m, '8')
    util_5353.assert_list(eight_ret[m], None, '8', valid_values=pmids)
    util_5353.assert_int_range((0, len(test)), len(eight_ret[m]), '8')
  util_5353.assert_int_range((len(test)/2, len(test)), len(eight_ret['Humans']), '8')

  print(':: Problem 9 ::')
  nine_ret4 = evaluate(data, test, four_ret)
  nine_ret5 = evaluate(data, test, five_ret)
  nine_ret7 = evaluate(data, test, seven_ret)
  nine_ret8 = evaluate(data, test, eight_ret)
  for nine_ret in [nine_ret4, nine_ret5, nine_ret7, nine_ret8]:
    util_5353.assert_dict(nine_ret, '9')
    for m in mesh_list:
      util_5353.assert_dict_key(nine_ret, m, '9')
      util_5353.assert_dict(nine_ret[m], '9')
      for k in ['accuracy', 'precision', 'recall', 'f1']:
        util_5353.assert_dict_key(nine_ret[m], k, '9')
        util_5353.assert_float(nine_ret[m][k], '9')
        util_5353.assert_float_range((0.0, 1.0), nine_ret[m][k], '9')

  print('~~~ All Tests Pass ~~~')
