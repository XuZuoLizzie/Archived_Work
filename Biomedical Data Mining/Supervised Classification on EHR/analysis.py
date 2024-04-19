import sklearn
from sklearn.impute import SimpleImputer  # use Imputer if sklearn v0.19
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import csv
from statistics import mean
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt

import util_5353

# Problem A [0 points]
def read_data(dataset_id):
  data = None
  # Begin CODE
  if dataset_id == 'breast_cancer':
    data_list = []
    with open('wdbc.data', 'r') as f:
      reader_obj = csv.reader(f, delimiter=',')
      for i, row in enumerate(reader_obj, 1):
        label = [row[1]]
        feature = row[2:32]
        data_list.append(feature + label)
    data = np.array(data_list)

  elif dataset_id == 'hyperthyroidism':
    file_list = ['allhyper.data', 'allhyper.test']
    dataset_dict = {}
    for filename in file_list:
      data_list = []
      with open(filename, 'r') as f:
        reader_obj = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader_obj, 1):
          label = [row[-1].split('.')[0]]
          feature = row[0:29]
          data_list.append(feature + label)
      dataset_dict[filename] = np.array(data_list)

    data = np.concatenate((dataset_dict[file_list[0]], dataset_dict[file_list[1]]), axis=0)

  elif dataset_id == 'cervical_cancer':
    data_list = []
    with open('risk_factors_cervical_cancer.csv', 'r') as f:
      reader_obj = csv.reader(f, delimiter=',')
      next(f)
      for i, row in enumerate(reader_obj, 1):
        label = [row[-1]]
        feature = row[0:28]
        data_list.append(feature + label)
    data = np.array(data_list)

  elif dataset_id == 'liver_cancer':
    data_list = []
    with open('Indian Liver Patient Dataset (ILPD).csv', 'r') as f:
      reader_obj = csv.reader(f, delimiter=',')
      for i, row in enumerate(reader_obj, 1):
        data_list.append(row)
    data = np.array(data_list)

  else:
    print('Dataset does not exist.')
  # print(data)
  # End CODE
  return data

# Problem B [0 points]
def dimensions(dataset_id, dataset):
  dim = None
  # Begin CODE
  row, col = dataset.shape
  dim = (row, col-1)
  # End CODE
  return dim

# Problem C [0 points]
def feature_values(dataset_id, dataset):
  fvalues = []
  # Begin CODE
  row, col = dataset.shape
  features = np.transpose(dataset[:, 0:col - 1])
  features_list = features.tolist()

  def is_number(n):
    try:
      float(n)
    except ValueError:
      return False
    return True

  for num, feature in enumerate(features_list, 1):
    mask = [is_number(value) for value in feature]
    if True in mask:
      value_list = []
      for i in range(len(feature)):
        if mask[i] == True:
          value_list.append(feature[i])
      value_list = [float(i) for i in value_list]
      tup = (min(value_list), mean(value_list), max(value_list))
    fvalues.append(tup)
  # End CODE
  return fvalues

# Problem D [0 points]
def outcome_values(dataset_id, dataset):
  values = set()
  # Begin CODE
  row, col = dataset.shape
  outcome = dataset[:, col - 1]
  values = set(outcome)
  # End CODE
  return values

# Problem E [0 points]
def outcomes(dataset_id, instances):
  outcomes = []
  # Begin CODE
  splited_set = np.array(instances)
  outcomes = splited_set[:, -1].tolist()
  # End CODE
  return outcomes

# Problem 1 [10 points]
def data_split(dataset_id, dataset, percent_train):
  split = None
  # Begin CODE
  row, col = dataset.shape
  train_row = int(row * percent_train)
  train = dataset[:train_row, :]
  test = dataset[train_row:, :]
  split = (train.tolist(), test.tolist())
  # End CODE
  return split

# Problem 2 [10 points]
def baseline(dataset_id, dataset):
  baseline = None
  # Begin CODE
  outcome_np = dataset[:, -1]
  (value, count) = np.unique(outcome_np, return_counts=True)
  baseline = str(value[count.argmax()])
  # End CODE
  return baseline

def encode_features(dataset_id, x):
  x_encode = x
  if dataset_id == 'breast_cancer':
    x_encode = x.astype(float)
  elif dataset_id == 'hyperthyroidism':
    x[x == '?'] = float('nan')
    x[x == 'f'] = 0
    x[x == 't'] = 1
    x[x == 'F'] = 0
    x[x == 'M'] = 1
    le = preprocessing.LabelEncoder()
    x[:, 28] = le.fit_transform(x[:, 28])
    x_encode = x
  elif dataset_id == 'cervical_cancer':
    x[x == '?'] = float('nan')
    x_encode = x.astype(float)
  elif dataset_id == 'liver_cancer':
    x[x == ''] = float('nan')
    x[x == 'Female'] = 0
    x[x == 'Male'] = 1
    x_encode = x.astype(float)

  return x_encode

# Problem 3 [15 points]
def decision_tree(dataset_id, train, test):
  predictions = []
  # Begin CODE
  train = np.array(train)
  test = np.array(test)

  train_row, train_col = train.shape
  X_train = train[:, :train_col-1]
  y_train = train[:, train_col-1]
  test_row, test_col = test.shape
  X_test = test[:, :test_col - 1]
  y_test = test[:, test_col - 1]

  X_train = encode_features(dataset_id, X_train)
  X_test = encode_features(dataset_id, X_test)
  imputer = SimpleImputer(missing_values=float('nan'), strategy='mean')
  imputer.fit(X_train)
  X_train = imputer.transform(X_train)
  X_test = imputer.transform(X_test)

  le = preprocessing.LabelEncoder()
  y_train_le = le.fit_transform(y_train)
  y_test_le = le.fit_transform(y_test)

  model = DecisionTreeClassifier(max_depth=5, random_state=1)
  model.fit(X_train, y_train_le)
  y_predict = model.predict(X_test)
  predictions = le.inverse_transform(y_predict).tolist()
  # print(predictions)
  # End CODE
  return predictions

# Problem 4 [15 points]
def knn(dataset_id, train, test):
  predictions = []
  # Begin CODE
  train = np.array(train)
  test = np.array(test)

  train_row, train_col = train.shape
  X_train = train[:, :train_col - 1]
  y_train = train[:, train_col - 1]
  test_row, test_col = test.shape
  X_test = test[:, :test_col - 1]
  y_test = test[:, test_col - 1]

  X_train = encode_features(dataset_id, X_train)
  X_test = encode_features(dataset_id, X_test)
  imputer = SimpleImputer(missing_values=float('nan'), strategy='mean')
  imputer.fit(X_train)
  X_train = imputer.transform(X_train)
  X_test = imputer.transform(X_test)

  le = preprocessing.LabelEncoder()
  y_train_le = le.fit_transform(y_train)
  y_test_le = le.fit_transform(y_test)

  model = KNeighborsClassifier(n_neighbors=3)
  model.fit(X_train, y_train_le)
  y_predict = model.predict(X_test)
  predictions = le.inverse_transform(y_predict).tolist()
  # print(predictions)
  # End CODE
  return predictions

# Problem 5 [15 points]
def naive_bayes(dataset_id, train, test):
  predictions = []
  # Begin CODE
  train = np.array(train)
  test = np.array(test)

  train_row, train_col = train.shape
  X_train = train[:, :train_col - 1]
  y_train = train[:, train_col - 1]
  test_row, test_col = test.shape
  X_test = test[:, :test_col - 1]
  y_test = test[:, test_col - 1]

  X_train = encode_features(dataset_id, X_train)
  X_test = encode_features(dataset_id, X_test)
  imputer = SimpleImputer(missing_values=float('nan'), strategy='mean')
  imputer.fit(X_train)
  X_train = imputer.transform(X_train)
  X_test = imputer.transform(X_test)

  le = preprocessing.LabelEncoder()
  y_train_le = le.fit_transform(y_train)
  y_test_le = le.fit_transform(y_test)

  model = GaussianNB()
  model.fit(X_train, y_train_le)
  y_predict = model.predict(X_test)
  predictions = le.inverse_transform(y_predict).tolist()
  # print(predictions)
  # End CODE
  return predictions

# Problem 6 [15 points]
def svm(dataset_id, train, test):
  predictions = []
  # Begin CODE
  train = np.array(train)
  test = np.array(test)

  train_row, train_col = train.shape
  X_train = train[:, :train_col - 1]
  y_train = train[:, train_col - 1]
  test_row, test_col = test.shape
  X_test = test[:, :test_col - 1]
  y_test = test[:, test_col - 1]

  X_train = encode_features(dataset_id, X_train)
  X_test = encode_features(dataset_id, X_test)
  imputer = SimpleImputer(missing_values=float('nan'), strategy='mean')
  imputer.fit(X_train)
  X_train = imputer.transform(X_train)
  X_test = imputer.transform(X_test)

  le = preprocessing.LabelEncoder()
  y_train_le = le.fit_transform(y_train)
  y_test_le = le.fit_transform(y_test)

  model = SVC(C=1.0, kernel='rbf', gamma=2.0, random_state=1)
  model.fit(X_train, y_train_le)
  y_predict = model.predict(X_test)
  predictions = le.inverse_transform(y_predict).tolist()
  # print(predictions)
  # End CODE
  return predictions

# Problem 7 [10 points]
def evaluate(dataset_id, gold, predictions):
  evaluation = {}
  # Begin CODE
  label_list = sorted(set(gold))
  accuracy = {'accuracy': metrics.accuracy_score(gold, predictions).tolist()}
  prf = metrics.precision_recall_fscore_support(gold, predictions, average=None, labels=label_list, zero_division=0)

  label_dict = {}
  for i in range(len(label_list)):
    prf_dict = {'precision': prf[0][i].tolist(), 'recall': prf[1][i].tolist(), 'f1': prf[2][i].tolist()}
    label_prf = {label_list[i]: prf_dict}
    label_dict.update(label_prf)

  evaluation = accuracy.copy()
  evaluation.update(label_dict)
  # End CODE
  return evaluation

# Problem 8 [10 points]
def learning_curve(dataset_id, train_sets, test, class_func):
  accuracies = []
  # Begin CODE
  for train in train_sets:
    predictions = class_func(dataset_id, train, test)
    test = np.array(test)
    test_row, test_col = test.shape
    y_test = test[:, test_col - 1]
    accuracy = metrics.accuracy_score(y_test, np.array(predictions)).tolist()
    accuracies.append(accuracy)
  # End CODE
  return accuracies

# Problem 9 [10 points extra]
def visualize(dataset_id, train_sets, test, class_func):
  # Begin CODE
  stats = learning_curve(dataset_id, train_sets, test, class_func)
  train_sizes = []
  for i in train_sets:
    train_sizes.append(len(i))
  fig = plt.figure()
  chart = plt.plot(train_sizes, stats, 'o-', label=dataset_id)
  # plt.axis([0, 1.1, 0, 1.1])
  plt.grid(True, linestyle='--', axis='y', color='grey')
  plt.xlabel('Percentage of training data used', fontweight='bold')
  plt.ylabel('Accuracy', fontweight='bold')
  plt.title('Learning Curve')
  plt.legend(loc='lower right')
  plt.show()
  # End CODE
  pass

# Note: don't mess with this code block!  Your code will be tested by an outside
# program that will not call this __main__ block.  So if you mess with the
# following block of code you might crash the autograder.  You're definitely
# encouraged to look at this code, however, especially if your code crashes.
if __name__ == '__main__':

  datasets = ['breast_cancer',\
              'hyperthyroidism',\
              'cervical_cancer',\
              'liver_cancer']
  dims =    {'breast_cancer':(569, 30),
             'hyperthyroidism':(3772, 29),
             'cervical_cancer':(858, 28),
             'liver_cancer':(583,10)}
  targets = {'breast_cancer':set(['B', 'M']),
             'hyperthyroidism':set(['goitre', 'secondary toxic', 'negative', 'T3 toxic', 'hyperthyroid']),
             'cervical_cancer':set(['0', '1']),
             'liver_cancer':set(['1', '2'])}

  for dataset_id in datasets:
    print('::  DATASET: %s ::' % dataset_id)
    print('::: Problem 0-A :::')
    data = read_data(dataset_id)
    util_5353.assert_not_none(data, '0-A')

    print('::: Problem 0-B :::')
    b_ret = dimensions(dataset_id, data)
    util_5353.assert_tuple(b_ret, 2, '0-B')
    util_5353.assert_int(b_ret[0], '0-B')
    util_5353.assert_int(b_ret[1], '0-B')
    util_5353.assert_int_eq(dims[dataset_id][0], b_ret[0], '0-B')
    util_5353.assert_int_eq(dims[dataset_id][1], b_ret[1], '0-B')

    print('::: Problem 0-C :::')
    c_ret = feature_values(dataset_id, data)
    util_5353.assert_list(c_ret, dims[dataset_id][1], '0-C')
    for i in range(len(c_ret)):
      if type(c_ret[i]) == set:
        for item in c_ret[i]:
          util_5353.assert_str(item, '0-C')
      else:
        util_5353.assert_tuple(c_ret[i], 3, '0-C')
        util_5353.assert_float(c_ret[i][0], '0-C')
        util_5353.assert_float(c_ret[i][1], '0-C')
        util_5353.assert_float(c_ret[i][2], '0-C')
    if dataset_id == 'breast_cancer':
      util_5353.assert_float_range((6.980, 6.982), c_ret[0][0], '0-C')
      util_5353.assert_float_range((14.12, 14.13), c_ret[0][1], '0-C')
      util_5353.assert_float_range((28.10, 28.12), c_ret[0][2], '0-C')
      util_5353.assert_float_range((143.4, 143.6), c_ret[3][0], '0-C')
      util_5353.assert_float_range((654.8, 654.9), c_ret[3][1], '0-C')
      util_5353.assert_float_range((2500., 2502.), c_ret[3][2], '0-C')

    print('::: Problem 0-D :::')
    d_ret = outcome_values(dataset_id, data)
    util_5353.assert_set(d_ret, '0-D', valid_values=targets[dataset_id])

    print('::: Problem 1 :::')
    one_ret = data_split(dataset_id, data, 0.6)
    util_5353.assert_tuple(one_ret, 2, '1')
    util_5353.assert_list(one_ret[0], None, '1')
    util_5353.assert_list(one_ret[1], None, '1')
    if dataset_id == 'breast_cancer':
      util_5353.assert_list(one_ret[0], 341, '1')
    if dataset_id == 'cervical_cancer':
      util_5353.assert_list(one_ret[0], 514, '1')
    train = one_ret[0]
    test  = one_ret[1]

    print('::: Problem 0-E :::')
    train_out = outcomes(dataset_id, train)
    test_out  = outcomes(dataset_id, test)
    util_5353.assert_list(train_out, len(train), '0-E', valid_values=targets[dataset_id])
    util_5353.assert_list(test_out,  len(test),  '0-E', valid_values=targets[dataset_id])
    if dataset_id == 'breast_cancer':
      util_5353.assert_str_eq('M', train_out[0], '0-E')
      util_5353.assert_str_eq('B', test_out[-1], '0-E')

    print('::: Problem 2 :::')
    two_ret = baseline(dataset_id, data)
    util_5353.assert_str(two_ret, '2')
    if dataset_id == 'breast_cancer':
      util_5353.assert_str_eq('B', two_ret, '2')

    print('::: Problem 3 :::')
    three_ret = decision_tree(dataset_id, train, test)
    util_5353.assert_list(three_ret, len(test), '3')

    print('::: Problem 4 :::')
    four_ret = knn(dataset_id, train, test)
    util_5353.assert_list(four_ret, len(test), '4')

    print('::: Problem 5 :::')
    five_ret = naive_bayes(dataset_id, train, test)
    util_5353.assert_list(five_ret, len(test), '5')

    print('::: Problem 6 :::')
    six_ret = svm(dataset_id, train, test)
    util_5353.assert_list(six_ret, len(test), '6')

    print('::: Problem 7 :::')
    seven_ret_dt = evaluate(dataset_id, test_out, three_ret)
    seven_ret_kn = evaluate(dataset_id, test_out, four_ret)
    seven_ret_nb = evaluate(dataset_id, test_out, five_ret)
    seven_ret_sv = evaluate(dataset_id, test_out, six_ret)

    for seven_ret in [seven_ret_dt, seven_ret_kn, seven_ret_nb, seven_ret_sv]:
      util_5353.assert_dict(seven_ret, '7')
      util_5353.assert_dict_key(seven_ret, 'accuracy', '7')
      util_5353.assert_float(seven_ret['accuracy'], '7')
      util_5353.assert_float_range((0.0, 1.0), seven_ret['accuracy'], '7')
      for target in targets[dataset_id]:
        util_5353.assert_dict_key(seven_ret, target, '7')
        util_5353.assert_dict(seven_ret[target], '7')
        util_5353.assert_dict_key(seven_ret[target], 'precision', '7')
        util_5353.assert_dict_key(seven_ret[target], 'recall', '7')
        util_5353.assert_dict_key(seven_ret[target], 'f1', '7')
        util_5353.assert_float(seven_ret[target]['precision'], '7')
        util_5353.assert_float(seven_ret[target]['recall'], '7')
        util_5353.assert_float(seven_ret[target]['f1'], '7')
        util_5353.assert_float_range((0.0, 1.0), seven_ret[target]['precision'], '7')
        util_5353.assert_float_range((0.0, 1.0), seven_ret[target]['recall'], '7')
        util_5353.assert_float_range((0.0, 1.0), seven_ret[target]['f1'], '7')

    print('::: Problem 8 :::')
    train_sets = []
    for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
      train_sets.append(train[:int(percent*len(train))])
    eight_ret_dt = learning_curve(dataset_id, train_sets, test, decision_tree)
    eight_ret_kn = learning_curve(dataset_id, train_sets, test, knn)
    eight_ret_nb = learning_curve(dataset_id, train_sets, test, naive_bayes)
    eight_ret_sv = learning_curve(dataset_id, train_sets, test, svm)
    for eight_ret in [eight_ret_dt, eight_ret_kn, eight_ret_nb, eight_ret_sv]:
      util_5353.assert_list(eight_ret, len(train_sets), '8')
      for i in range(len(eight_ret)):
        util_5353.assert_float(eight_ret[i], '8')
        util_5353.assert_float_range((0.0, 1.0), eight_ret[i], '8')

    # print('::: Problem 9 :::')
    # visualize(dataset_id, train_sets, test, decision_tree)
  print('~~~ All Tests Pass ~~~')



