
import util_5353

from lxml import etree
import statistics
import edit_distance
from collections import Counter
from itertools import groupby
from sklearn.linear_model import ElasticNetCV
import numpy as np

MEDICATIONS = ['remeron', 'lexapro', 'effexor', 'zoloft', 'celexa', 
               'wellbutrin', 'paxil', 'savella', 'prozac', 'cymbalta']
MUTATIONS = ['chrom_1.pos_98539.ref_A.alt_T',  'chrom_1.pos_88327.ref_C.alt_A',
             'chrom_1.pos_63872.ref_C.alt_T',  'chrom_1.pos_96696.ref_A.alt_G',
             'chrom_2.pos_97561.ref_G.alt_A',  'chrom_2.pos_69421.ref_A.alt_C',
             'chrom_2.pos_70704.ref_G.alt_A',  'chrom_2.pos_30517.ref_A.alt_C',
             'chrom_3.pos_57245.ref_G.alt_A',  'chrom_3.pos_64337.ref_T.alt_C',
             'chrom_3.pos_48160.ref_A.alt_C',  'chrom_3.pos_14811.ref_G.alt_A',
             'chrom_4.pos_99335.ref_T.alt_C',  'chrom_4.pos_49304.ref_G.alt_T',
             'chrom_4.pos_93162.ref_A.alt_T',  'chrom_4.pos_35883.ref_A.alt_G',
             'chrom_5.pos_99641.ref_T.alt_A',  'chrom_5.pos_47810.ref_T.alt_A',
             'chrom_5.pos_41351.ref_T.alt_C',  'chrom_5.pos_30106.ref_A.alt_C',
             'chrom_6.pos_95091.ref_C.alt_G',  'chrom_6.pos_22806.ref_C.alt_G',
             'chrom_6.pos_6035.ref_T.alt_A',   'chrom_6.pos_57950.ref_A.alt_G',
             'chrom_7.pos_66842.ref_C.alt_A',  'chrom_7.pos_40665.ref_C.alt_T',
             'chrom_7.pos_16241.ref_T.alt_A',  'chrom_7.pos_46163.ref_T.alt_A',
             'chrom_8.pos_93350.ref_A.alt_G',  'chrom_8.pos_73332.ref_T.alt_G',
             'chrom_8.pos_17571.ref_C.alt_A',  'chrom_8.pos_92636.ref_C.alt_G',
             'chrom_9.pos_99676.ref_G.alt_A',  'chrom_9.pos_14535.ref_C.alt_A',
             'chrom_9.pos_35056.ref_A.alt_G',  'chrom_9.pos_28381.ref_A.alt_G',
             'chrom_10.pos_54525.ref_C.alt_G', 'chrom_10.pos_87597.ref_T.alt_A',
             'chrom_10.pos_54127.ref_G.alt_T', 'chrom_10.pos_13058.ref_C.alt_A']

# Problem A [0 points]
def read_data(filename, attempt):
  data = None
  # Begin CODE
  data = []
  keys = ['id', 'baseline_hamd', 'Genome', 'Results']
  tree = etree.parse(filename)
  root = tree.getroot()
  for patient in root:
    id = patient.attrib['id']
    baseline_hamd = patient.attrib['baseline_hamd']
    mutation_list = []
    for mutation in patient[0]:
      mutation_row = [mutation.attrib['chromosome'], mutation.attrib['pos'],
                      mutation.attrib['ref'], mutation.attrib['alt']]
      mutation_list.append(mutation_row)
    # print(mutation_list)
    result_list = []
    for result in patient[1]:
        result_row = [result.attrib['date'], result.attrib['medication'], result.attrib['hamd']]
        result_list.append(result_row)
    # print(result_list)

    values = [id, baseline_hamd, mutation_list, result_list]
    patient_dict = dict(zip(keys, values))
    data.append(patient_dict)
  # print(data[1])
  # End CODE
  return data

# Problem 1 [10 points]
def mean_missed_reports(data, attempt):
  mean = None
  # Begin CODE
  missed_num_list = []
  for patient in data:
      missed_num = 26 - len(patient['Results'])
      missed_num_list.append(missed_num)
  mean = statistics.mean(missed_num_list)
  # End CODE
  return mean

# Problem 2 [10 points]
def total_medication_misspellings(data, attempt):
  total = 0
  # Begin CODE
  misspelled_med_list = []
  for patient in data:
    for result in patient['Results']:
      if not (result[1] == 'none' or result[1] in MEDICATIONS):
        misspelled_med_list.append(result[1])
  # print(misspelled_med_list)
  total = len(misspelled_med_list)
  # End CODE
  return total

# Problem 3 [10 points]
def medications_by_frequency(data, attempt):
  medications = []
  # Begin CODE
  for patient in data:
    for result in patient['Results']:
      if not (result[1] == 'none' or result[1] in MEDICATIONS):
        distance_list = []
        for i in range(len(MEDICATIONS)):
          sm = edit_distance.SequenceMatcher(a=result[1], b=MEDICATIONS[i])
          distance_list.append(sm.distance())
        match_index = distance_list.index(min(distance_list))
        result[1] = MEDICATIONS[match_index]

  true_med_list = []
  for patient in data:
    med_list = []
    for result in patient['Results']:
      med_list.append(result[1])
    for i, group in groupby(med_list):
      consecutive_c = [*group]
      if not consecutive_c[0] == 'none':
        if len(consecutive_c) >= 3:
          true_med_list.append(consecutive_c[0])
        else:
          true_med_list.extend(consecutive_c)
  medication_dict = dict(Counter(true_med_list))
  medications = sorted(medication_dict, key=medication_dict.get, reverse=True)
  # End CODE
  return medications

# Problem 4 [10 points]
def total_mutation_corruptions(data, attempt):
  total = 0
  # Begin CODE
  mis_mut_list = []
  for patient in data:
    if patient['Genome']:
      for mutation in patient['Genome']:
        mut_string = 'chrom_' + mutation[0] + '.pos_' + mutation[1] \
                     + '.ref_' + mutation[2] + '.alt_' + mutation[3]
        # print(mut_string)
        if not mut_string in MUTATIONS:
          mis_mut_list.append(mut_string)
  total = len(mis_mut_list)
  # End CODE
  return total

# Problem 5 [10 points]
def mutations_by_frequency(data, attempt):
  mutations = []
  # Begin CODE
  mut_string_list = []
  mut_string_dict = {}
  for patient in data:
    if patient['Genome']:
      for mutation in patient['Genome']:
        mut_string = 'chrom_' + mutation[0] + '.pos_' + mutation[1] \
                     + '.ref_' + mutation[2] + '.alt_' + mutation[3]
        if not mut_string in MUTATIONS:
          distance_list = []
          for i in range(len(MUTATIONS)):
            sm = edit_distance.SequenceMatcher(a=mut_string, b=MUTATIONS[i])
            distance_list.append(sm.distance())
          match_index = distance_list.index(min(distance_list))
          mut_string = MUTATIONS[match_index]
          split_string = mut_string.split('.')
          mutation[0] = split_string[0].replace('chrom_', '')
          mutation[1] = split_string[1].replace('pos_', '')
          mutation[2] = split_string[2].replace('ref_', '')
          mutation[3] = split_string[3].replace('alt_', '')
        mut_string_list.append(mut_string)
  mut_string_dict = dict(Counter(mut_string_list))
  mutations = sorted(mut_string_dict, key=mut_string_dict.get, reverse=True)
  # End CODE
  return mutations

# Problem 6 [20 points]
def mutation_impact(data, attempt):
  impact = {m:None for m in MUTATIONS}
  # Begin CODE
  x = []
  y = []
  for patient in data:
    y.append(int(patient['baseline_hamd']))
    mutation_dummy = np.zeros((40,), dtype=int)
    mutation_dummy = mutation_dummy.tolist()
    if patient['Genome']:
      for mutation in patient['Genome']:
        mut_string = 'chrom_' + mutation[0] + '.pos_' + mutation[1] \
                     + '.ref_' + mutation[2] + '.alt_' + mutation[3]
        index_num = MUTATIONS.index(mut_string)
        mutation_dummy[index_num] = 1
      # print(mutation_dummy)
      x.append(mutation_dummy)
    else:
      x.append(mutation_dummy)

  clf = ElasticNetCV(max_iter=200, cv=10, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
  clf.fit(x, y)
  coefficients = clf.coef_

  impact = dict(zip(MUTATIONS, coefficients.tolist()))
  # print(impact)
  # End CODE
  return impact

# Problem 7 [10 points]
def mutation_medication_impact(data, attempt):
  impact = {m:{med:None for med in MEDICATIONS} for m in MUTATIONS}
  # Begin CODE
  x = []
  y = []
  for patient in data:
    # for result in patient['Results']:
    for i in range(1, len(patient['Results'])):
      cur_res = patient['Results'][i]
      pre_res = patient['Results'][i-1]
      cur_med_dummy = np.zeros((10,), dtype=int).tolist()
      pre_med_dummy = np.zeros((10,), dtype=int).tolist()
      if not cur_res[1] == 'none':
        cur_med_index = MEDICATIONS.index(cur_res[1])
        cur_med_dummy[cur_med_index] = 1
      if not pre_res[1] == 'none':
        pre_med_index = MEDICATIONS.index(pre_res[1])
        pre_med_dummy[pre_med_index] = 1

      baseline = int(patient['baseline_hamd'])
      pair_dummy = np.zeros((400,), dtype=int).tolist()
      if not cur_res[1] == 'none':
        cur_med_index = MEDICATIONS.index(cur_res[1])
        if patient['Genome']:
          for mutation in patient['Genome']:
            mut_string = 'chrom_' + mutation[0] + '.pos_' + mutation[1] \
                         + '.ref_' + mutation[2] + '.alt_' + mutation[3]
            mut_index = MUTATIONS.index(mut_string)
            pair_index = cur_med_index * 40 + mut_index
            pair_dummy[pair_index] = 1

      variables = pair_dummy + cur_med_dummy + pre_med_dummy
      hamd = int(cur_res[2]) - baseline
      x.append(variables)
      y.append(hamd)

  clf = ElasticNetCV(max_iter=200, cv=10, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
  clf.fit(x, y)
  # print(clf.score(x, y))
  coefficients = clf.coef_

  pair_coef = coefficients[0:400]
  pair_coef_2d = np.array(pair_coef).reshape((10, 40))
  pair_coef_trans = np.transpose(pair_coef_2d)
  pair_coef_reverse = pair_coef_trans.flatten().tolist()

  pair_dict = {}
  for i in range(len(MUTATIONS)):
    med_dic = {}
    for j in range(len(MEDICATIONS)):
      med = {MEDICATIONS[j]: pair_coef_reverse[i*10+j]}
      med_dic.update(med)
    pair = {MUTATIONS[i]: med_dic}
    pair_dict.update(pair)
  impact = pair_dict
  # End CODE
  return impact

# Problem 8 [20 points]
def medication_impact(data, attempt):
  medications = {med:None for med in MEDICATIONS}
  # Begin CODE
  x = []
  y = []
  for patient in data:
    # for result in patient['Results']:
    for i in range(1, len(patient['Results'])):
      cur_res = patient['Results'][i]
      pre_res = patient['Results'][i - 1]
      cur_med_dummy = np.zeros((10,), dtype=int).tolist()
      pre_med_dummy = np.zeros((10,), dtype=int).tolist()
      if not cur_res[1] == 'none':
        cur_med_index = MEDICATIONS.index(cur_res[1])
        cur_med_dummy[cur_med_index] = 1
      if not pre_res[1] == 'none':
        pre_med_index = MEDICATIONS.index(pre_res[1])
        pre_med_dummy[pre_med_index] = 1

      baseline = int(patient['baseline_hamd'])
      pair_dummy = np.zeros((400,), dtype=int).tolist()
      if not cur_res[1] == 'none':
        cur_med_index = MEDICATIONS.index(cur_res[1])
        if patient['Genome']:
          for mutation in patient['Genome']:
            mut_string = 'chrom_' + mutation[0] + '.pos_' + mutation[1] \
                         + '.ref_' + mutation[2] + '.alt_' + mutation[3]
            mut_index = MUTATIONS.index(mut_string)
            pair_index = cur_med_index * 40 + mut_index
            pair_dummy[pair_index] = 1

      variables = pair_dummy + cur_med_dummy + pre_med_dummy
      hamd = int(cur_res[2]) - baseline
      x.append(variables)
      y.append(hamd)

  clf = ElasticNetCV(max_iter=200, cv=10, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
  clf.fit(x, y)
  # print(clf.score(x, y))
  coefficients = clf.coef_.tolist()
  med_coef = coefficients[400:410]
  med_dict = {}
  for i in range(len(MEDICATIONS)):
    med = {MEDICATIONS[i]: med_coef[i]}
    med_dict.update(med)
  medications = med_dict
  # End CODE
  return medications

# Note: don't mess with this code block!  Your code will be tested by an outside
# program that will not call this __main__ block.  So if you mess with the
# following block of code you might crash the autograder.  You're definitely
# encouraged to look at this code, however, especially if your code crashes.
if __name__ == '__main__':

  participants = 10000
  # participants = 1000
  attempts = [1, 2, 3]   # change to [1] or [1, 2] if appropriate
  
  for attempt in attempts:
    print('::  ATTEMPT %d ::' % attempt)
    print('::: Problem 0 :::')
    data = read_data('depression_study.' + str(participants) + '.xml', attempt)

    print('::: Problem 1 :::')
    one_ret = mean_missed_reports(data, attempt)
    util_5353.assert_float(one_ret, '1')
    util_5353.assert_float_range((0.01, 26.0), one_ret, '1')

    print('::: Problem 2 :::')
    two_ret = total_medication_misspellings(data, attempt)
    util_5353.assert_int(two_ret, '2')
    util_5353.assert_int_range((1, 26 * participants), two_ret, '2')

    print('::: Problem 3 :::')
    three_ret = medications_by_frequency(data, attempt)
    util_5353.assert_list(three_ret, len(MEDICATIONS), '3', valid_values=MEDICATIONS)

    print('::: Problem 4 :::')
    four_ret = total_mutation_corruptions(data, attempt)
    util_5353.assert_int(two_ret, '4')
    util_5353.assert_int_range((1, 40 * participants), two_ret, '4')

    print('::: Problem 5 :::')
    five_ret = mutations_by_frequency(data, attempt)
    util_5353.assert_list(five_ret, None, '5', valid_values=MUTATIONS)
    util_5353.assert_int_range((1, 40), len(five_ret), '5')
    util_5353.assert_int_eq(len(five_ret), len(set(five_ret)), '5')

    print('::: Problem 6 :::')
    six_ret = mutation_impact(data, attempt)
    util_5353.assert_dict(six_ret, '6')
    util_5353.assert_int_eq(len(MUTATIONS), len(six_ret), '6')
    for k,v in six_ret.items():
      util_5353.assert_str(k, '6', valid_values=MUTATIONS)
      util_5353.assert_float(v, '6')
      util_5353.assert_float_range((-10.0, 10.0), v, '6')

    print('::: Problem 7 :::')
    seven_ret = mutation_medication_impact(data, attempt)
    util_5353.assert_dict(seven_ret, '7')
    util_5353.assert_int_eq(len(MUTATIONS), len(seven_ret), '7')
    for k,v in seven_ret.items():
      util_5353.assert_str(k, '7', valid_values=MUTATIONS)
      util_5353.assert_int_eq(len(MEDICATIONS), len(v), '7')
      for k2,v2 in seven_ret[k].items():
        util_5353.assert_str(k2, '7', valid_values=MEDICATIONS)
        util_5353.assert_float(v2, '7')
        util_5353.assert_float_range((-10.0, 10.0), v2, '7')

    print('::: Problem 8 :::')
    eight_ret = medication_impact(data, attempt)
    util_5353.assert_dict(eight_ret, '8')
    util_5353.assert_int_eq(len(MEDICATIONS), len(eight_ret), '8')
    for k,v in eight_ret.items():
      util_5353.assert_str(k, '8', valid_values=MEDICATIONS)
      util_5353.assert_float(v, '8')
      util_5353.assert_float_range((-20.0, 10.0), v, '8')

  print('~~~ All Tests Pass ~~~')



