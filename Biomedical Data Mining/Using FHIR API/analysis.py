import gzip
import json
import os
import hashlib

# Python 2
# import urllib2

# Python 3
import urllib

import util_5353

import numpy as np
from datetime import date
from collections import Counter
import statistics

BASE_URL = 'https://syntheticmass.mitre.org/fhir/'
MAX_PATIENTS = 50000
# MAX_PATIENTS = 1000
CACHE_FILE = 'cache.dat'
PATH_CACHE = {}

# Returns the JSON result at the given URL.  Caches the results so we don't
# unnecessarily hit the FHIR server.  Note this ain't the best caching, as
# it's going to save a bunch of tiny files that could probably be handled more
# efficiently.
def get_url(url):
  # First check the cache
  if len(PATH_CACHE) == 0:
    for line in open(CACHE_FILE).readlines():
      split = line.strip().split('\t')
      cached_path = split[0]
      cached_url = split[1]
      PATH_CACHE[cached_url] = cached_path
  if url in PATH_CACHE:
    return json.loads(gzip.open(PATH_CACHE[url]).read().decode('utf-8'))

  print('Retrieving:', url)

  print('You are about to query the FHIR server, which probably means ' + \
        'that you are doing something wrong.  But feel free to comment ' + \
        'out this bit of code and proceed right ahead.')
  exit(1)
  print('Note: the code below is not tested for Python 3, you will likely ' + \
        'need to make a few changes, e.g., urllib2')

  resultstr = urllib.urlopen(url).read()
  json_result = json.loads(resultstr)

  # Remove patient photos, too much space
  if url.replace(BASE_URL, '').startswith('Patient'):
    for item in json_result['entry']:
      item['resource']['photo'] = 'REMOVED'

  m = hashlib.md5()
  m.update(url)
  md5sum = m.hexdigest()

  path_dir = 'cache/' + md5sum[0:2] + '/' + md5sum[2:4] + '/'
  if not os.path.exists('cache'):
    os.mkdir('cache')
  if not os.path.exists('cache/' + md5sum[0:2]):
    os.mkdir('cache/' + md5sum[0:2])
  if not os.path.exists(path_dir):
    os.mkdir(path_dir)
  path = path_dir + url.replace(BASE_URL, '')

  w = gzip.open(path, 'wb')
  w.write(json.dumps(json_result))
  w.close()
  w = open(CACHE_FILE, 'a')
  w.write(path + '\t' + url + '\n')
  w.close()
  PATH_CACHE[url] = path

  return json_result

# For pagination, returns the next URL
def get_next(result):
  links = result['link']
  for link in links:
    if link['relation'] == 'next':
      return link['url']

# Returns the list of patients based on the given filter
get_patients_count = 0
def get_patients(pt_filter):
  # Helpful logging to point out some programming flaws
  global get_patients_count
  get_patients_count += 1
  if get_patients_count >= 10:
    print('WARNING: get_patients called too many times')

  patients = []
  url = BASE_URL + 'Patient?_offset=0&_count=1000'
  while url is not None:
    patients_page = get_url(url)
    if 'entry' not in patients_page:
      break
    for patient_json in patients_page['entry']:
      patients.append(patient_json['resource'])
      if MAX_PATIENTS is not None and len(patients) == MAX_PATIENTS:
        return [p for p in patients if pt_filter.include(p)]
    url = get_next(patients_page)
  return [p for p in patients if pt_filter.include(p)]

# Returns the conditions for the patient with the given patient_id
get_conditions_count = 0
def get_conditions(patient_id):
  global get_conditions_count
  get_conditions_count += 1
  if get_conditions_count >= MAX_PATIENTS * 5:
    print('WARNING: get_conditions called too many times')

  url = BASE_URL + 'Condition?patient=' + patient_id + '&_offset=0&_count=1000'
  conditions = []
  while url is not None:
    conditions_page = get_url(url)
    if 'entry' in conditions_page:
      conditions.extend([c['resource'] for c in conditions_page['entry']])
    url = get_next(conditions_page)
  return conditions

# Returns the observations for the patient with the given patient_id
get_observations_count = 0
def get_observations(patient_id):
  # Helpful logging to point out some programming flaws
  global get_observations_count
  get_observations_count += 1
  if get_observations_count >= MAX_PATIENTS * 3:
    print('WARNING: get_observations called too many times')

  url = BASE_URL + 'Observation?patient=' + patient_id + '&_offset=0&_count=1000'
  observations = []
  while url is not None:
    observations_page = get_url(url)
    if 'entry' in observations_page:
      observations.extend([o['resource'] for o in observations_page['entry']])
    url = get_next(observations_page)
  return observations

# Returns the medications for the patient with the given patient_id
get_medications_count = 0
def get_medications(patient_id):
  # Helpful logging to point out some programming flaws
  global get_medications_count
  get_medications_count += 1
  if get_medications_count >= MAX_PATIENTS * 5:
    print('WARNING: get_medications called too many times')

  url = BASE_URL + 'MedicationRequest?patient=' + patient_id + '&_offset=0&_count=1000'
  medications = []
  DBG = 0
  while url is not None:
    medications_page = get_url(url)
    if 'entry' in medications_page:
      medications.extend([c['resource'] for c in medications_page['entry']])
    url = get_next(medications_page)
  return medications

# Problem 1 [10 points]
def num_patients(pt_filter):
  tup = None
  # Begin CODE
  # Count number of patients
  patient_list = get_patients(pt_filter)
  count = len(patient_list)

  # Parse surnames
  familyname_list = []
  for patient in patient_list:
    name = patient['name']
    familyname = name[0]['family']
    familyname_list.append(familyname)

  # Remove numbers
  surname_list = []
  for familyname in familyname_list:
    surname = ''.join(i for i in familyname if not i.isdigit())
    surname_list.append(surname)
  surname_list = np.array(surname_list)
  surname_count = len(np.unique(surname_list))

  tup = (count, surname_count)
  # End CODE
  return tup


# Problem 2 [10 points]
def patient_stats(pt_filter):
  stats = {}
  # Begin CODE
  patient_list = get_patients(pt_filter)
  patient_num = len(patient_list)
  stats_keys = ['gender', 'marital_status', 'race', 'ethnicity', 'age', 'with_address']
  # print(patient_list[0])

  # Parse gender
  gender_dict = {}
  gender_list = []
  for patient in patient_list:
    gender = patient['gender']
    gender_list.append(gender)
  gender_list = np.array(gender_list)
  value, count = np.unique(gender_list, return_counts=True)
  count = count / patient_num
  gender_dict = dict(zip(value.tolist(), count.tolist()))
  # print(gender_dict)

  # Parse maritalStatus
  maritalStatus_dict = {}
  maritalStatus_list = []
  for patient in patient_list:
    if 'maritalStatus' in patient.keys():
      maritalStatus = patient['maritalStatus']['coding'][0]['code']
    else:
      maritalStatus = 'UNK'
    maritalStatus_list.append(maritalStatus)
  maritalStatus_list = np.array(maritalStatus_list)
  value, count = np.unique(maritalStatus_list, return_counts=True)
  count = count / patient_num
  maritalStatus_dict = dict(zip(value.tolist(), count.tolist()))
  # print(maritalStatus_dict)

  # Parse race
  race_dict = {}
  race_list = []
  for patient in patient_list:
    race = patient['extension'][0]['valueCodeableConcept']['coding'][0]['display']
    race_list.append(race)
  race_list = np.array(race_list)
  value, count = np.unique(race_list, return_counts=True)
  count = count / patient_num
  race_dict = dict(zip(value.tolist(), count.tolist()))
  # print(race_dict)

  # Parse ethnicity
  ethnicity_dict = {}
  ethnicity_list = []
  for patient in patient_list:
    ethnicity = patient['extension'][1]['valueCodeableConcept']['coding'][0]['display']
    ethnicity_list.append(ethnicity)
  ethnicity_list = np.array(ethnicity_list)
  value, count = np.unique(ethnicity_list, return_counts=True)
  count = count / patient_num
  ethnicity_dict = dict(zip(value.tolist(), count.tolist()))
  # print(ethnicity_dict)

  # Parse birth year
  year_dict = {}
  year_list = []
  for patient in patient_list:
    birth_year = date.fromisoformat(patient['birthDate']).year
    year_list.append(birth_year)

  year_group = np.arange(1900, 2021, 10)
  hist = np.histogram(year_list, year_group)
  year_values = hist[0]/patient_num
  year_keys = ['1900', '1910', '1920', '1930', '1940', '1950', '1960',
               '1970', '1980', '1990', '2000', '2010']
  year_dict = dict(zip(year_keys, year_values.tolist()))
  # print(year_dict)

  # Parse address
  address_dict = {}
  address_list = []
  for patient in patient_list:
    if 'address' in patient.keys():
      address = 'yes_address'
    else:
      address = 'no_address'
    address_list.append(address)
  address_list = np.array(address_list)
  value, count = np.unique(address_list, return_counts=True)
  count = count / patient_num
  address_dict = dict(zip(value.tolist(), count.tolist()))
  stats = dict(zip(stats_keys, [gender_dict, maritalStatus_dict, race_dict,
                                ethnicity_dict, year_dict, address_dict]))
  # End CODE
  return stats

# Problem 3 [15 points]
def diabetes_quality_measure(pt_filter):
  tup = None
  # Begin CODE
  diabetes_id_list = []
  patient_list = get_patients(pt_filter)
  for patient in patient_list:
    id = patient['id']
    condition_list = get_conditions(id)
    if condition_list:
      code_list = []
      for condition in condition_list:
        code_list.append(condition['code']['coding'][0]['code'])
      if '44054006' in code_list:
        diabetes_id_list.append(id)

  hemoglobin_id_list = []
  for id in diabetes_id_list:
    observation_list = get_observations(id)
    if observation_list:
      test_list = []
      for observation in observation_list:
        test_list.append(observation['code']['coding'][0]['code'])
      if test_list.count('4548-4') > 0:
        hemoglobin_id_list.append(id)

  value_id_list = []
  for id in hemoglobin_id_list:
    observation_list = get_observations(id)
    value_list = []
    for observation in observation_list:
      if observation['code']['coding'][0]['code'] == '4548-4':
        value_list.append(observation['valueQuantity']['value'])
    if any(value > 6.0 for value in value_list):
      value_id_list.append(id)

  tup = (len(diabetes_id_list),len(hemoglobin_id_list), len(value_id_list))
  # print(tup)
  # End CODE
  return tup

# Problem 4 [10 points]
def common_condition_pairs(pt_filter):
  pairs = []
  # Begin CODE
  patient_list = get_patients(pt_filter)
  pt_condition_list = []
  for patient in patient_list:
    id = patient['id']
    condition_list = get_conditions(id)
    if condition_list:
      co_condition = []
      for condition in condition_list:
        if condition['clinicalStatus'] == 'active':
          co_condition.append(condition['code']['coding'][0]['display'])
      temp = list(set(co_condition))
      pt_condition = sorted(temp)
      pt_condition_list.append(pt_condition)

  condition_pair_list = []
  for pt_condition in pt_condition_list:
    for i in range(len(pt_condition)):
      for j in range(i+1, len(pt_condition)):
        pair = pt_condition[i] + '-' + pt_condition[j]
        condition_pair_list.append(pair)

  np_list = np.array(condition_pair_list)
  value, count = np.unique(np_list, return_counts=True)
  condition_pair_dict = dict(zip(value.tolist(), count.tolist()))

  def keyfunction(k):
    return condition_pair_dict[k]
  for key in sorted(condition_pair_dict, key=keyfunction, reverse=True)[:10]:
    top_pair = tuple(key.split('-'))
    pairs.append(top_pair)
  # End CODE
  return pairs

# Problem 5 [10 points]
def common_medication_pairs(pt_filter):
  pairs = []
  # Begin CODE
  patient_list = get_patients(pt_filter)
  pt_medication_list = []
  for patient in patient_list:
    id = patient['id']
    medication_list = get_medications(id)
    if medication_list:
      co_medication = []
      for medication in medication_list:
        if medication['status'] == 'active':
          co_medication.append(medication['medicationCodeableConcept']['coding'][0]['display'])
      temp = list(set(co_medication))
      pt_medication = sorted(temp)
      pt_medication_list.append(pt_medication)

  medication_pair_list = []
  for pt_medication in pt_medication_list:
    if len(pt_medication) > 1:
      for i in range(len(pt_medication)):
        for j in range(i + 1, len(pt_medication)):
          pair = pt_medication[i] + '#' + pt_medication[j]
          medication_pair_list.append(pair)

  np_list = np.array(medication_pair_list)
  value, count = np.unique(np_list, return_counts=True)
  medication_pair_dict = dict(zip(value.tolist(), count.tolist()))

  def keyfunction(k):
    return medication_pair_dict[k]
  for key in sorted(medication_pair_dict, key=keyfunction, reverse=True)[:10]:
    top_pair = tuple(key.split('#'))
    pairs.append(top_pair)
  # End CODE
  return pairs

# Problem 6 [10 points]
def conditions_by_age(pt_filter):
  tup = None
  # Begin CODE
  patient_list = get_patients(pt_filter)
  over_50_id = []
  under_15_id = []
  for patient in patient_list:
    if patient['birthDate']:
      birth_date = date.fromisoformat(patient['birthDate'])
      if birth_date <= date(1968, 1, 31):
        # print(birth_date)
        over_50_id.append(patient['id'])
      elif birth_date >= date(2003, 2, 1):
        under_15_id.append(patient['id'])

  id_tuple = (over_50_id, under_15_id)
  non_inflam_tuple = ([], [])
  for i in range(len(id_tuple)):
    for id in id_tuple[i]:
      if get_conditions(id):
        condition_list = get_conditions(id)
        non_inflam = []
        display_list = []
        for condition in condition_list:
          if condition['clinicalStatus'] == 'active':
            display_list.append(condition['code']['coding'][0]['display'])
        display_list = list(set(display_list))
        for display in display_list:
          if not 'itis' in display:
            non_inflam.append(display)
        for non_inflam_condition in non_inflam:
          non_inflam_tuple[i].append(non_inflam_condition)

  tup = ([], [])
  for i in range(len(non_inflam_tuple)):
    c = Counter(non_inflam_tuple[i])
    top_ten = c.most_common(10)
    for j in range(len(top_ten)):
      tup[i].append(top_ten[j][0])
  # print(tup)
  # End CODE
  return tup

# Problem 7 [10 points]
def medications_by_gender(pt_filter):
  tup = None
  # Begin CODE
  patient_list = get_patients(pt_filter)
  male = []
  female = []

  for patient in patient_list:
    gender = patient['gender']
    if gender == 'male':
      male.append(patient['id'])
    elif gender == 'female':
      female.append(patient['id'])
    else:
      print("No gender recorded.")

  id_tuple = (male, female)
  medication_tuple = ([], [])
  for i in range(len(id_tuple)):
    for id in id_tuple[i]:
      medication_list = get_medications(id)
      active = []
      if medication_list:
        for medication in medication_list:
          if medication['status'] == 'active':
            active.append(medication['medicationCodeableConcept']['coding'][0]['display'])
      active = list(set(active))
      for active_medication in active:
        medication_tuple[i].append(active_medication)

  tup = ([], [])
  for i in range(len(medication_tuple)):
    c = Counter(medication_tuple[i])
    top_ten = c.most_common(10)
    for j in range(len(top_ten)):
      tup[i].append(top_ten[j][0])
  # print(tup)
  # End CODE
  return tup

# Problem 8 [25 points]
def bp_stats(pt_filter):
  stats = []
  # Begin CODE
  patient_list = get_patients(pt_filter)
  id_tuple = ([], [], [])
  for patient in patient_list:
    if get_observations(patient['id']):
      observation_list = get_observations(patient['id'])
      code_list = []
      for observation in observation_list:
        code_list.append(observation['code']['coding'][0]['code'])
      if '55284-4' in code_list:
        bp_list = []
        for observation in observation_list:
          if observation['code']['coding'][0]['code'] == '55284-4':
            bp_list.append(observation['component'])

        normal_num = 0
        for test in bp_list:
          if (90 <= test[0]['valueQuantity']['value'] <= 140) and \
                  (60 <= test[1]['valueQuantity']['value'] <= 90):
            normal_num += 1
        if (normal_num / len(bp_list)) >= 0.9:
          id_tuple[0].append(patient['id'])
        else:
          id_tuple[1].append(patient['id'])
      else:
        id_tuple[2].append(patient['id'])
    else:
      id_tuple[2].append(patient['id'])

  stats = [{}, {}, {}]
  keys = ['min', 'max', 'mean', 'median', 'stddev']
  for i in range(len(id_tuple)):
    num_list = []
    values = []
    for id in id_tuple[i]:
      condition_list = get_conditions(id)
      code_list = []
      if condition_list:
        for condition in condition_list:
          code = condition['code']['coding'][0]['code']
          code_list.append(code)
        code_list = list(set(code_list))
      num_list.append(len(code_list))

    values = [min(num_list), max(num_list), statistics.mean(num_list),
              statistics.median(num_list), statistics.stdev(num_list)]
    stats[i] = dict(zip(keys, values))
  # print(stats)
  # End CODE
  return stats


# Basic filter, lets everything pass
class all_pass_filter:
  def id(self):
    return 'all_pass'
  def include(self, patient):
    util_5353.assert_dict_key(patient, 'id', 'pt_filter')
    util_5353.assert_dict_key(patient, 'name', 'pt_filter')
    util_5353.assert_dict_key(patient, 'address', 'pt_filter')
    util_5353.assert_dict_key(patient, 'birthDate', 'pt_filter')
    util_5353.assert_dict_key(patient, 'gender', 'pt_filter')
    return True

# Note: don't mess with this code block!  Your code will be tested by an outside
# program that will not call this __main__ block.  So if you mess with the
# following block of code you might crash the autograder.  You're definitely
# encouraged to look at this code, however, especially if your code crashes.
if __name__ == '__main__':

  # Include all patients
  pt_filter = all_pass_filter()

  print('::: Problem 1 :::')
  one_ret = num_patients(pt_filter)
  util_5353.assert_tuple(one_ret, 2, '1')
  util_5353.assert_int_range((0, 10000000), one_ret[0], '1')
  util_5353.assert_int_range((0, 10000000), one_ret[1], '1')

  print('::: Problem 2 :::')
  two_ret = patient_stats(pt_filter)
  util_5353.assert_dict(two_ret, '2')
  for key in ['gender', 'marital_status', 'race', 'ethnicity', 'age', 'with_address']:
    util_5353.assert_dict_key(two_ret, key, '2')
    util_5353.assert_dict(two_ret[key], '2')
    for key2 in two_ret[key].keys():
      util_5353.assert_str(key2, '2')
    util_5353.assert_prob_dict(two_ret[key], '2')
  for key2 in two_ret['age'].keys():
    if not key2.isdigit():
      util_5353.die('2', 'age key should be year: %s', key2)

  print('::: Problem 3 :::')
  three_ret = diabetes_quality_measure(pt_filter)
  util_5353.assert_tuple(three_ret, 3, '3')
  util_5353.assert_int_range((0, 1000000), three_ret[0], '3')
  util_5353.assert_int_range((0, 1000000), three_ret[1], '3')
  util_5353.assert_int_range((0, 1000000), three_ret[2], '3')
  if three_ret[0] < three_ret[1] or three_ret[1] < three_ret[2]:
    util_5353.die('3', 'Values should be in %d >= %d >= %d', three_ret)

  print('::: Problem 4 :::')
  four_ret = common_condition_pairs(pt_filter)
  util_5353.assert_list(four_ret, 10, '4')
  for i in range(len(four_ret)):
    util_5353.assert_tuple(four_ret[i], 2, '4')
    util_5353.assert_str(four_ret[i][0], '4')
    util_5353.assert_str(four_ret[i][1], '4')

  print('::: Problem 5 :::')
  five_ret = common_medication_pairs(pt_filter)
  util_5353.assert_list(five_ret, 10, '5')
  for i in range(len(five_ret)):
    util_5353.assert_tuple(five_ret[i], 2, '5')
    util_5353.assert_str(five_ret[i][0], '5')
    util_5353.assert_str(five_ret[i][1], '5')

  print('::: Problem 6 :::')
  six_ret = conditions_by_age(pt_filter)
  util_5353.assert_tuple(six_ret, 2, '6')
  util_5353.assert_list(six_ret[0], 10, '6')
  util_5353.assert_list(six_ret[1], 10, '6')
  for i in range(len(six_ret[0])):
    util_5353.assert_str(six_ret[0][i], '6')
    util_5353.assert_str(six_ret[1][i], '6')

  print('::: Problem 7 :::')
  seven_ret = medications_by_gender(pt_filter)
  util_5353.assert_tuple(seven_ret, 2, '6')
  util_5353.assert_list(seven_ret[0], 10, '6')
  util_5353.assert_list(seven_ret[1], 10, '6')
  for i in range(len(seven_ret[0])):
    util_5353.assert_str(seven_ret[0][i], '6')
    util_5353.assert_str(seven_ret[1][i], '6')

  print('::: Problem 8 :::')
  eight_ret = bp_stats(pt_filter)
  util_5353.assert_list(eight_ret, 3, '8')
  for i in range(len(eight_ret)):
    util_5353.assert_dict(eight_ret[i], '8')
    util_5353.assert_dict_key(eight_ret[i], 'min', '8')
    util_5353.assert_dict_key(eight_ret[i], 'max', '8')
    util_5353.assert_dict_key(eight_ret[i], 'median', '8')
    util_5353.assert_dict_key(eight_ret[i], 'mean', '8')
    util_5353.assert_dict_key(eight_ret[i], 'stddev', '8')

  print('~~~ All Tests Pass ~~~')


