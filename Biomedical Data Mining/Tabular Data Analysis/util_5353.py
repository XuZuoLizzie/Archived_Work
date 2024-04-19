import os

def assert_not_none(v, prob):
  if v is None:
    print('[Problem %s]: Value should not be None' % prob)
    exit(1)

def assert_int(v, prob):
  if type(v) != int:
    print('[Problem %s]: Value should be an integer, found %s' % \
        (prob, type(v)))
    exit(1)

def assert_int_eq(gold_int, guess_int, prob):
  if gold_int != guess_int:
    print('[Problem %s]: Value should be %d, found %d' % \
        (prob, gold_int, guess_int))
    exit(1)

def assert_int_range(int_range, v, prob):
  if v < int_range[0] or v > int_range[1]:
    print('[Problem %s]: Value should be in range [%d,%d], found %d' % \
        (prob, int_range[0], int_range[1], v))
    exit(1)

def assert_float(v, prob):
  if type(v) != float:
    print('[Problem %s]: Value should be a float, found %s' % \
        (prob, type(v)))
    exit(1)

def assert_float_range(float_range, v, prob):
  if v < float_range[0] or v > float_range[1]:
    print('[Problem %s]: Value should be in range [%d,%d], found %d' % \
        (prob, float_range[0], float_range[1], v))
    exit(1)

def assert_str(v, prob, valid_values=None):
  if type(v) != str and type(v) != unicode:
    print('[Problem %s]: Value should be a string, found %s' % \
        (prob, type(v)))
    exit(1)
  if valid_values is not None and v not in valid_values:
    print('[Problem %s]: Not a valid value: %s, potential values: %s' % \
        (prob, v, valid_values))
    exit(1)

def assert_str_eq(gold_str, guess_str, prob):
  if gold_str != guess_str:
    print('[Problem %s]: Value should be \'%s\', found \'%s\'' % \
        (prob, gold_str, guess_str))
    exit(1)

def assert_str_neq(gold_str, guess_str, prob):
  if gold_str == guess_str:
    print('[Problem %s]: Value should not be \'%s\'' % (prob, gold_str))
    exit(1)

def assert_tuple(v, tup_len, prob):
  if type(v) != tuple:
    print('[Problem %s]: Value should be a tuple, found %s' % \
        (prob, type(v)))
    exit(1)
  if len(v) != tup_len:
    print('[Problem %s]: Tuple len should be %d, found %d' % (len(v), tup_len))

def assert_dict(v, prob):
  if type(v) != dict:
    print('[Problem %s]: Value should be a dict, found %s' % \
        (prob, type(v)))
    exit(1)

def assert_file(v, prob):
  if not os.path.exists(v):
    print('[Problem %s]: File should exist: %s' % (prob, v))
    exit(1)

