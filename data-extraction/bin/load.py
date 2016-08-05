import numpy as np
import pandas as pd

import root_numpy

import warnings
warnings.filterwarnings('ignore')

import traceback

import json

def get_index(leaves, indxes):
  """
  Produces names of branches with indexes in them.
  """
  return [ leaf + "[%d]" % index for leaf in leaves for index in indxes ]

def split_by_events(data_root, leaves, batch_size, test_leaf = 0, test_leaf_name = 'momentum'):
  """
  Turns data from root numpy into matrix <N events> by <number of features>.
  Returns if another batch is needed by testing test_leaf (usually pt - momentum) against exact float zero.
  If any particles has test_leaf equal to exact zeros, they are just doesn't exist in original root file, so
  needed to be truncated.
  Otherwise, event might be read incompletely, i.e. number of particles > batch_size.
  """
  events = list()

  for event_i in xrange(data_root.shape[0]):
    event = data_root[event_i]
    d = np.array([ event[i] for i in xrange(len(leaves) * batch_size) ]).reshape(len(leaves), -1).T

    event_idx = d[:, test_leaf] > 0.0

    ### other features must be exact zero also
    if not np.all(d[np.logical_not(event_idx), :] == 0.0):
      from warnings import warn
      warn('Not all features given non-negative test leaf (%s) are exact zero!' % test_leaf_name)
      nan_index = np.logical_not(event_idx)

      affected_leaves = [ (i, leaf) for i, leaf in enumerate(leaves) if np.any(d[nan_index, i] != 0.0)]
      warn('Non-zero leafs: [%s]' % (', '.join(leaf for i, leaf in affected_leaves)))

    events.append(d[event_idx, :])

  need_another_batch = np.any([
                                event.shape[0] == batch_size for event in events
                                ])

  return need_another_batch, events


def read_batch(path, treename, leaves, batch_size, each = 1, test_leaf = 0):
  event_batches = None
  need_another_batch = True

  batch_offset = 0

  while need_another_batch:
    branches = get_index(leaves, np.arange(batch_size)[::each] + batch_offset)

    data_root = root_numpy.root2array(path, treename=treename, branches=branches, )

    need_another_batch, events = split_by_events(data_root, leaves, batch_size / each, test_leaf = test_leaf)

    batch_offset += batch_size

    if event_batches is None:
      event_batches = [ [event] for event in events ]
    else:
      assert len(event_batches) == len(events)
      event_batches = [
        batches + [batch] for batches, batch in zip(event_batches, events)
        ]


  return [ np.vstack(event) for event in event_batches ]

def read_lumidata(path, lumifeatures):
  names = [ f.split('.')[-1] for f in lumifeatures ]

  lumidata = root_numpy.root2array(path, treename='Events', branches=lumifeatures)
  lumi = np.zeros(shape=(lumidata.shape[0], len(lumifeatures)))

  for i in xrange(lumidata.shape[0]):
    lumi[i, :] = np.array([ lumidata[i][j] for j in range(len(lumifeatures)) ])

  lumi = pd.DataFrame(lumi, columns=names)
  lumi['luminosityBlock_'] = lumi['luminosityBlock_'].astype('int64')
  lumi['run_'] = lumi['run_'].astype('int64')

  return lumi

def read_lumisection(path, features):
  lumi = read_lumidata(path, features['per_lumisection'])

  events = dict()
  for category in features['per_event']:
    fs = features['per_event'][category]['branches']
    read_each = features['per_event'][category]['read_each']
    batch_size = features['per_event'][category]['batch']

    assert batch_size > read_each
    assert batch_size % read_each == 0

    events[category] = read_batch(path, treename='Events', leaves=fs,
                                  batch_size=batch_size, each=read_each, test_leaf=0)

  return lumi, events

def get_percentile_paticles(event, n = 3, test_feature = 0):
  sort_idx = np.argsort(event[:, test_feature])
  event = event[sort_idx, :]

  if event.shape[0] >= n:
    ### preserving the last event with maximal momentum (test_feature)
    fetch_idx = [i * (event.shape[0] / n) for i in xrange(n-1)] + [event.shape[0] - 1]
    return event[fetch_idx, :]
  else:
    missing = n - event.shape[0]
    return np.vstack([
      np.zeros(shape=(missing, event.shape[1])),
      event
    ])

def integrate(event):
  try:
    pt = event[:, 0]
    eta = event[:, 1]
    phi = event[:, 2]

    theta = 2.0 * np.arctan(np.exp(-eta))

    px = np.sum(pt * np.cos(theta))
    py = np.sum(pt * np.sin(theta) * np.cos(phi))
    pz = np.sum(pt * np.sin(theta) * np.sin(phi))
    return np.array([px, py, pz])
  except:
    return np.zeros(shape=3)


def process_channel(channel, branches, prefix = "", n = 3, test_feature=0):
  selected = np.array([
                        get_percentile_paticles(event, n = n, test_feature = test_feature).flatten()
                        for event in channel
                        ]).astype('float32')

  total_momentum = np.array([
                              integrate(event[:3, :])
                              for event in channel
                              ]).astype('float32')

  branch_names = [ prefix + '_' + branch.split(".")[-1] for branch in branches ]

  feature_names = [
                    "%s_q%d" % (branch, q + 1) for q in range(n) for branch in branch_names
                    ] + [
                    prefix + '_' + "P%s" % component for component in list("xyz")
                    ]

  df = pd.DataFrame(np.hstack([selected, total_momentum]), columns = feature_names)
  return df

def process(data, features, lumidata, n = 3, test_feature = 0):
  channels = list()
  names = list(lumidata.columns)

  for category in data:
    d = process_channel(data[category], features[category]['branches'],
                        prefix = category, n = n, test_feature = test_feature)

    names += list(d.columns)

    channels.append(d)

  df = pd.concat([lumidata] + channels, axis=1, names = names, ignore_index=True)
  df.columns = names
  return df

def main(cfg, path, out_dir):
  import os.path as osp
  d, file_name = osp.split(path)
  outfile = osp.join(out_dir, file_name + ".pickled")

  if osp.exists(outfile):
    print "File has already been processed. Skip..."
    return

  with open(cfg, 'r') as f:
    features = json.load(f)

  lumidata, events = read_lumisection(path, features)
  df = process(events, features['per_event'], lumidata, n = 5, test_feature=0)

  df.to_pickle(outfile)

if __name__ == "__main__":
  import sys
  import time

  try:
    _, cfg, file_list, out_dir = sys.argv
  except:
    print "Usage: %s config_file file_list out_dir" % sys.argv[0]
    sys.exit(1)

  with open(file_list) as f:
    files = [ x.strip() for x in f.read().split("\n") ]
    files = [ f for f in files if len(f) > 0 ]

  for path in files:
    try:
      print "Processing file %s" % path
      start = time.time()
      main(cfg, path, out_dir)
      end = time.time()
      print 'Done %.1f minutes' % ((end - start) / 60.0)
    except Exception:
      print "An error occurred during processing file %s" % path
      print traceback.format_exc()
      print 'Skipping...'