# CMS DQM

Daily operation of a large scale experimental setup is a challenging task both in terms of maintenance and monitoring. In this work we describes an approach for automated Data Quality system. Based on the Machine Learning methods it can be trained online on manually-labeled data by human experts. Trained model can assist data quality managers filtering obvious cases (both good and bad) and asking for further estimation only of fraction of poorly-recognizable datasets.

The system is trained on CERN open data portal data published by CMS experiment. We demonstrate that our system is able to save at least 20\% of person power without increase in pollution (false positive) and loss (false negative) rates. In addition, for data not labeled automatically system provides its estimates and hints for a possible source of anomalies which leads to overall improvement of data quality estimations speed and higher purity of collected data.

## Data exctraction

Extraction is done by `data-extraction/bin/load.py` which has the following signature

```
data-extraction/bin/load.py <config> <URL list> <output directory>

config --- path to configuration file;
URL list --- list of URLs to root files with data;
output directory --- path of output directory
```

Script performs 3 essential steps:

1. downloads original data;
2. from each event selects features according to the given configuration file;
3. writes results as pickled pandas' `DataFrame`

### Feature extraction

The feature extraction is mainly defined by configuration file.
This file is in JSON format and has the following structure:

- per_lumisection: lists lumisection features;
- per_event object is devided into **channels** (PF, muons etc). For each channels there are 3 fields:
  - read_each: number
  - batch: number
  - branches: list of feature names in terms of ROOT's pathes.

Parameters `read_each` and `batch` of `per_event` section are technical and introduced sololy for speeding up feature extraction.

#### root2numpy details
This section contians description of some technical details about using `root2numpy` package thus may be skipped.

The data extraction is done with help of `root2numpy` package which provides an easy way to work with ROOT data format.
However, it requeres explicitly passing feature list.
Since technically a CMS's event is a set of particles with their features, the latter ones are stored as array,
thus requieres indecies along with feature names.
For example, if a particle feature has ROOT name `momentum`, passing `['momentum[8]', 'momentum[9]']` to `root2numpy` will return momentums of 8-th and 9-th particle for each event.

Since number of particles is uknown for each event, the extraction is performed in a batch manner until the end of the processed event.
Fortunately, arrays in the CMS's data are sorted by momentum, so that the first particle has largest momentum within event.
Having exactly zero momentum for a particle in a batch means the end of the batch has been reached and further reading is unnecessary.

The sizes of batch presented in provided configuration files were manually obtained in order to optimize reading performance.

Since the next stage of feature extraction selects `N` quantile particles, one can safely read only each k-th particle if `M/k >> N` where `M` - is total number of particles in the event. The parameter `k` is named `read_each` in the configuration files and speeds up computations in `k` times, however, introducing some statisticall inaccuracy.

### Feature extraction process
The scripts processes each channels seperatly.
After reading events as described above, for each event:

- total momentum is computed;
- `N` quantile particles are selected to represent the event.

The first selected particle has the largest momentum within event (nevertheless, the last one has not the smallest momentum).

This data is then combined and stored,
