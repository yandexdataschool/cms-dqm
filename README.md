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
