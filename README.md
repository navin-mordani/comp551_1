# Mini Project - 1
## Predicting Participation and Race Time for Montreal Marathon 2016

Assignment for Comp 551 Fall 2016, McGill University

## Instructions

To run linear regression, use the `linearReg.py` script and run like follows :

```
python models/linearReg.py -X [Train Feature Data File] -y [Train Target Data File] -q[Query Feature Data File] -s [Submission File Name]
```

To run logistic regression, use the `logisticReg.py` script and run like follows :

```
python models/logisticReg.py -t [Training Data File] -s [Submission File Name]
```

To run naive Bayes, use the `naiveBayes.py` script and run like follows :

```
python models/naiveBayes.py -t [Training Data File] -s [Submission File Name]
```


## Data Cleaning

Data is downloaded largely from [Sportstats.ca](https://sportstats.ca)

Data cleaning such as row transformation, feature extraction, missing data handling is done using [preprocess.py](https://github.com/navin-mordani/comp551_1/tree/master/preprocessing/preprocess.py).

The cleaned data can be found [in the data folder](https://github.com/navin-mordani/comp551_1/tree/master/data).


## Authors

* Jeremy Georges-Filteau, _jeremy.georges-filteau@mail.mcgill.ca_
* Jaspal Singh,  _jaspal.singh2@mail.mcgill.ca_
* Navin Mordani, _navin.mordani@mail.mcgill.ca_
