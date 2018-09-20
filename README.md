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

Data was downloaded from largely from [Sportstats.ca](https://sportstats.ca)

Data Cleaning is done in two parts. First the given dataset is row transformed, basic features extracted and location added using [data-process.py](preprocessing/data-process.py). Then using SQL Server the processed data is again mined in [FinalScript.sql](preprocessing/FinalScript.sql) for new features and made ready for Regression. In order to run the SQL file, Microsoft SQL Server 2016 should be installed and along with it necessary drivers.

The cleaned data can be found [in this public location](https://github.com/koustuvsinha/data-adventures/tree/master/montreal2016).

## Testing

In order to test our algorithm implementation, we have created a simple test file in [learner_test.py](test/learner_test.py). To run it simply use python command line : `python test/learner_test.py`

## Authors

* Jeremy Georges-Filteau, _jeremy.georges-filteau@mail.mcgill.ca_
* Jaspal Singh,  _jaspal.singh2@mail.mcgill.ca_
* Navin Mordani, _navin.mordani@mail.mcgill.ca_
