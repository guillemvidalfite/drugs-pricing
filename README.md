# Drug Price Prediction

## Project Description

The objective is to predict the price for each drug in the test data set (`drugs_test.csv`). Please refer to the `sample_submission.csv` file for the correct format for submissions.

## Guidelines

Build a machine learning pipeline that will:

1. Preprocess the data.
2. Train and evaluate a model of your choice.
3. Generate predictions on the test set.

We expect an application layout, not notebooks (but feel free to also share your notebooks if you want). The code should respect production standards, think that your code will be the first iteration of a pipeline the company will use in production.

You are free to define appropriate performance metrics that fit the problem and chosen algorithm.

Please modify the `README.md` to add (in English):

1. Instructions on how to run your code.
2. A paragraph or two about what algorithm was chosen for which problem, why (including pros/cons). What you are particularly proud of in your implementation, and why.
3. Overall performance of your algorithm(s).
4. Next steps and potential improvements.

You should not spend more than 6 hours working on the test. We're aware that it's not enough to fully answer the problem, but that's OK. We will value quality over quantity and the most important aspect of the test will be to have the basics right. Don’t waste your time developing things that you are not asked for, such as an API to serve your model, you won’t be more valued if you do that. 


## Evaluation criteria

Evaluation of your submission will be based on the following criteria:

- **Documentation**: We should be able to understand your approach.
- **Sofware engineering**: 
  1. Code quality: Code is written once but read many times. Please make sure that your code is well-documented, and is free of programmatic and stylistic errors.
  2. Reproducibility: We should be able to reproduce your work and achieve the same results.
- **Machine learning**:
  1. Feature engineering: What features in the data set were used and why?
  2. Modeling: Did you apply an appropriate machine learning algorithm for the problem and why you have chosen it?
  3. Evaluation: Did you apply an appropriate evaluation metric for the use case and why you have chosen it?

There are many ways and algorithms to solve these questions; we ask that you approach them in a way that showcases one of your strengths. Please make sure you follow all the instructions before submission.

## Files & Field Descriptions

You'll find five CSV files:
- `drugs_train.csv`: training data set,
- `drugs_test.csv`: test data set,
- `drug_label_feature_eng.csv`: feature engineering on the text description,
- `sample_submission.csv`: the expected output for the predictions.

### Drugs

Filenames: `drugs_train.csv` and `drugs_test.csv`

| Field | Description |
| --- | --- |
| `drug_id` | Unique identifier for the drug. |
| `description` | Drug label. |
| `administrative_status` | Administrative status of the drug. |
| `approved_for_hospital_use` | Whether the drug is approved for hospital use (`oui`, `non` or `inconnu`). |
| `reimbursement_rate` | Reimbursement rate of the drug. |
| `marketing_declaration_date` | Marketing declaration date. |
| `marketing_authorization_date` | Marketing authorization date. |
| `marketing_authorization_process` | Marketing authorization process. |
| `pharmaceutical_companies` | Companies owning a license to sell the drug. Comma-separated when several companies sell the same drug. |
| `price` | Price of the drug (i.e. the output variable to predict). |

**Note:** the `price` column only exists for the train data set.


### Text Description Feature Engineering

Filename: `drug_label_feature_eng.csv`

This file is here to help you and provide some feature engineering on the drug labels.

| Field | Description |
| --- | --- |
| `description` | Drug label. |
| `label_XXXX` | Dummy coding using the words in the drug label (e.g. `label_ampoule` = `1` if the drug label contains the word `ampoule` - vial in French). |
| `count_XXXX` | Extract the quantity from the description (e.g. `count_ampoule` = `32` if the drug label  the sequence `32 ampoules`). |

**Note:** This data has duplicate records and some descriptions in `drugs_train.csv` or `drugs_test.csv` might not be present in this file.

Good luck.

# Proposed Solution Comments:

## 1. Installation

The package will be simply delivered in a compressed zip file and needs to be unzipped in any desired directory.
It includes the following files:
- `drugs_pricing.py`: Python script to be executed
- `config.json`: Configuration file where certain parameters can be defined for the execution
- `README.md`: Readme file including instructions

### 1.1. Installation requirements
Python version used:
- `Python 3.7` (or higher)

The following python libraries are required:
- `Pandas 1.3.5` (or higher)
- `Scikit-learn 1.0.2` (or higher)

### 1.2. Installation instructions
Uncompress the delivered file in the wished location and configure the included config.json file according to your system.
All paths are required to be set and other parameters are optional.

#### Configuration file
The configuration file includes the following keys:
- `csv_decimal`: Decimal separator for the imported csv files (set to ".")
- `csv_separator`: Main separator in the imported csv files (set to ",")
- `data_file_test`: Full path of the test data file to be used 
- `data_file_train`: Full path of the train data file to be used
- `data_file_label_features`: Full path of the label features file to be used
- `datasets_directory`: Directory where the resulting output files will be written 
- `log_directory`: Directory where the resulting log files will be located
- `log_level`: Log level for output messages in both the system output and file
- `model_split_size`: Split size to be used in the evaluation
- `model_num_trees`: Number of trees included in the Random Forest model used

**Note**: JSON special characters will need to be escaped with \ in the configured paths (especially in Windows as \ is a special character).
Validity of the JSON file can be checked by using a JSON compatible editor.

## 2. Execution
The program consists in a python script and a configuration file that needs to be referenced.
It needs to be executed from a command line having access to a Python environment with installation requirements fulfilled.

The following command will need to be processed:

`python "/full/path/to/your/script/drugs_pricing.py" --config "/full/path/to/your/config.json"`

**Notes:**
- Please replace paths in the command according to your environment 
- Paths are passed between double quotes for more robustness
- Paths can be adapted to any OS

### Execution outputs
System output and a log file will include output detailed messages for each execution.
The ultimate output will be a csv resulting file generated in the configured directory.
Both files are generated with a datestamp for each execution in the directories previously configured.

## 3. Data Processing

### 3.1. Data cleaning
- Dates were only yearly, that's why month and day have been removed to keep a year.
- The reimbursement rate has been transformed to an integer.
- Categorical fields have been stripped.
- Label fields have been removed as they would be redundant with count fields without providing additional information.
- Multiple columns have been removed and probably still some should be removed by performing feature selection.
- Duplicates were removed from the training set.

### 3.2. Missing values
Many count features were empty, potentially reducing performance in a considerable percent of the data. 
Considering a Random Forest algorithm will be chosen which deals correctly with empty values, as an initial simple approach, empty values have been simply encoded to -1.
However it would be easy to recalculate those empty values from descriptions as follows.

For each word listed:
- First label field would be informed by simply checking the presence of the word.
- Then count would be represented by the integer found just before the word in the string.
Implementation was not included because of time matters.

### 3.3. Feature engineering
- Pharmacy company has been simplified to only the initial word. It contained over 350 values and some were quite similar. By only using the initial word cardinality is reduced to 250 by grouping many similar companies. This was a quick solution it would be better to work it out with domain knowledge. Also I did not test it because of lack of time. It was only intuition and might be negative.
- I tried extracting the country out of pharmacy companies (sometimes between parenthesis) and tested but the feature had almost no effect in my model importance as most were empty so I quickly discarded it.
- Calculated the difference between marketing dates provided in case it had an impact.
- Calculated the age of drugs which in general represents more than actual years.

### 3.4. Feature selection
Feature selection was not performed. Initial tests fields importances gave an idea to drop some of the features only.
With more time, additional feature selection would take place manually or by applying existing algorithms such as Recursive Feature Elimination.
Note that feature selection is not much of a problem for Random Forests too.

## 4. Algorithm
After inspecting the data it appeared empty values were present, together with categorical fields having unbalanced distributions. Especially, the distribution of the price as the objective field was very unbalanced: most of drugs had a price under 50.
As Decision Trees use to work well with unstructured Data it already seemed a good way to go. Also the dataset was not very large and decision trees are very prone to overfitting so going for ensembles would be a better decision.
To take a final decision I uploaded a dataset into BigML.com (a ML as a Service platform and the company I currently work for) and made some quick tests.
Auto ML tests to optimize R2 by using cross-validation showed Random Forests (with around 100 trees of 1000 nodes and a 50% candidate ratio) performed better than other regressors (Bootstrap ensembles, Boosting ensembles, Deep networks and Linear Regressions) with our training dataset.
That's why Random Forest was picked as the algorithm for this case.
For the past years I have been using BigML models via BigML API in Python, which are very handy as they incorporate many automatic features. However for this test I thought it would be easier for you using standard Python libraries so I picked sklearn. 

## 5. Evaluation
A simple 80/20 train/test split has been performed in the tool (Cross-validation would have been nicer).
The classical metrics used were:
- Mean Absolute Error: It is always interesting for regressions to have an idea of the mean error.
- R-Squared: As the universal regression metric, R-Squared considers data distribution and always should have values between 0 and 1 (<0 would mean unacceptable performance). In our case it is under 0.45 and means the algorithm is performing, however we could be much closer to 1. So I think in our case it helps us understand where we are even without knowing about drug prices. In this case, to me 0.45 is an indicator that we need to go for more iterations in that use case.
- On another hand, we used the **Mean Absolute Percentage Error** considering the price field distribution. A MAE of 14 could have a very different meaning for a drug costing 40€ than a drug costing 500€ so it would be important to see the percentual error. It appeared to be over 1% which in the end seems like a good initial result.

## 6. Next Steps
Time was limited and many improvements could be added. A few are listed below:
- **Data cleaning**: Fill missing values by recalculating labels and counts as mentioned above
- **Cleaning outliers**: An anomaly detector could be used to clean outliers. This could be tested quickly.
- **Feature engineering**: Spend more time considering Pharmaceutical companies simplifications, especially with domain knowledge ideas could come. Test performance for different formats.
- **Clusters**: A cluster could be used to segment drugs by adding the centroid as an additional feature. This could be tested quickly. 
- **Feature selection**: Spot and remove potentially useless features.
- **Algorithm**: Test hyper parameters.
- **Packaging**: Delivery packaging could be improved if required. A simple approach has been chosen.



