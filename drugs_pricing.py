import argparse
import json
import sys
import logging
import os
import pandas as pd
import time
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

#### INIT CONFIG ################################################################################
def init_config(json_file_path):
    """Initializes environment variables from provided JSON file into a dictionary"""
    with open(json_file_path, "r") as f:
        config_dict = json.load(f)

    return config_dict

#### INIT LOGGER ################################################################################
def init_logger(log_level, file_location, log_filename):
    """ Initializes log structure, logs will appear into a file and in the system output """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    timestr = time.strftime("%Y%m%d%H%M%S")
    log_full_path = os.path.normpath(file_location + '/' + timestr + '_' + log_filename)

    time_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level,
                        format=time_format,
                        filename=log_full_path,
                        filemode='w')

    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(time_format)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

#### READ DATA FILE ################################################################################
def read_data_file(data_file_path, csv_decimal, csv_separator, log):
    """ Reads csv file into a Pandas dataframe and returns dataframe """
    log.info("Reading csv data file into a dataframe %s" % data_file_path)

    dataframe = pd.read_csv(data_file_path,
                       decimal=csv_decimal,
                       sep=csv_separator,
                       skipinitialspace=True,
                       encoding='utf-8',
                       dtype = { 'marketing_declaration_date': str,
                                 'marketing_authorization_date': str,
                                 'pharmaceutical_companies': str,
                                 'administrative_status': str,
                                 'approved_for_hospital_use': str,
                                 'marketing_authorization_process': str})

    # remove empty rows and columns
    log.debug("Cleaning empty and unnamed data...")
    dataframe.dropna(how='all', axis=0)
    dataframe.dropna(how='all', axis=1)

    return dataframe

#### ONE HOT ENCODE MASTER ################################################################################
def one_hot_encode_master(train_df, test_df, column, log):
    """ Build a master dataframe containing all unique values of the categorical column
    with their corresponding encoded columns """

    log.info("")
    log.info("Building ONE HOT encoding master for column: %s" % column)

    # concatenate both datasets to obtain all existing values
    all_values = pd.concat([train_df, test_df])

    # get unique values dataframe and sort
    unique_values = pd.DataFrame(all_values[column].unique(), columns=[column]).sort_values(column)

    # copy original column for reference (otherwise it disappears with get dummies)
    unique_values[column + '_orig'] = unique_values[column]
    # get dummies as one hot encoding from original values
    master_df = pd.get_dummies(unique_values, columns=[column])
    master_df.rename(columns={column + '_orig': column}, inplace=True)

    return master_df

#### ENCODE DATAFRAME ################################################################################
def encode_data(df, master_encoded_df, column):
    """ Encodes dataframe column based on master dataframe containing corresponding encoded values.
        Returns encoded dataframe without the original column"""
    enc_df = pd.merge(df, master_encoded_df, how='left', on=column)
    enc_df = enc_df.drop(column, axis=1)

    return enc_df

#### PROCESS DATA ################################################################################
def process_data(df, label_features_df, log, train=True):
    """ Performs data cleaning and transformations over input dataframe and returns the transformed
        machine learning ready dataframe """

    log.info("")
    log.info("Processing %s data..." % ("TRAIN" if train else "TEST"))

    # REMOVE DUPLICATES by checking all original features
    if train:
        log.info("Removing duplicate rows...")
        df = df.drop_duplicates(subset=['description',
                                        'administrative_status',
                                        'approved_for_hospital_use',
                                        'reimbursement_rate',
                                        'marketing_declaration_date',
                                        'marketing_authorization_date',
                                        'marketing_authorization_process',
                                        'pharmaceutical_companies'], keep="last")

    # move objective column to the start of the dataframe (only if training)
    if train:
        log.info("Reordering columns...")
        df = df[['price',
                 'drug_id',
                 'description',
                 'administrative_status',
                 'approved_for_hospital_use',
                 'reimbursement_rate',
                 'marketing_declaration_date',
                 'marketing_authorization_date',
                 'marketing_authorization_process',
                 'pharmaceutical_companies']]

    # CLEAN DATASET
    # dates extract year (the rest is not useful)
    log.info("Cleaning marketing dates...")
    df['marketing_declaration_year'] = df['marketing_declaration_date'].str[:4].astype(int)
    df['marketing_authorization_year'] = df['marketing_authorization_date'].str[:4].astype(int)

    # set reimbursement rate to numeric
    log.info("Cleaning reimbursement_rate...")
    df['reimbursement_rate'] = df['reimbursement_rate'].str.replace(r'%', '').astype(int)

    # Strip string features to clean leading and tail spaces
    log.info("Stripping categorical data...")
    df['administrative_status'] = df['administrative_status'].str.strip()
    df['approved_for_hospital_use'] = df['approved_for_hospital_use'].str.strip()
    df['marketing_authorization_process'] = df['marketing_authorization_process'].str.strip()
    #    df['pharmaceutical_companies'] = df['pharmaceutical_companies'].str.strip()

    # ADD FEATURES
    # years between declaration and authorization
    log.info("Calculating tramitation_years feature...")
    df['tramitation_years'] = df['marketing_declaration_year'] - df['marketing_authorization_year']
    # years count since the drug has been authorized
    log.info("Calculating authorized_since feature...")
    df['authorized_since'] = date.today().year - df['marketing_authorization_year']
    # pharma one word: use only initial word from pharmaceutical company name. This will reduce cardinality
    log.info("Calculating pharma_one_word feature...")
    df['pharma_one_word'] = df['pharmaceutical_companies'].str.strip().str.split(' ').str[0]
    # join with description features provided
    log.info("Adding label features...")
    df = pd.merge(df, label_features_df, how='left', on='description')

    # CLEAN count missings: replace by -1 as a simple approach (TODO improve by treating the description)
    log.info("Replacing count missing values...")
    for col in df.columns:
        if col[0:5] == 'count':
            log.debug('Replacing missing values for column: %s' % col)
            df[col] = df[col].fillna(-1)

    # DROP useless columns
    log.info("Dropping useless columns...")
    df = df.drop(columns=['description',
                          'marketing_declaration_date',
                          'marketing_authorization_date',
                          'pharmaceutical_companies'], axis=1)

    # DROP label columns
    for col in df.columns:
        if col[0:5] == 'label':
            log.debug('Dropping label column: %s' % col)
            df = df.drop(col, axis=1)

    # ENCODE basic categorical columns
    log.info("Encoding categorical columns (except pharmacy companies)...")
    df = pd.get_dummies(df, columns=['administrative_status', 'approved_for_hospital_use',
                                     'marketing_authorization_process'])

    return df

#### EVALUATE ################################################################################
def evaluate(y_pred, y_test, log):
    """ Evaluates prediction results by calculating 3 metrics comparing to actual results.
        Displays results in the log
        Returns a dictionary containing the 3 metrics"""
    log.info("")
    log.info("EVALUATION: predictions over actual prices")
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred)

    # print evaluation results
    log.info("Mean Absolute Error: %s" % mae)
    log.info("R_Squared: %s" % r2)
    log.info("Mean Absolute Percentage Error: %s" % mape)
    log.info("")

    return {"mae": mae, "r2": r2, "mape": mape}

#### EXPORT DATAFRAME ################################################################################
def export_dataframe(dataframe, csv_path, csv_filename, log):
    """ Exports dataframe into CSV according to inputs and returns path """
    timestr = time.strftime("%Y%m%d%H%M%S")
    full_dataset_filename = timestr + "_" + csv_filename + ".csv"
    # build dataset path according to OS characteristics
    full_dataset_path = os.path.normpath(csv_path + "/" + full_dataset_filename)
    log.info("Exporting data into csv file %s ..." % full_dataset_path)
    dataframe.to_csv(full_dataset_path, index=False)

    return full_dataset_path


####################################################################################
####### MAIN
####################################################################################
def main(args=sys.argv[1:]):
    """Parses command-line parameters and calls the actual main function."""
    # Process arguments
    parser = argparse.ArgumentParser(description="Drugs pricing")

    # config param
    parser.add_argument('--config',
                        required=True,
                        action='store',
                        dest='config',
                        default=None,
                        help="Full path for the JSON config file")

    args = parser.parse_args(args)

    # get config
    json_config_file = args.config

    # initialize json configuration variables into a dictionary
    config_dict = init_config(json_config_file)

    # initialize logger
    log = init_logger(config_dict["log_level"],
                      config_dict["log_directory"],
                      'drugs_pricing.log')

    log.info("Starting drugs pricing with config %s" % json_config_file)

    # read csv data files into dataframes
    train_original = read_data_file(config_dict['data_file_train'],
                                    config_dict['csv_decimal'],
                                    config_dict['csv_separator'],
                                    log)

    test_original = read_data_file(config_dict['data_file_test'],
                                    config_dict['csv_decimal'],
                                    config_dict['csv_separator'],
                                    log)

    label_features = read_data_file(config_dict['data_file_label_features'],
                                    config_dict['csv_decimal'],
                                    config_dict['csv_separator'],
                                    log)

    # process train and test data
    ml_train = process_data(train_original, label_features, log)
    ml_test = process_data(test_original, label_features, log, train=False)

    # build one hot encoder for pharma_one_word
    pharma_one_word_master = one_hot_encode_master(ml_train, ml_test, 'pharma_one_word', log)

    log.info("Encoding pharma_one_word training data...")
    ml_train_enc = encode_data(ml_train, pharma_one_word_master, 'pharma_one_word')

    log.info("Encoding pharma_one_word test data...")
    ml_test_enc = encode_data(ml_test, pharma_one_word_master, 'pharma_one_word')

    # define features and objective fields
    # features will be all remaining columns except the initial 2 (price and drug_id)
    features = ml_train_enc.columns[2:]
    objective = ['price']

    # Split provided train data to evaluate a model
    log.info("")
    log.info("Splitting provided training data to evaluate a Random Forest as a local test...")
    local_train, local_test = train_test_split(ml_train_enc, test_size=config_dict["model_split_size"], random_state=0)

    # define X and y datasets local_train and local_test
    X_local_train = local_train[features]
    y_local_train = local_train[objective]
    X_local_test = local_test[features]
    y_local_test = local_test[objective]

    # Train a RandomForestRegressor with prepared sub training data
    log.info("Training local RandomForestRegressor...")
    regressor = RandomForestRegressor(n_estimators=config_dict["model_num_trees"], n_jobs=4, random_state=0)
    regressor.fit(X_local_train, y_local_train.values.ravel())
    log.info("Performing predictions over local test dataset...")
    y_local_pred = regressor.predict(X_local_test)

    # Evaluate Local RandomForestRegressor
    eval_result = evaluate(y_local_pred, y_local_test, log)

    # Define official X,y datasets
    X_official_train = ml_train_enc[features]
    y_official_train = ml_train_enc[objective]
    X_official_test = ml_test_enc[features]

    # Train official RandomForestRegressor using all available data
    log.info("Training official RandomForestRegressor...")
    regressor.fit(X_official_train, y_official_train.values.ravel())

    # perform predictions over test data
    log.info("Performing predictions over official test dataset...")
    y_official_test_pred = regressor.predict(X_official_test)

    # generate results dataset
    log.info("Generating resulting drug price predictions...")
    result_df = pd.DataFrame(ml_test_enc['drug_id'], columns=['drug_id'])
    # predicted prices are rounded to 2 decimals in the final output
    result_df['price'] = [round(num, 2) for num in y_official_test_pred]
    result_file = export_dataframe(result_df, config_dict["datasets_directory"], 'drugs_price_predictions', log)

    log.info("")
    log.info("Resulting price predictions can be found in file: %s" %result_file)

if __name__ == "__main__":
    main()
