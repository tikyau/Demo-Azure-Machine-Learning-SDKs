
import os
import argparse
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from azureml.core import Run

# get hold of the current run
run = Run.get_submitted_run()

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
args = parser.parse_args()

data_folder = args.data_folder
data_path = os.path.join(data_folder, 'data_after_prep.pkl')
run.log('Data path', data_path)

# load data
pd_dataframe = pd.read_pickle(data_path)
run.log('Data loading', 'finished')

# data processing
le = preprocessing.LabelEncoder()
le.fit(["N", "Y"])
pd_dataframe["store_and_fwd_flag"] = le.transform(pd_dataframe["store_and_fwd_flag"])

le.fit(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
pd_dataframe["pickup_weekday"] = le.transform(pd_dataframe["pickup_weekday"])
pd_dataframe["dropoff_weekday"] = le.transform(pd_dataframe["dropoff_weekday"])
run.log('Data processing', 'finished')

# load dataset into numpy arrays
y = np.array(pd_dataframe["trip_duration"]).astype(float)
y = np.log(y)
X = np.array(pd_dataframe.drop(["trip_duration"],axis = 1))

# normalize data
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
run.log('Normalization', 'finished')

# split data into train and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state = 20)

# train LR model
lm = LinearRegression()
lm.fit(X_train, y_train)
run.log('Model training', 'finished')

y_pred = lm.predict(X_val)
run.log('Prediction', 'finished')

# evaluation
mse = mean_squared_error(y_val, y_pred)
run.log('Evaluation', 'finished')
run.log('Mean Squared Error', np.float(mse))

os.makedirs('outputs', exist_ok=True)
# note!!! file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=lm, filename='outputs/nyc_taxi_model.pkl')