# Demo for new Azure Machine Learning SDKs

The purpose of this demo is to test the APIs and features provided by the new Azure Machine Learning SDKs (internal preview version):

### 1. Azure Machine Learning SDK for Python

Azure Workbench application and some other early features were replaced in the September 2018 release to make way for an improved architecture. The core functionality from experiment runs to model deployment has not changed, but now we can use the robust SDK and CLI to accomplish our machine learning tasks and pipelines.

### 2. Azure Machine Learning Data Prep SDK

The Azure Machine Learning Data Prep SDK is used to load, transform, and write data for machine learning workflows. We can interact with the SDK in any Python environment, including Jupyter Notebooks or other Python IDEs.

Compared with other popular data preparation package, for example `Pandas`, the Azure Machine Learning Data Prep SDK has some advantages, for example:

- The SDK can automatically detect any of the supported file types. You don’t need to use special file readers for CSV, text, Excel, etc., or to specify delimiter, header, or encoding parameters.
- Instead of loading all the data into memory, the SDK engine serves data using streaming, allowing it to scale and perform better on large datasets.

## Demo

The demo is inspired by a Kaggle machine learning competition **[« New York City Taxi Trip Duration »](https://www.kaggle.com/c/nyc-taxi-trip-duration)**. The goal of the competition is to predict the total ride duration of taxi trips in New York City.

## Files

- `Create workspace config file.ipynb`: Create configuration file for the workspace before training models
- `Data preparation.ipynb`: Data cleaning & Featrue engineering
- `Deploy models.ipynb`: Deploy the model as a web service & Scoring 
- `Download Data from Azure Storage.ipynb`: Download data from cloud
- `Train models with Batch.ipynb`: Train the model with a cluster of low-priority virtual machines
- `Train models with DSVM.ipynb`: Train the model with a Azure Data Science Virtual Machine
