# Databricks notebook source
dbutils.library.installPyPI("mlflow", "1.0.0")

# COMMAND ----------

import boto3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 1. Select any of the F1 datasets in AWS S3 to build your model. You are allowed to join multiple datasets.
# MAGIC Will build a model to predict pit times using the pit_stops dataset and the races dataset (joined to get circuitId to use as a feature). Data imported and joined below.

# COMMAND ----------

s3 = boto3.client(
    's3',
    aws_access_key_id='',
    aws_secret_access_key=''
)

# COMMAND ----------

bucket = "ne-gr5069"
f1_pit_stops = "raw/pit_stops.csv"
f1_races = "raw/races.csv"

obj_pit_stops = s3.get_object(Bucket= bucket, Key= f1_pit_stops) 
obj_races = s3.get_object(Bucket= bucket, Key= f1_races) 

# COMMAND ----------

df_pit_stops = pd.read_csv(obj_pit_stops['Body'])
display(df_pit_stops)

# COMMAND ----------

df_races = pd.read_csv(obj_races['Body'])
df_races_select = df_races[['raceId', 'circuitId', 'name']]
display(df_races_select)

# COMMAND ----------

df_pits_races = df_pit_stops.merge(df_races_select, on = 'raceId', how = 'left')

display(df_pits_races)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 2. Build any model of your choice
# MAGIC Use df_pits_races to build a random forest model...
# MAGIC 
# MAGIC **Features**
# MAGIC * Circuit ID (OneHotEncoded)
# MAGIC * driverId (OneHotEncoded)
# MAGIC * stop
# MAGIC * lap
# MAGIC 
# MAGIC **Target**
# MAGIC * Milliseconds

# COMMAND ----------

df_pits_races_4_model = df_pits_races[['circuitId', 
                                      'driverId',
                                      'stop',
                                      'lap',
                                      'milliseconds']]

df_pits_races_4_model_encoded = pd.get_dummies(df_pits_races_4_model,
                                              columns = ['circuitId',
                                                         'driverId']
                                              )

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(df_pits_races_4_model_encoded.drop(["milliseconds"],
                                                                                       axis=1),
                                                    df_pits_races_4_model_encoded[["milliseconds"]],
                                                    random_state=42)

# COMMAND ----------

# MAGIC %md 
# MAGIC # ML Flow Setup
# MAGIC ### 3. Log the parameters used in the model in each run
# MAGIC ### 4. Log the model
# MAGIC ### 5. Log every possible metric from the model
# MAGIC ### 6. Log at least two artifacts (plots, or csv files)
# MAGIC 
# MAGIC Logged artifacts - residual plot and feature importance CSV is available in the
# MAGIC detailed run pages. For run 12 (the best model) These artifacts have been
# MAGIC downloaded and pushed to the repo in the folder deliverables/run_12_logged_items

# COMMAND ----------

with mlflow.start_run(run_name="Basic Experiment") as run:
  runID = run.info.run_uuid
  experimentID = run.info.experiment_id

# COMMAND ----------

def log_rf(experimentID, run_name, params, X_train, X_test, y_train, y_test):
  import os
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import explained_variance_score, max_error
  from sklearn.metrics import mean_absolute_error, mean_squared_error
  from sklearn.metrics import mean_squared_log_error, median_absolute_error 
  from sklearn.metrics import r2_score, mean_poisson_deviance
  from sklearn.metrics import mean_gamma_deviance
  import tempfile

  with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")

    # Log params
    [mlflow.log_param(param, value) for param, value in params.items()]

    # Create metrics
    exp_var = explained_variance_score(y_test, predictions)
    max_err = max_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared = False)
    mslogerror = mean_squared_log_error(y_test, predictions)
    medianae = median_absolute_error(y_test,predictions)
    r2 = r2_score(y_test, predictions)
    mean_poisson = mean_poisson_deviance(y_test, predictions)
    mean_gamma = mean_gamma_deviance(y_test, predictions)
    
    # Print metrics
    print("  explained variance: {}".format(exp_var))
    print("  max error: {}".format(max_err))
    print("  mae: {}".format(mae))
    print("  mse: {}".format(mse))
    print("  rmse: {}".format(rmse))
    print("  mean square log error: {}".format(mslogerror))
    print("  median abosulte error: {}".format(medianae))
    print("  R2: {}".format(r2))
    print("  mean poisson deviance: {}".format(mean_poisson))    
    print("  mean gamma deviance: {}".format(mean_gamma))
    
    # Log metrics
    mlflow.log_metric("explained variance", exp_var)
    mlflow.log_metric("max error", max_err)  
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)  
    mlflow.log_metric("mean square log error", mslogerror)  
    mlflow.log_metric("median abosulte error", medianae)
    mlflow.log_metric("R2", r2)  
    mlflow.log_metric("mean poisson deviance", mean_poisson)  
    mlflow.log_metric("mean gamma deviance", mean_gamma)

    
    # Create feature importance
    importance = pd.DataFrame(list(zip(df_pits_races_4_model_encoded.columns,
                                       rf.feature_importances_)), 
                                columns=["Feature", "Importance"]
                              ).sort_values("Importance", ascending=False)
    
    # Log importances using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
    temp_name = temp.name
    try:
      importance.to_csv(temp_name, index=False)
      mlflow.log_artifact(temp_name, "feature-importance.csv")
    finally:
      temp.close() # Delete the temp file
    
    # Create plot
    fig, ax = plt.subplots()

    sns.residplot(predictions, y_test.values.ravel(), lowess=False)
    plt.xlabel("Predicted values pit duration")
    plt.ylabel("Residual")
    plt.title("Residual Plot for pitting")

    # Log residuals using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="residuals_pit_model", suffix=".png")
    temp_name = temp.name
    try:
      fig.savefig(temp_name)
      mlflow.log_artifact(temp_name, "residuals_pit_model.png")
    finally:
      temp.close() # Delete the temp file
      
    display(fig)
    return run.info.run_uuid

# COMMAND ----------

# MAGIC %md
# MAGIC # Model training and tracking
# MAGIC ### 7. Track your MLFlow experiment and run at least 10 with different parameters

# COMMAND ----------

params_run1 = {'n_estimators': 100,
               'max_depth': 5,
               'random_state': 42
              }

log_rf(experimentID, 'Run 1', params_run1, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_run2 = {'n_estimators': 100,
               'max_depth': 4,
               'random_state': 42
              }

log_rf(experimentID, 'Run 2', params_run2, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_run3 = {'n_estimators': 100,
               'max_depth': 3,
               'random_state': 42
              }

log_rf(experimentID, 'Run 3', params_run3, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_run4 = {'n_estimators': 100,
               'max_depth': 2,
               'random_state': 42
              }

log_rf(experimentID, 'Run 4', params_run4, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_run5 = {'n_estimators': 100,
               'max_depth': 1,
               'random_state': 42
              }

log_rf(experimentID, 'Run 5', params_run5, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_run6 = {'n_estimators': 1000,
               'max_depth': 5,
               'random_state': 42
              }

log_rf(experimentID, 'Run 6', params_run6, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_run7 = {'n_estimators': 1000,
               'max_depth': 4,
               'random_state': 42
              }

log_rf(experimentID, 'Run 7', params_run7, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_run8 = {'n_estimators': 1000,
               'max_depth': 3,
               'random_state': 42
              }

log_rf(experimentID, 'Run 8', params_run8, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_run9 = {'n_estimators': 1000,
               'max_depth': 2,
               'random_state': 42
              }

log_rf(experimentID, 'Run 9', params_run9, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_run10 = {'n_estimators': 1000,
               'max_depth': 1,
               'random_state': 42
              }

log_rf(experimentID, 'Run 10', params_run10, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_run11 = {'n_estimators': 100,
               'max_depth': 6,
               'random_state': 42
              }

log_rf(experimentID, 'Run 11', params_run11, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_run12 = {'n_estimators': 1000,
               'max_depth': 6,
               'random_state': 42
              }

log_rf(experimentID, 'Run 12', params_run12, X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md 
# MAGIC # 8. Select your best model run and explain why
# MAGIC 
# MAGIC Run 12 is the best model run. It has the best score between all models that were tested
# MAGIC on multiple metrics. Actually, if I understand mean gamma deviance and mean poisson 
# MAGIC deviance correctly, it does the best on all the available metrics sklearn has for 
# MAGIC regression (except median absolute error where it loses to run 11), so there isn't
# MAGIC much contest.
# MAGIC 
# MAGIC You can also see from ther residual plots that while all the RF models tend to 
# MAGIC mis-predict in similar ways, by looking at the y-axis you can see that mispredictions
# MAGIC are kept to a smaller (absolute) range when max_depth is higher (with it being highest
# MAGIC for runs 11 and 12). Comparing just run 11 and 12 (which differ in the number of trees),
# MAGIC you can't see much difference, but the metrics tip the scales in favour of run 12 (which
# MAGIC didn't take much longer to train than run 11 anyway, despite having 10x more trees).
# MAGIC 
# MAGIC ### 9. Take a screenshot of your MLFlow Homepage as part of your assignment submission 
# MAGIC ### 10. Take a screenshot of your detailed run page
# MAGIC 
# MAGIC Both of MLFlow Homepage screenshot and detailed run page for Run 12 have been
# MAGIC added to the repo in the folder deliverables/mlflow_screenshots

# COMMAND ----------

# MAGIC %md
# MAGIC # Assignment 4 - Data Visualisation with Tableau
# MAGIC 
# MAGIC Predictions taken from run 12 model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Get predictions, merge with original data

# COMMAND ----------

#Get Run 12
run12= RandomForestRegressor(n_estimators = 1000, max_depth = 6, random_state = 42)
run12.fit(X_train, y_train)
predictions = run12.predict(X_test)

# COMMAND ----------

predictions_rounded = pd.DataFrame(data = np.round(predictions, decimals = 0),
                                   index = X_test.index,
                                   columns = ['predicted_laptime'])

# COMMAND ----------

# Get driver names
bucket = "ne-gr5069"
f1_drivers = "raw/drivers.csv"
obj_drivers = s3.get_object(Bucket= bucket, Key= f1_drivers) 
df_drivers = pd.read_csv(obj_drivers['Body'])

# COMMAND ----------

df_drivers['full_name'] = df_drivers['forename'] + " " + df_drivers['surname']
df_drivers_for_merge = df_drivers[['driverId', 'full_name']]
display(df_drivers_for_merge)

# COMMAND ----------

df_test_data = df_pits_races_4_model[df_pits_races_4_model.index.isin(X_test.index)]
df_test_data_w_pred = pd.concat([df_test_data, predictions_rounded], axis = 1)\
  .rename(columns = {'milliseconds': 'laptime (ms)', 'predicted_laptime': 'predicted laptime (ms)'})

# COMMAND ----------

# replace IDs with names
df_test_data_w_pred = df_test_data_w_pred.merge(df_races_select[['circuitId', 'name']], on = 'circuitId')\
  .merge(df_drivers[['driverId', 'full_name']], on = 'driverId')\
  .drop(['circuitId', 'driverId'], axis=1)\
  .rename(columns = {'name': 'circuit_name'})

display(df_test_data_w_pred)

# COMMAND ----------

# Convert to spark df
spark_df_test_data_w_pred = spark.createDataFrame(df_test_data_w_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Write to MySQL server

# COMMAND ----------

spark_df_test_data_w_pred.write.format('jdbc').options(
      url='jdbc:mysql://gr5069.cgknx318yygb.us-east-1.rds.amazonaws.com/gr5069',
      driver='com.mysql.jdbc.Driver',
      dbtable='df_pits_races',
      user='admin',
      password='gr5069_demo').mode('overwrite').save() #mode can be override/append
