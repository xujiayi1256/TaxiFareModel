from tempfile import TemporaryFile

import joblib
import mlflow
from google.cloud import storage
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from TaxiFareModel.data import get_data, clean_data, df_optimized
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.params import *
from TaxiFareModel.utils import compute_rmse

MLFLOW_URI = "https://mlflow.lewagon.ai/"


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.EXPERIMENT_NAME = "[CN] [Shanghai] [xujiayi1256] TaxiFareModel v1.0"
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe,
             ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression(n_jobs=-1))
        ])
        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_param("model", "LinearRegression")
        self.mlflow_log_metric("rmse", rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


def save_model(reg):
    """ Save the trained model into a model.joblib file """

    joblib.dump(reg, 'model.joblib')
    print("saved model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


def upload_model_to_gcp():
    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.joblib')


def download_model_from_gcp():
    client = storage.Client()

    bucket = client.get_bucket(BUCKET_NAME)
    # select bucket file
    blob = bucket.blob(STORAGE_LOCATION)
    with TemporaryFile() as temp_file:
        # download blob into temp file
        blob.download_to_file(temp_file)
        temp_file.seek(0)
        # load into joblib
        model = joblib.load(temp_file)
        return model


if __name__ == "__main__":
    # get data
    df = get_data(source='aws', nrows=1_000_000)
    print(df.shape)
    # clean data
    df = clean_data(df)
    print(df.shape)
    df = df_optimized(df)
    print(df.shape)
    # set X and y
    X = df.drop(columns=["fare_amount"])
    y = df.fare_amount
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    train = Trainer(X_train, y_train)
    train.set_pipeline()
    model = train.run()
    # evaluate
    print("RMSE:", train.evaluate(X_test, y_test))

    experiment_id = train.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{experiment_id}")
    save_model(model)
