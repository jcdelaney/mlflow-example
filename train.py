import click

import pandas as pd
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

@click.command()
@click.option("--csv-path")
@click.option("--outcome-name")
def run(csv_path, outcome_name):

    data = pd.read_csv(csv_path)

    x = data.drop([outcome_name], axis=1)
    y = data[[outcome_name]]

    lr = LogisticRegression()
    lr.fit(x, y.values.ravel())

    mlflow.sklearn.log_model(lr, "model")

if __name__ == '__main__':
    run()