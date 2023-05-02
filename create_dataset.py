#!/bin/python3

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://raw.githubusercontent.com/dccuchile/CC6205/master/assignments/new/assignment_1/train/train.tsv", sep="\t")

X_train, X_test, y_train, y_test = train_test_split(
    data.texto,
    data.clase,
    shuffle=True,
    test_size=0.30,
    random_state=11,
    stratify=data.clase
)

X_test, X_dev, y_test, y_dev = train_test_split(
    X_test,
    y_test,
    shuffle=True,
    test_size=0.33,
    random_state=11,
    stratify=y_test
)

pd.concat(
    [X_train, y_train],
    axis=1
).to_csv(
    "data/processed/train.csv",
    index=False,
    header=["text","label"]
)

pd.concat(
    [X_test, y_test],
    axis=1
).to_csv(
    "data/processed/test.csv",
    index=False,
    header=["text","label"]
)

pd.concat(
    [X_dev, y_dev],
    axis=1
).to_csv(
    "data/processed/dev.csv",
    index=False,
    header=["text","label"]
)