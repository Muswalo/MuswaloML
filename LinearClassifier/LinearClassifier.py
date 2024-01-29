from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib


dftrain = pd.read_csv('../data/train.csv') # training dataset
dfeval = pd.read_csv ('../data/eval.csv')  # testing dataset

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')


# create feature columns. 
# what are feature colums?
# feature colums are colums that contain only the unique records from each feature in the dataframe
# we seperate numerical from categorical colums. this is done in order to acheive the above.
# the categorical colums are then encoded with numerical values so they can be passed to tensorflow


CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMS = ['age', 'fare']

feature_colums = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # gets a list of all unique values from a given featurename
    feature_colums.append (tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMS: 
    feature_colums.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# so basically what we are doing here is creating a function that 
# will define how the input function will be made. the make_input_fn i basically a blueprint of the input function
# that will create the tf.Dataset object from the current df

def make_input_fn ( data_df, label_df, num_epochs=10, shuffle=True, batch_size=32 ):
    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:
            ds = ds.shuffle(1000)

        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_fn

# we now make an input function for both our training and testing dataset using the blueprint 
# make_input_fn we defined

train_input_fn = make_input_fn (dftrain, y_train)
eval_input_fn = make_input_fn (dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier (feature_columns=feature_colums)

linear_est.train(train_input_fn)

result = linear_est.evaluate(eval_input_fn)

result = list(linear_est.predict(eval_input_fn))

print (dfeval.loc[10])
print (y_eval.loc[10])
print (result[10]['probabilities'][1])

