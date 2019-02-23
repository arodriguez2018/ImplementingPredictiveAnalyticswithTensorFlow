from __future__ import print_function

import math

from IPython import display
import matplotlib

from matplotlib import cm
from matplotlib import gridspec
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
pd.options.display.width = 1000
pd.options.display.float_format = '{:.1f}'.format

housing_df = pd.read_csv('train_LR.csv', sep = ',')
housing_df = shuffle(housing_df) #make sure not pre-ordered.

#print(housing_df)

#breakup into features and labels (targets)
processed_features = housing_df[['GrLivArea']]
output_targets = housing_df[['SalePrice']]

#SPLIT INTO 3 PARTS:  TRAIN, VAL, TEST
training_examples = processed_features[0:1060]
training_targets = output_targets[0:1060]

val_examples = processed_features[1060:1260] #fine tune model parameters
val_targets = output_targets[1060:1260]

test_examples = processed_features[1260:1460] #test for accuracy
test_targets = output_targets[1260:1460]

# build model

# CONFIGURE A NUMERIC FEATURE COLUMN FOR GRLIVAREA
my_feature_columns = [tf.feature_column.numeric_column('GrLivArea')]

# Define the preferred optimizer:  in this case lets use gradient descent
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# configure the LR model without feature columns and optimizer:
#model = tf.estimator.LinearRegressor(feature_columns=my_feature_columns, optimizer=my_optimizer)
model = tf.estimator.DNNRegressor(feature_columns=my_feature_columns, hidden_units=[12,12], optimizer=my_optimizer)

# Define the input function:
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # Convert pandas data ina a dict of np arrays
    features = {key:np.array(value) for key, value in dict(features).items()}

    # Construct a datasetj, and configure batching/reapeating.
    ds = Dataset.from_tensor_slices((features, targets)) # warning:  2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels


# Train the model from existing data
training = model.train(input_fn = lambda:my_input_fn(training_examples, training_targets), steps=1000)

# Evaluate the model with RMSE:
train_predictions = model.predict(input_fn = lambda: my_input_fn(training_examples, training_targets, num_epochs=1, shuffle=False))
val_predictions = model.predict(input_fn = lambda: my_input_fn(val_examples, val_targets, num_epochs=1, shuffle=False))
test_predictions = model.predict(input_fn = lambda: my_input_fn(test_examples, test_targets, num_epochs=1, shuffle=False))

# Format predictions as np arrays so we can calc error metrics
train_predictions = np.array([item['predictions'][0] for item in train_predictions])
val_predictions = np.array([item['predictions'][0] for item in val_predictions])
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

# print MSE and RMSE:
mean_squared_error = metrics.mean_squared_error(train_predictions, training_targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print(f'Root mean squared error (on training data): {root_mean_squared_error}')
mean_squared_error = metrics.mean_squared_error(val_predictions, val_targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print(f'Root mean squared error (on val data): {root_mean_squared_error}')
mean_squared_error = metrics.mean_squared_error(test_predictions, test_targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print(f'Root mean squared error (on test data): {root_mean_squared_error}')
"""
# fine tune.  look visually first:

# Evaluate the model with visuals:
train_sample = housing_df[0:1060].sample(30)
val_sample = housing_df[1060:1260].sample(10)
test_sample = housing_df[1260:1460].sample(10)

# get the min and max total_rooms values
x_0 = train_sample['GrLivArea'].min()
x_1 = train_sample['GrLivArea'].max()

# retrieve the final weight and bias generated while training:
weight = model.get_variable_value('linear/linear_model/GrLivArea/weights')[0]
bias = model.get_variable_value('linear/linear_model/bias_weights')

# get the predicted medeian_house_values for the min and max total_rooms values
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

# plot our regression line from (x_0, y_0) to (x_1, y_1):
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# label the graph axes
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')

# plot a scatter plot from our data sample
plt.scatter(train_sample['GrLivArea'], train_sample['SalePrice'])
plt.scatter(val_sample['GrLivArea'], val_sample['SalePrice'])
plt.scatter(test_sample['GrLivArea'], test_sample['SalePrice'])

# display the graph
plt.show()

"""