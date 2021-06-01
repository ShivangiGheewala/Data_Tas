import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch
import time 
import warnings
warnings.filterwarnings('ignore')

#load dataset
dataframe = pd.read_csv('./dataset/Air_Quality_Index_Prediction.csv')
print(dataframe.head())

#data processing
print("Data shape: {}\n".format(dataframe.shape))
#count missing values in each column of dataset
print(dataframe.isna().sum())
print('\n')

#find the index of missing values
print(dataframe['PM 2.5'][dataframe['PM 2.5'].isna()])
print('\n')

#fill missing values with zero
dataframe.iloc[:185,-1] = dataframe.iloc[:185,-1].fillna(0)
print(dataframe.iloc[184])
print('\n')

#split into train and test
data_input=dataframe.iloc[:,:-1] 
data_output=dataframe.iloc[:,-1] 
train_input, test_input, train_output, test_output = train_test_split(data_input, data_output, test_size=0.3, random_state=0)


#sequential Model
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
                            #    activation=hp.Choice('act_' + str(i), 
                            #                 ['relu','sigmoid'])))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model

#compile Model
LOG_DIR = f"{int(time.time())}"
tuner = RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=5,
    executions_per_trial=3,
    directory='Keras_Hyperparameters_Tuning',
    project_name='Air_quality_prediction_'+LOG_DIR)

#parameters summary
tuner.search_space_summary()

#fit the model
tuner.search(train_input, train_output, epochs=5, validation_data=(test_input,test_output))

#tuner summary
print("\nTuner Summary")
tuner.results_summary()