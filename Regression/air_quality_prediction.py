import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.python.keras.losses import mean_absolute_error
import seaborn as sns
from sklearn.model_selection import train_test_split
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


#sequential model
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(480, input_shape=(8,), activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))
    return model

#compile model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model = build_model()
model.compile(optimizer=optimizer,
                loss=keras.losses.MeanAbsoluteError(),
                metrics=keras.metrics.mean_absolute_error)

#fit the training data
print('\n')
network = model.fit(train_input, train_output, epochs=5, validation_data=(test_input,test_output))


#model summary
plt.plot(network.history['mean_absolute_error'])
plt.title("Air Quality Index Prediction")
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['training curve'], loc="upper left")
plt.show()

#evaluate model
print("\nEvaluation Result")
_, mean_absolute_error = model.evaluate(test_input, test_output)
print('\nMean Absolute Error: %.2f\n' %(mean_absolute_error))

#predict model
prediction = model.predict(test_input[0:1])
input_data = test_input[0:1].to_string(header=False, index=False)
output_data = test_output[0:1].to_string(header=False, index=False)
print ("{:<50} {:<15} {:<15}".format('InputVectors','ActualOutput','PredictedOutput'))
print ("{:<50} {:<15} {:<15}".format(input_data,output_data,str(prediction)))