import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

#load dataset
dataframe = pd.read_csv('./dataset/healthcare-dataset-stroke-data.csv')

#data processing
#find missing values
missing_data = dataframe.isna().sum()
print(missing_data)
missing_data_bmi = (missing_data['bmi']/len(dataframe)) * 100
print("\nMissing values in variable bmi: {:.2f}%".format(missing_data_bmi))

#handling missing values
dataframe['bmi']=dataframe['bmi'].fillna(dataframe['bmi'].mean())
# clean_missing_data = dataframe[dataframe['bmi'].notnull()]
# clean_missing_data.drop(columns='id',axis=1,inplace=True)

#convert string into integer
print('\n')
print(dataframe.gender.unique())
print(dataframe.ever_married.unique())
print(dataframe.work_type.unique())
print(dataframe.Residence_type.unique())
print(dataframe.smoking_status.unique())
gender_dict = LabelEncoder()
ever_married_dict = LabelEncoder()
work_type_dict = LabelEncoder()
residence_type_dict = LabelEncoder()
smoking_status_dict = LabelEncoder()

dataframe['gender'] = gender_dict.fit_transform(dataframe['gender'])
dataframe['ever_married'] = ever_married_dict.fit_transform(dataframe['ever_married'])
dataframe['work_type'] = work_type_dict.fit_transform(dataframe['work_type'])
dataframe['Residence_type'] = residence_type_dict.fit_transform(dataframe['Residence_type'])
dataframe['smoking_status'] = smoking_status_dict.fit_transform(dataframe['smoking_status'])
print('\n')
print(dataframe.head(30))

#shuffle dataset
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

#split dataset into training and testing
data_input=dataframe.iloc[:,1:-1] 
data_output=dataframe.iloc[:,-1] 
train_input, test_input, train_output, test_output = train_test_split(data_input, data_output, test_size=0.3, random_state=0)


#data visualization
figure = dataframe['avg_glucose_level'].hist(figsize=(8,6))
plt.title('Average glucose level')
plt.ylabel('range', fontsize=12)
plt.xlabel('glucose level', fontsize=12)
plt.tight_layout()
plt.show()

#features correlation
figure, axis = plt.subplots(figsize=(10,8))
correlation = axis.matshow(dataframe.corr())
# axis.set_xticks(np.arange(dataframe.shape[1]))
# axis.set_yticks(np.arange(dataframe.shape[1]))
axis.set_xticklabels(dataframe.columns,rotation=90)
axis.set_yticklabels(dataframe.columns)
correlation_bar = axis.figure.colorbar(correlation, ax=axis)
correlation_bar.ax.set_ylabel("Correlation", rotation=-90, va="bottom", fontsize=12)
plt.tight_layout()
plt.show()


#sequential model
model = keras.Sequential()
model.add(keras.layers.Dense(12, input_shape=(10,), activation='sigmoid'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='relu'))

#compile the model
optimizer = SGD(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#fit the training data
print('\n')
network = model.fit (train_input, train_output, validation_split=0.33,epochs=10, batch_size=10)

#evaluate the model
print("\nEvaluation Result")
_, accuracy = model.evaluate(test_input, test_output)
print('\nAccuracy: %.2f' %(accuracy*100))

#network summary
plt.plot(network.history['accuracy'])
plt.plot(network.history['loss'])
plt.title('Heart Stroke Prediction')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['accuracy', 'loss'], loc='lower right')
plt.show()