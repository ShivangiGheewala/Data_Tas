import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


#load dataset
train_dataframe = pd.read_csv('./dataset/train.csv')
test_dataframe = pd.read_csv('./dataset/test.csv')
print(train_dataframe.head())

#data processing
print('\n')
print(train_dataframe.leaf.unique())
output_dict ={'red':0,'green':1}
train_dataframe['leaf'] = train_dataframe.leaf.apply(lambda x:output_dict[x])
test_dataframe['leaf'] = test_dataframe.leaf.apply(lambda x:output_dict[x])
train_input = train_dataframe.iloc[:,0:8]
train_output = train_dataframe.iloc[:,8]
test_input = test_dataframe.iloc[:,0:8]
test_output = test_dataframe.iloc[:,8]
#shuffle dataset
train_dataframe = train_dataframe.sample(frac=1).reset_index(drop=True)
print(train_dataframe.head())

#data visualization
print('\n')
#bar chart
output_count  = train_dataframe['leaf'].value_counts()
plt.figure(figsize=(8,5))
labels = ['red', 'green']
leaf_colours = ['red', 'green']
sns.barplot(output_count.index, output_count.values, alpha=0.8, palette=leaf_colours)
plt.title('Class distribution')
plt.ylabel('vectors', fontsize=12)
plt.xlabel('leaves', fontsize=12)
plt.show()

# pie graph
print('\n')
figure, axis = plt.subplots()
axis.pie(output_count, labels=labels, autopct='%1.2f%%',
        shadow=True, startangle=0, counterclock=False, colors=leaf_colours)
axis.axis('equal') 
axis.set_title('Class distribution')
plt.show()

#sequential model
model = keras.Sequential()
model.add(keras.layers.Dense(4, input_shape=(8,), activation='relu'))
model.add(keras.layers.Dense(2, activation='sigmoid'))

#compile model
model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

#fit the training data
print('\n')
network = model.fit(train_input, train_output, batch_size=5, epochs=5)


#model summary
print('\n')
print(network.history)
print('\n')
print(network.history.keys())
plt.plot(network.history['accuracy'])
plt.title("Leaf Classification")
plt.ylabel('Model Accuracy')
plt.xlabel('Epoch')
plt.legend(['training curve'], loc="upper left")
plt.show()

#evaluate model
print("\nEvaluation Result")
_, accuracy = model.evaluate(test_input, test_output)
print('Accuracy: %.2f' %(accuracy*100))

#predict model
print('\n')
prediction = model.predict_classes(test_input[0:1])
input_data = test_input[0:1].to_string(header=False, index=False)
output_data = test_output[0:1].to_string(header=False, index=False)
print ("{:<80} {:<15} {:<15}".format('InputVectors','ActualOutput','PredictedOutput'))
print ("{:<80} {:<15} {:<15}".format(input_data,output_data,str(prediction)))

#predict first five values
print('\n')
predictions = model.predict_classes(test_input)
print ("{:<15} {:<10} {:<15}".format('InputVectors','ActualOutput','PredictedOutput'))
for i in range(5):
    print ("{:<15} {:<10} {:<15}".format((i+1),predictions[i],test_output[i]))



