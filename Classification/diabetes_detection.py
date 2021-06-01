from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

#load dataset
dataset = loadtxt('./dataset/pima-indians-diabetes.csv', delimiter=',')

#split dataset into input and output variables
x = dataset[:,0:8]
y = dataset[:,8]

#keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add (Dense(1, activation='sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the training data
print('\n')
network = model.fit (x, y, validation_split=0.33,epochs=150, batch_size=10)

#evaluate the model
_, accuracy = model.evaluate(x, y)
print('\nAccuracy: %.2f' %(accuracy*100))

#make class predictions with the model
print('\n')
predictions = model.predict_classes(x)
for i in range(5):
    print ('%s => %d (predicted %d)' % (x[i].tolist(),predictions[i],y[i]))