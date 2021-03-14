import tensorflow
import pandas
import numpy

#Downloading Data
dataset = pd.read_csv("Wellbeing_and_lifestyle_data.csv")

#Data Preparation


x = dataset.iloc[: ,1: 20]
x = x.drop(labels =["DAILY_STRESS","BMI_RANGE"], axis=1)
x = numpy.asarray(x).astype(numpy.float32)
y = dataset.iloc[: , 9].values

#Encoding

from sklearn.preprocessing import  LabelEncoder
le = LabelEncoder()
y[:, ] = le.fit_transform(y[:, ])

#Spliting Data and Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
tensorflow.convert_to_tensor(y_train)
tensorflow.convert_to_tensor(X_train)
x
type(x)

#Initializing Artificial Neural Network

ann = tensorflow.keras.models.Sequential()

ann.add(tensorflow.keras.layers.Dense(units=21, activation='relu'))
ann.add(tensorflow.keras.layers.Dense(units=21, activation='relu'))
ann.add(tensorflow.keras.layers.Dense(units=21, activation='relu'))
ann.add(tensorflow.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Start Training 
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#predicting 
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(numpy.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

#see results
accuracy_score(y_test, y_pred)

