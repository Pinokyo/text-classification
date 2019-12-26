#%%
"""You have download *keras* to compile it"""
import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download('stopwords')
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

test_size = 0.3 # %70 for training and  %30 for testing


"""Data Pre-processing"""

results_not_spam = []

not_spam = open('./data/sms_ham.txt', 'r', encoding="utf8")
for message in not_spam:
    message = re.sub('[^a-zA-ZçğıöşüÇĞİÖŞÜ]', ' ', message)
    message = message.lower()
    message = message.split()
    message = [ps.stem(word) for word in message if not word in set(stopwords.words('turkish'))]
    message = ' '.join(message)
    results_not_spam.append(message)

results_spam = []

spam = open('./data/sms_spam.txt', 'r', encoding="utf8")
for message in spam:
    message = re.sub('[^a-zA-ZçğıöşüÇĞİÖŞÜ]', ' ', message)
    message = message.lower()
    message = message.split()
    message = [ps.stem(word) for word in message if not word in set(stopwords.words('turkish'))]
    message = ' '.join(message)
    results_spam.append(message)

results = results_not_spam + results_spam

cv = CountVectorizer()
x = cv.fit_transform(results).toarray()

y_not_spam = np.ones((len(results_not_spam),))
y_spam = np.zeros((len(results_spam),))
y = np.append(y_not_spam, y_spam, axis = 0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

num_of_samples = len(X_train[0])
num_of_epochs = 15

model = Sequential()
model.add(Dense(int(num_of_samples / 2), kernel_initializer = "uniform", activation = "relu", input_dim = num_of_samples))
model.add(Dense(int(num_of_samples / 2), kernel_initializer = "uniform", activation = "relu"))
model.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = num_of_epochs)
y_pred = model.predict(X_test)
y_pred = y_pred > 0.5

"""Confusion Matrix"""
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy:", (cm[0][0] + cm[1][1]) / sum(map(sum, cm)))