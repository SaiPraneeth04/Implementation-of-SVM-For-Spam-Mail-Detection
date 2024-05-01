# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.


## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Sai Praneeth K
RegisterNumber: 212222230067
```
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('/content/spam.csv', encoding='ISO-8859-1')
df.head()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['v2'])
y = df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = svm.SVC (kernel='linear') 
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy: ", accuracy_score (y_test, predictions)) 
print("Classification Report: ")
print(classification_report (y_test, predictions))
```

## Output:
## DATASET :
![9 1](https://github.com/SaiPraneeth04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390353/700380d2-6032-473e-a0a1-b589b86dbfdf)


## Kernel Model:
![9 2](https://github.com/SaiPraneeth04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390353/6182f32b-181b-4a46-9966-49114b205fc8)


## Accuracy and Classification Report :  
![9 3](https://github.com/SaiPraneeth04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119390353/bf7f15dd-83f5-4291-8fae-1eb07448b37f)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
