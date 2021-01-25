import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("dataset.csv")

X = df["Value"].values
y = df["Target"].values

tfidf = TfidfVectorizer()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

mNB = MultinomialNB()
mNB.fit(X_train, y_train)

print(mNB.score(X_test, y_test))

pred = mNB.predict(X_test)
print(pred)


#################################### <-SAVE-> ####################################
import pickle

file_name = "multinomialNB"
pickle.dump(logr, open(file_name, 'wb')) #Saves to where the .py code file is. 


#################################### <-LOAD-> ####################################
file_name = "multinomialNB"
loaded = pickle.load(open(file_name, 'rb'))

y_pred = loaded.predict(X_test)
print(y_pred)
