import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

data = {
    "text":[
        "I love this product",
        "Amazing experience",
        "Very good service",
        "Bad experience",
        "I hate this",
        "Worst product"
    ],

    "label":[1,1,1,0,0,0]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = MultinomialNB()
model.fit(X,y)

pickle.dump(model,open("model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

print("Model Trained")