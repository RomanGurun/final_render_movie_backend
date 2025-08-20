import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
file_path = os.path.join(BASE_DIR, "data", "IMDB Dataset.csv")  # renamed without space
df = pd.read_csv(file_path)

# If your dataset columns are named "review" and "sentiment"
df = df.rename(columns={
    "review": "content",
    "sentiment": "label"
})

# 1. Load training data
corpus = df["content"].astype(str).tolist()
labels = df["label"].tolist()

# 2. Fit vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# 3. Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)

# 4. Save fitted vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# 5. Save trained classifier
with open("clf.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Model and vectorizer trained & saved")
