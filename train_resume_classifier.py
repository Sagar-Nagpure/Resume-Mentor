import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("resume_data.csv")
df = df.dropna()

X_train, X_test, y_train, y_test = train_test_split(
    df['Resume'], df['Category'], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'resume_classifier.pkl')
print("✅ Model saved as resume_classifier.pkl")
