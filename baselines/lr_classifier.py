import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load cleaned data
# Make sure src/preprocessing.py has been run to generate this file
data_path = "data/mental_health_classification_clean.csv"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found. Please run src/preprocessing.py first.")
    exit()

df = pd.read_csv(data_path)

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['coarse_label'], test_size=0.2, random_state=42
)

# 3. Feature Engineering (TF-IDF)
print("Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train the model (Logistic Regression)
print("Training Baseline Logistic Regression classifier...")
clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# 5. Evaluation
y_pred = clf.predict(X_test_tfidf)
print("\n[Baseline Classification Results]:")
print(classification_report(y_test, y_pred))

# --- Model Persistence (Saving for Member D/Inference) ---

print("\nSaving model and vectorizer...")
# Ensure the baselines directory exists
os.makedirs('baselines', exist_ok=True)

# Save the model (classification logic)
joblib.dump(clf, 'baselines/lr_model.joblib')

# Save the vectorizer (Member D will need this to transform user queries)
joblib.dump(vectorizer, 'baselines/tfidf_vectorizer.joblib')

print("Model and vectorizer saved successfully!")
print("Outputs:\n1. baselines/lr_model.joblib\n2. baselines/tfidf_vectorizer.joblib")