import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('src/main/resources/StudentPerformanceFactors.csv')

# Preprocessing (using previous problem)
category_col = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                'Internet_Access', 'School_Type', 'Peer_Influence',
                'Learning_Disabilities', 'Parental_Education_Level', 'Gender']
df = pd.get_dummies(df, columns=category_col, drop_first=True)

remaining_category_col = ['Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Distance_from_Home']
df = pd.get_dummies(df, columns=remaining_category_col, drop_first=True)

# Prepare features and target
df['Exam_Score_Binary'] = df['Exam_Score'].apply(lambda x: 1 if x >= 75 else 0)
X = df.drop(['Exam_Score', 'Exam_Score_Binary'], axis=1).values
y = df['Exam_Score_Binary'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Step 4: Establish Baseline with Dummy Classifier
dummy_stratified = DummyClassifier(strategy='stratified', random_state=42)
dummy_most_frequent = DummyClassifier(strategy='most_frequent')

# Fit and evaluate Dummy Classifiers
dummy_stratified.fit(X_train, y_train)
dummy_most_frequent.fit(X_train, y_train)

y_test_dummy_stratified = dummy_stratified.predict(X_test)
y_test_dummy_most_frequent = dummy_most_frequent.predict(X_test)

# Evaluate Dummy Classifiers
accuracy_dummy_stratified = accuracy_score(y_test, y_test_dummy_stratified)
accuracy_dummy_most_frequent = accuracy_score(y_test, y_test_dummy_most_frequent)

print(f"Dummy Classifier (Stratified) Accuracy: {accuracy_dummy_stratified:.4f}")
print(f"Dummy Classifier (Most Frequent) Accuracy: {accuracy_dummy_most_frequent:.4f}")
