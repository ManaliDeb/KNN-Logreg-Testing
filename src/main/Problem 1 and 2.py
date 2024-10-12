import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# load dataset
df = pd.read_csv('src/main/resources/StudentPerformanceFactors.csv')

# preprocessing
category_col = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Internet_Access',
                'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Gender']
df = pd.get_dummies(df, columns=category_col, drop_first=True)

remaining_category_col = ['Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Distance_from_Home']
df = pd.get_dummies(df, columns=remaining_category_col, drop_first=True)

# convert exam score to binary
df['Exam_Score_Binary'] = df['Exam_Score'].apply(lambda x: 1 if x >= 75 else 0)

# features and labels
X = df.drop(['Exam_Score', 'Exam_Score_Binary'], axis=1)
y = df['Exam_Score_Binary']

# normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],  # choose solvers that support both penalties
    'penalty': ['l1', 'l2']
}

cv = StratifiedKFold(n_splits=3)
logreg = LogisticRegression(max_iter=1000)

# grid search with cross-validation
grid_search = GridSearchCV(logreg, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# evaluate on dev set
best_model = grid_search.best_estimator_
y_dev_pred = best_model.predict(X_dev)

# evaluation metrics
accuracy = accuracy_score(y_dev, y_dev_pred)
f1 = f1_score(y_dev, y_dev_pred, average='weighted')

print(f"Dev Set Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Best Parameters: {grid_search.best_params_}")

# problem 5
# evaluate on test set
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"Test Set Accuracy: {test_accuracy:.4f}")
print(f"Test Set F1 Score: {test_f1:.4f}")
