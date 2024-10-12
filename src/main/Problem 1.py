import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

# load dataset
df = pd.read_csv('src/main/resources/StudentPerformanceFactors.csv')

# preprocessing
category_col = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Internet_Access',
                'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Gender']
df = pd.get_dummies(df, columns=category_col, drop_first=True)

remaining_category_col = ['Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Distance_from_Home']
df = pd.get_dummies(df, columns=remaining_category_col, drop_first=True)

# exam score target, feature and labels
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']

# normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# problem 2 improvements
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100]
    'solver': ['liblinear', 'lbfgs', 'saga']
    'penalty': ['l2']
}
# model train
logreg = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# eval on dev set
y_pred = model.predict(X_dev)

# eval metric based on dataset balance
accuracy = accuracy_score(y_dev, y_pred)
f1 = f1_score(y_dev, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
