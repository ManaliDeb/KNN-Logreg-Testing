import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# load dataset
df = pd.read_csv('StudentPerformanceFactors.csv')

# preprocessing
category_col = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Internet_Access',
                'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Gender']
df = pd.get_dummies(df, columns=category_col)

# exam score target, feature and labels
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']

# normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
