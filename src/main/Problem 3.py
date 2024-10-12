import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # distances between x and all examples
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # k nearest neighbor indices
        k_indices = np.argsort(distances)[:self.k]
        # labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    # load dataset
    df = pd.read_csv('src/main/resources/StudentPerformanceFactors.csv')

    # preprocessing (using previous problem)
    category_col = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                    'Internet_Access', 'School_Type', 'Peer_Influence',
                    'Learning_Disabilities', 'Parental_Education_Level', 'Gender']
    df = pd.get_dummies(df, columns=category_col, drop_first=True)

    # prep features and target
    X = df.drop('Exam_Score', axis=1).values
    y = df['Exam_Score'].values

    # normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
