import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np

# Đọc dữ liệu từ file CSV
data = pd.read_csv('student_stress_factors.csv')

# Chọn các đặc trưng để sử dụng cho phân cụm
features = data[['Kindly Rate your Sleep Quality',
                 'How many times a week do you suffer headaches?',
                 'How would you rate you academic performance?',
                 'how would you rate your study load?',
                 'How many times a week you practice extracurricular activities ?',
                 'How would you rate your stress levels?']]

# Chuẩn hóa dữ liệu
scaled_features = (features - features.mean()) / features.std()

# Tính toán centroid và nhãn cụm
def kmeans(X, K, max_iters=100):
    centroids = X[np.random.choice(range(len(X)), K, replace=False)]
    for _ in range(max_iters):
        labels = assign_labels(X, centroids)
        new_centroids = update_centroids(X, labels, K)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

def assign_labels(X, centroids):
    labels = []
    for x in X.values:
        distances = np.linalg.norm(x - centroids, axis=1)
        label = np.argmin(distances)
        labels.append(label)
    return np.array(labels)

def update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))
    counts = np.zeros(K)
    for i, x in enumerate(X.values):
        cluster = labels[i]
        centroids[cluster] += x
        counts[cluster] += 1
    for i in range(K):
        if counts[i] > 0:
            centroids[i] /= counts[i]
    return centroids

# Initialize KMeans model
K = 5
labels, centroids = kmeans(scaled_features, K)

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử (90% - 10%)
def train_test_split(X, y, test_size=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(scaled_features.values, labels, test_size=0.1, random_state=42)

# Khởi tạo mô hình K-Means và huấn luyện trên tập huấn luyện
class KMeans:
    def __init__(self, n_clusters, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(range(len(X)), self.n_clusters, replace=False)]
        for _ in range(self.max_iters):
            labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X, labels)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        self.labels = labels

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=-1)
        return np.argmin(distances, axis=-1)

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=-1)
        return np.argmin(distances, axis=-1)

    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        counts = np.zeros(self.n_clusters)
        for i, x in enumerate(X):
            cluster = labels[i]
            centroids[cluster] += x
            counts[cluster] += 1
        for i in range(self.n_clusters):
            if counts[i] > 0:
                centroids[i] /= counts[i]
        return centroids

kmeans_train = KMeans(n_clusters=K, max_iters=100)
kmeans_train.fit(X_train)

# Dự đoán nhãn cho tập kiểm thử
y_pred = kmeans_train.predict(X_test)