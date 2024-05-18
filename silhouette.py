from sklearn.metrics import silhouette_score

# Đọc dữ liệu từ file CSV
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv('student_stress_factors.csv')
features = data[['Kindly Rate your Sleep Quality',
                 'How many times a week do you suffer headaches?',
                 'How would you rate you academic performance?',
                 'how would you rate your study load?',
                 'How many times a week you practice extracurricular activities ?',
                 'How would you rate your stress levels?']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=20, n_init='auto', init='k-means++')
kmeans.fit(scaled_features)

# Dự đoán nhãn cụm cho mỗi điểm dữ liệu
labels = kmeans.labels_

# Tính độ đo Silhouette
silhouette_avg = silhouette_score(scaled_features, labels)
print(f"Silhouette Score: {silhouette_avg}")
