from sklearn.metrics import davies_bouldin_score

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

# Khởi tạo mô hình K-Means
kmeans = KMeans(n_clusters=50, n_init='auto', init='k-means++')
kmeans.fit(scaled_features)

# Dự đoán nhãn cụm cho mỗi điểm dữ liệu
labels = kmeans.labels_

# Tính độ đo Davies-Bouldin
davies_bouldin = davies_bouldin_score(scaled_features, labels)
print(f"Davies-Bouldin Score: {davies_bouldin}")
