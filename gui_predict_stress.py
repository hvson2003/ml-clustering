import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('student_stress_factors.csv')
Data_train, Data_test = train_test_split(data, test_size=0.2, random_state=42)

scaler = StandardScaler()
Data_train_scaled = scaler.fit_transform(Data_train)

# Khởi tạo mô hình K-Means và huấn luyện trên tập huấn luyện
kmeans_train = KMeans(n_clusters=50, n_init='auto', init='k-means++')
kmeans_train.fit(Data_train)

# Dự đoán nhãn cho tập kiểm thử
cluster_labels_test = kmeans_train.predict(Data_test)

# Tạo hàm để dự đoán mức độ căng thẳng
def predict_stress():
    new_data = pd.DataFrame({
        'Kindly Rate your Sleep Quality': float(sleep_quality_entry.get()),
        'How many times a week do you suffer headaches?': float(headaches_entry.get()),
        'How would you rate you academic performance?': float(academic_performance_entry.get()),
        'how would you rate your study load?': float(study_load_entry.get()),
        'How many times a week you practice extracurricular activities ?': float(extracurricular_activities_entry.get()),
        'How would you rate your stress levels?': 0
    }, index=[0])

    # Chuẩn hóa dữ liệu mới
    scaled_new_data = scaler.transform(new_data)

    # Dự đoán nhãn cụm cho dữ liệu mới
    cluster_label = kmeans_train.predict(scaled_new_data)

    # Hiển thị kết quả
    result_label.config(text=f'Predicted Stress Level: Cluster {cluster_label[0] + 1}')

    # Đo độ chất lượng của phân cụm trên tập kiểm thử
    silhouette_test = silhouette_score(Data_test, cluster_labels_test)
    davies_bouldin_test = davies_bouldin_score(Data_test, cluster_labels_test)

    # Hiển thị độ đo trên giao diện
    silhouette_label.config(text=f'Silhouette Score (Test): {silhouette_test:.4f}')
    davies_bouldin_label.config(text=f'Davies-Bouldin Score (Test): {davies_bouldin_test:.4f}')

# Tạo giao diện
root = tk.Tk()
root.title('Stress Prediction')
sleep_quality_label = ttk.Label(root, text='Sleep Quality:')
sleep_quality_entry = ttk.Entry(root)

headaches_label = ttk.Label(root, text='Headaches per week:')
headaches_entry = ttk.Entry(root)

academic_performance_label = ttk.Label(root, text='Academic Performance Rating:')
academic_performance_entry = ttk.Entry(root)

study_load_label = ttk.Label(root, text='Study Load Rating:')
study_load_entry = ttk.Entry(root)

extracurricular_activities_label = ttk.Label(root, text='Extracurricular Activities per week:')
extracurricular_activities_entry = ttk.Entry(root)

# Button thực hiện dự đoán
predict_button = ttk.Button(root, text='Predict Stress', command=predict_stress)

# Label hiển thị kết quả
result_label = ttk.Label(root, text='Predicted Stress Level:')
silhouette_label = ttk.Label(root, text='Silhouette Score (Test):')
davies_bouldin_label = ttk.Label(root, text='Davies-Bouldin Score (Test):')

# Grid layout
sleep_quality_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')
sleep_quality_entry.grid(row=0, column=1, padx=5, pady=5)

headaches_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')
headaches_entry.grid(row=1, column=1, padx=5, pady=5)

academic_performance_label.grid(row=2, column=0, padx=5, pady=5, sticky='e')
academic_performance_entry.grid(row=2, column=1, padx=5, pady=5)

study_load_label.grid(row=3, column=0, padx=5, pady=5, sticky='e')
study_load_entry.grid(row=3, column=1, padx=5, pady=5)

extracurricular_activities_label.grid(row=4, column=0, padx=5, pady=5, sticky='e')
extracurricular_activities_entry.grid(row=4, column=1, padx=5, pady=5)

predict_button.grid(row=5, column=0, columnspan=2, pady=10)

result_label.grid(row=6, column=0, columnspan=2)
silhouette_label.grid(row=7, column=0, columnspan=2)
davies_bouldin_label.grid(row=8, column=0, columnspan=2)

root.mainloop()
