import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.preprocessing import StandardScaler

#Đọc dữ liệu từ file CSV vào DataFrame
df = pd.read_csv("D:/KhaiPha/python/heart_statlog_cleveland_hungary_final.csv")


# Kiểm tra và xóa các bản ghi trùng nhau dựa trên tất cả các cột
df.drop_duplicates(inplace=True)


# Lưu DataFrame đã được xử lý vào file CSV mới
df.to_csv("D:/KhaiPha/python/diabetenew.csv", index=False)

# Đọc dữ liệu từ file CSV
data = pd.read_csv('D:/KhaiPha/python/diabetenew.csv')

# Phân chia dữ liệu thành features và target
X = data.drop('target', axis=1)
y = data['target']
# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Khởi tạo và huấn luyện mô hình MLP
model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter = 1000)
model.fit(X_train, y_train)

# In ra các trong số của mạng neural network
print("Các trọng số của mạng neural network:")
for i, coef in enumerate(model.coefs_):
    print(f"Layer {i}: {coef}")

# Lưu mô hình vào file
joblib.dump(model, 'D:/KhaiPha/python/neural_network_model.pkl')