import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

#Đọc dữ liệu từ file CSV vào DataFrame
data = pd.read_csv('D:/KhaiPha/python/diabetenew.csv')

# Phân chia dữ liệu thành features và target
X = data.drop('target', axis=1)
y = data['target']

# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# định hình lại dữ liệu CNN
X_train = X_train.values.reshape(-1, X_train.shape[1], 1)
X_test = X_test.values.reshape(-1, X_test.shape[1], 1)

# Khởi tạo và huấn luyện mô hình CNN
# Xây dựng mô hình CNN bằng TensorFlow
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Lưu mô hình vào file
joblib.dump(model, 'D:/KhaiPha/python/neural_networkCNN_model.pkl')
