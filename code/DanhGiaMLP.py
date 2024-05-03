import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report,confusion_matrix
from joblib import load

# Đường dẫn tới tệp dữ liệu test
data_test_path = "D:/KhaiPha/python/data_test.csv"

# Phân chia dữ liệu test thành features và target
data_test = pd.read_csv(data_test_path)
X_test = data_test.drop(columns=['target'])
y_test = data_test['target']

# Load mô hình từ file
model_path = "D:/KhaiPha/python/neural_network_model.pkl"
model = load(model_path)

# Dự đoán nhãn cho dữ liệu test
y_pred = model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Tính toán các thông số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# In kết quả đánh giá
print("Confusion Matrix:")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


class_report = classification_report(y_test,y_pred)
print("Classification Report: ")
print(class_report)