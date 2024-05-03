import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

# Đọc dữ liệu từ các tệp csv
df1 = pd.read_csv("D:/KhaiPha/python/heart_statlog_cleveland_hungary_final.csv")
df2 = pd.read_csv("D:/KhaiPha/python/Data2.csv")
# Lọc trùng lặp trong df1 và df2
df1.drop_duplicates(inplace=True)
df2.drop_duplicates(inplace=True)

# Gộp các khung dữ liệu đã lọc trùng lặp vào df_combined
df_combined = pd.concat([df1, df2], ignore_index=True)

# Kiểm tra và xóa các bản ghi trùng nhau dựa trên tất cả các cột
df_combined.drop_duplicates(inplace=True)


# Lưu DataFrame đã được xử lý vào file CSV mới
df_combined.to_csv("D:/KhaiPha/python/diabetenew.csv", index=False)