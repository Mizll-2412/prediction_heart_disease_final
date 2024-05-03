import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np


# Tải mô hình được đào tạo trước
neural_network_model = joblib.load('D:/KhaiPha/python/neural_network_model.pkl')


# Chức năng lấy giá trị thanh trượt
def get_slider_value(slider, var):
    value = float(slider.get())
    var.set(f"{value}")


# Chức năng dự đoán bệnh tim dựa vào thông tin đầu vào của người dùng
def predict_diabetes_disease():
    age = int(age_entry.get())
    sex = int(sex_entry.get())
    chest_pain_type = int(chest_pain_entry.get())
    resting_bp_s = int(resting_bp_s_entry.get())
    cholesterol = int(cholesterol_entry.get())
    fasting_blood_sugar = int(fasting_blood_sugar_entry.get())
    resting_ecg = int(resting_ecg_entry.get())
    max_heart_rate = int(max_heart_rate_entry.get())
    exercise_angina = int(exercise_angina_entry.get())
    oldpeak = float(oldpeak_entry.get())
    ST_slope = int(ST_slope_entry.get())

    input_data = np.array([[age, sex, chest_pain_type, resting_bp_s, cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope]])
    prediction = neural_network_model.predict(input_data)

    result_label.config(text=f'Kết Quả Dự Đoán: {"Bị bệnh tim" if prediction == 1 else "Không bị bệnh tim"}')


# Chức năng thiết lập lại biểu mẫu
def reset_form():
    result_label.config(text='Kết Quả Dự Đoán: ')
    age_entry.delete(0, 'end')
    sex_entry.delete(0, 'end')
    chest_pain_entry.delete(0, 'end')
    resting_bp_s_entry.delete(0, 'end')
    cholesterol_entry.delete(0, 'end')
    fasting_blood_sugar_entry.delete(0, 'end')
    resting_ecg_entry.delete(0, 'end')
    max_heart_rate_entry.delete(0, 'end')
    exercise_angina_entry.delete(0, 'end')
    oldpeak_entry.delete(0, 'end')
    ST_slope_entry.delete(0, 'end')


root = tk.Tk()
root.title("Dự đoán bệnh tim")


# Widget để nhập dữ liệu
age_label = ttk.Label(root, text="Tuổi:")
age_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
age_entry = ttk.Entry(root)
age_entry.grid(row=0, column=1, padx=10, pady=5)

sex_label = ttk.Label(root, text="Giới tính(Nữ-0, Nam-1):")
sex_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
sex_entry = ttk.Entry(root)
sex_entry.grid(row=1, column=1, padx=10, pady=5)


chest_pain_label = ttk.Label(root, text="Loại đau ngực(1-4):")
chest_pain_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
chest_pain_entry = ttk.Entry(root)
chest_pain_entry.grid(row=2, column=1, padx=10, pady=5)


resting_bp_s_label = ttk.Label(root, text="Huyết áp nghỉ(mm/Hg):")
resting_bp_s_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
resting_bp_s_entry = ttk.Entry(root)
resting_bp_s_entry.grid(row=3, column=1, padx=10, pady=5)


cholesterol_label = ttk.Label(root, text="Cholesterol(mg/dl):")
cholesterol_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
cholesterol_entry = ttk.Entry(root)
cholesterol_entry.grid(row=4, column=1, padx=10, pady=5)


fasting_blood_sugar_label = ttk.Label(root, text="Đường huyết nhanh(1 nếu > 120 mg/dl, 0 nếu không):")
fasting_blood_sugar_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
fasting_blood_sugar_entry = ttk.Entry(root)
fasting_blood_sugar_entry.grid(row=5, column=1, padx=10, pady=5)


resting_ecg_label = ttk.Label(root, text="Điện tâm đồ nghỉ(0-2):")
resting_ecg_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
resting_ecg_entry = ttk.Entry(root)
resting_ecg_entry.grid(row=6, column=1, padx=10, pady=5)


max_heart_rate_label = ttk.Label(root, text="Tần số tim tối đa:")
max_heart_rate_label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
max_heart_rate_entry = ttk.Entry(root)
max_heart_rate_entry.grid(row=7, column=1, padx=10, pady=5)


exercise_angina_label = ttk.Label(root, text="Đau ngực do tập thể dục (1 nếu có, 0 nếu không):")
exercise_angina_label.grid(row=8, column=0, padx=10, pady=5, sticky="w")
exercise_angina_entry = ttk.Entry(root)
exercise_angina_entry.grid(row=8, column=1, padx=10, pady=5)


oldpeak_label = ttk.Label(root, text="Giảm đỉnh ST gây ra bởi tập thể dục so với nghỉ ngơi:")
oldpeak_label.grid(row=9, column=0, padx=10, pady=5, sticky="w")
oldpeak_entry = ttk.Entry(root)
oldpeak_entry.grid(row=9, column=1, padx=10, pady=5)


ST_slope_label = ttk.Label(root, text="Giai đoạn ST của đỉnh tập thể dục:")
ST_slope_label.grid(row=10, column=0, padx=10, pady=5, sticky="w")
ST_slope_entry = ttk.Entry(root)
ST_slope_entry.grid(row=10, column=1, padx=10, pady=5)

# Reset button
reset_button = ttk.Button(root, text="Reset", command=reset_form)
reset_button.grid(row=11, column=0, columnspan=2, padx=10, pady=5, sticky="we")

# Prediction button
predict_button = ttk.Button(root, text="Submit", command=predict_diabetes_disease)
predict_button.grid(row=12, column=0, columnspan=2, padx=10, pady=5, sticky="we")

# Result label
result_label = ttk.Label(root, text="Kết Quả Dự Đoán: ")
result_label.grid(row=13 ,column=0, columnspan=2, padx=10, pady=5, sticky="we")

root.mainloop()