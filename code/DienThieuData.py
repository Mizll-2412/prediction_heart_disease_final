import pandas as pd
from sklearn.impute import SimpleImputer


dataframe = pd.read_csv('D:/New folder/Python/Data/DataBTL/ThieuDuLieu.csv', header=None, na_values=None)

print(dataframe)

n_miss_per_row = dataframe.isnull().sum(axis=1)


for i, count in enumerate(n_miss_per_row):
    print(f"Row{i+1}: {count} missing data")



imputer = SimpleImputer(strategy="most_frequent")
dataframe = imputer.fit_transform(dataframe)


dataframe = pd.DataFrame(dataframe)
dataframe = dataframe.drop_duplicates()

print('Data sau khi dien thieu va loc trung: ')
print(dataframe)

dataframe.to_csv("D:/New folder/Python/Data/DataBTL/test_dienthieu.csv", index=False)
