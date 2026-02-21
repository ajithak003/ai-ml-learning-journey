import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer


csv_path = Path(__file__).resolve().parents[1] / "data" / "car_prices.csv"
df = pd.read_csv(csv_path)

print(f"info: {df.info()}")
print(f"describe: {df.describe()}")

# Task 1: clean null values
print(f"missing values in each column:\n {df.isnull().sum()}")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent.sort_values(ascending=False))

# Very small null count → Drop rows
df.dropna(subset=["vin","sellingprice", "saledate"], inplace=True)

# Important categorical columns → Use "Unknown"
df["make"].fillna("Unknown", inplace=True)
df["model"].fillna("Unknown", inplace=True)
df["trim"].fillna("Unknown", inplace=True)
print(df.isnull().sum())

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"numerical columns: {numerical_cols}")
print(f"categorical columns: {categorical_cols}")

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
print(f"after handling missing values:\n {df.isnull().sum()}")

# Task 2: cleaning
print(f"before cleaning:\n {df.head()}")
print(df.dtypes)

#convert datatypes
df["year"] = df["year"].astype(int)
df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce")
print(f"after converting datatypes:\n {df.dtypes}")

#remove duplicates
print(f"before removing duplicates: {len(df)}")
df.drop_duplicates(inplace=True)
print(f"after removing duplicates: {len(df)}")

#check for outliers in sellingprice
print(f"sellingprice describe:\n {df['sellingprice'].describe()}")

df = df[df["sellingprice"] > 1000]

Q1 = df["sellingprice"].quantile(0.25)
Q3 = df["sellingprice"].quantile(0.75)
print(f"Q1: {Q1}, Q3: {Q3}")

IQR = Q3 - Q1
print(f"IQR: {IQR}")

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print(f"lower: {lower}, upper: {upper}")

df = df[(df["sellingprice"] >= lower) & (df["sellingprice"] <= upper)]
print(f"after removing outliers:\n {df['sellingprice'].describe()}")

# Save cleaned data
output_path = Path(__file__).resolve().parents[1] / "data" / "cleaned_car_data.csv"
df.to_csv(output_path, index=False)

#Task 3: Show Summary statistics
print(f"summary statistics:\n {df.describe(include='all')}")

#Task 4: Group By
print(f"average selling price by make:\n {df.groupby('make')['sellingprice'].mean().sort_values(ascending=False)}")
print(f"count of cars by make:\n {df.groupby('make')['make'].count().sort_values(ascending=False)}")
print(f"count of cars by color:\n {df.groupby('color')['sellingprice'].count().sort_values(ascending=False)}")
print(f"average selling price by make and transmission:\n {df.groupby(['make', 'transmission'])['sellingprice'].mean()}")
print(f"average selling price by state:\n {df.groupby('state')['sellingprice'].mean()}")