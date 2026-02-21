import pandas as pd

data = {
"name":["Ajith","Ram","John","Riya","Kumar"],
"age":[25,30,None,28,35],
"salary":[50000,60000,45000,None,70000],
"dept":["IT","HR","IT","FIN","IT"]
}

df = pd.DataFrame(data)

#PANDAS FUNDAMENTALS - Create & explore DataFrame
print(f"header: {df.head()}")
print(f"tail: {df.tail()}")
print(f"shape: {df.shape}")
print(f"df.columns.tolist(): {df.columns.tolist()}")
print(f"df.info(): {df.info()}")
print(f"df.describe(): {df.describe()}")

#SELECT & FILTER DATA
print(f"selected single column: {df['name']}")
print(f"selected multiple columns: {df[['name', 'age']]}")
print(f"filtered rows for IT department: {df[df['dept'] =='IT']}")
print(f"filtered rows for age > 28: {df[df['age'] > 28]}")
print(f"filtered rows for salary > 50000 and dept == IT: {df[(df['salary'] > 50000) & (df['dept'] == 'IT')]}")

# HANDLE MISSING VALUES
print(f"df.isnull(): {df.isnull()}")
print(f"missing values in each column: {df.isnull().sum()}")


df["age"].fillna(df["age"].mean(), inplace=True)
df["salary"].fillna(0, inplace=True)
df.dropna(inplace=True)
print(f"after handling missing values: {df}")

#NEW COLUMN CREATION
df["Senior"] = df["age"].apply(lambda x: "Yes" if x > 30 else "N0")
print(f"after creating new column: {df}")

#GROUPBY
print(f"Average salary by department: {df.groupby('dept')['salary'].mean()}")
print(f"Count of employees by department: {df.groupby('dept')['name'].count()}")

#SORTING
print(f"df sorted by salary in descending order: {df.sort_values(by='salary', ascending=False)}")