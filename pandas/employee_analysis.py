import pandas as pd

data = {
"name":["Ajith","Ram","John","Riya","Kumar"],
"age":[25,30,None,28,35],
"salary":[50000,60000,45000,None,70000],
"dept":["IT","HR","IT","FIN","IT"]
}

df = pd.DataFrame(data)

# HANDLE MISSING VALUES
print(f"df.isnull(): {df.isnull()}")
print(f"missing values in each column:\n {df.isnull().sum()}")


df["age"].fillna(df["age"].mean(), inplace=True)
df["salary"].fillna(0, inplace=True)
print(f"after handling missing values:\n {df}")

#Task 1: Average salary
print(f"Average salary:\n {df['salary'].mean()}")

#Task 2: Highest salary person
print(f"Person with highest salary:\n {df[df['salary'] == df['salary'].max()]['name'].values[0]}")

#Task 3: IT dept avg salary
print(f"Average salary in IT department:\n {df[df['dept'] == 'IT']['salary'].mean()}")

# Task 4: Employees older than 30
print(f"Employees older than 30:\n {df[df['age'] > 30]}")

# Task 5: Sort by salary
print(f"sort by salary:\n {df.sort_values(by='salary', ascending=False)}")

# Task 6: Number of employees in each department
print(f"Number of employees in each department:\n {df.groupby('dept')['name'].count()}")