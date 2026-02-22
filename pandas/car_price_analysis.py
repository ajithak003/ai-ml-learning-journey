import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer


def load_data():
    """Load car data from CSV file."""
    try:
        data_path = Path(__file__).resolve().parents[1] / "data" / "car_prices.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records from {data_path.name}")
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    

def clean_data(df):
    """Clean and preprocess the data."""
    print("\n--- Dataset Info ---")
    df.info()
    print(f"\n--- Describe statistics ---")
    print(df.describe(include="all"))

    # Clean null values
    print(f"\n--- Missing values in each column ---\n{df.isnull().sum()}")
    missing_percent = (df.isnull().sum() / len(df)) * 100
    print(f"\nMissing values percentage:\n{missing_percent.sort_values(ascending=False)}")

    # Very small null count → Drop rows
    df = df.dropna(subset=["vin", "sellingprice", "saledate"])

    # Important categorical columns → Use "Unknown"
    df["make"] = df["make"].fillna("Unknown")
    df["model"] = df["model"].fillna("Unknown")
    df["trim"] = df["trim"].fillna("Unknown")
    print(f"\nNull values after filling:\n{df.isnull().sum()}")

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"\nNumerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")

    # Impute missing values
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    print(f"\nAfter handling missing values:\n{df.isnull().sum()}")

    # Convert datatypes
    df["year"] = df["year"].astype(int)
    df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce")
    print(f"\nAfter converting datatypes:\n{df.dtypes}")

    # Remove duplicates
    print(f"\nBefore removing duplicates: {len(df)} rows")
    df = df.drop_duplicates()
    print(f"After removing duplicates: {len(df)} rows")

    return df

    
def remove_outliers(df):
    """Remove outliers from the sellingprice column using IQR method."""
    print("\n--- Outlier Detection (IQR Method) ---")
    print(f"Before removal:\n{df['sellingprice'].describe()}")

    # Remove unrealistic selling prices
    df = df[df["sellingprice"] > 1000]

    # Calculate IQR
    Q1 = df["sellingprice"].quantile(0.25)
    Q3 = df["sellingprice"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    print(f"\nQ1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Lower bound: {lower:.2f}, Upper bound: {upper:.2f}")

    # Filter outliers
    df = df[(df["sellingprice"] >= lower) & (df["sellingprice"] <= upper)]
    print(f"\nAfter removal:\n{df['sellingprice'].describe()}")
    
    return df


def analyze_data(df):
    """Analyze and print summary statistics of the data."""
    print("\n--- Data Analysis ---")
    print(f"Average selling price: ${df['sellingprice'].mean():.2f}")
    print(f"Median selling price: ${df['sellingprice'].median():.2f}")
    print(f"Most common make: {df['make'].mode()[0]}")
    print(f"Most common model: {df['model'].mode()[0]}")
    print(f"Most common color: {df['color'].mode()[0]}")


def main():
    """Main function to orchestrate the data pipeline."""
    df = load_data()
    df = clean_data(df)
    df = remove_outliers(df)

    # Show Summary statistics
    print(f"\n--- Summary Statistics ---\n{df.describe(include='all')}")

    # Save cleaned data
    output_path = Path(__file__).resolve().parents[1] / "data" / "cleaned_car_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path.name}")

    analyze_data(df)


if __name__ == "__main__":
    main()