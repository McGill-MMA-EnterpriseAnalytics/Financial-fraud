import os, json, gdown, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import sys
from sklearn.preprocessing import StandardScaler

# ---
# Downloading the datasets. 
# ---

def download_transactions_data():
    file_id = "1qLXHvCA8TUwTqHXkHedNX3E4VSN0f1Ox"
    output_file = "transactions_data.csv"
    if not os.path.exists(output_file):
        print(f"Downloading {output_file}. takeing from Google Drive")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_file, quiet=False)
        print("complete")
    else:
        print("skip download")

def load_transactions_data():
    path = "transactions_data.csv"
    try:
        df = pd.read_csv(path)
        print("Transactions Data Sample:")
        print(df.head())
        return df
    except FileNotFoundError:
        print("Error")
        return None

def load_mcc_codes():
    path = "mcc_codes.json"
    try:
        with open(path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(list(data.items()), columns=["MCC_Code", "Category"])
        print("\nMCC Codes Data Sample:")
        print(df.head())
        return df
    except FileNotFoundError:
        print("Error")
        return None

def load_fraud_labels():
    path = "train_fraud_labels.json"
    try:
        with open(path, "r") as f:
            data = json.load(f)
        target_data = data["target"]
        df = pd.DataFrame(target_data.items(), columns=["Transaction_ID", "Fraud_Label"])
        print("\nFraud Labels Data Sample:")
        print(df.head())
        return df
    except FileNotFoundError:
        print("Error")
        return None

def load_cards_data():
    path = "cards_data.csv"
    try:
        df = pd.read_csv(path)
        print("\nCards Data Sample:")
        print(df.head())
        return df
    except FileNotFoundError:
        print("Error")
        return None

def load_users_data():
    path = "users_data.csv"
    try:
        df = pd.read_csv(path)
        print("\nUsers Data Sample:")
        print(df.head())
        return df
    except FileNotFoundError:
        print("Error")
        return None
    
    

# ---
# Data Cleaning Section 
# ---

def clean_currency(series):
    return series.replace({'\$': '', ',': ''}, regex=True).astype(float)

def clean_transactions_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = clean_currency(df['amount'])
    return df

def clean_cards_data(df):
    df['acct_open_date'] = pd.to_datetime(df['acct_open_date'], format='%m/%Y', errors='coerce')
    df['expires'] = pd.to_datetime(df['expires'], format='%m/%Y', errors='coerce')
    df['credit_limit'] = clean_currency(df['credit_limit'])
    df['card_number'] = df['card_number'].astype(str)
    return df

def clean_users_data(df):
    df['per_capita_income'] = clean_currency(df['per_capita_income'])
    df['yearly_income'] = clean_currency(df['yearly_income'])
    df['total_debt'] = clean_currency(df['total_debt'])
    return df

# ---
# Perform some pre-modeling checks 
# ---

def check_duplicates(df, name):
    dups = df.duplicated().sum()
    print(f"[{name}] Duplicate rows: {dups}")

def check_consistency(trans_df, users_df, cards_df):
    missing_users = set(trans_df['client_id']) - set(users_df['client_id'])
    missing_cards = set(trans_df['card_id']) - set(cards_df['card_id'])
    if missing_users:
        print(f"Warning: {len(missing_users)} client_ids in Transactions missing from Users Data.")
    else:
        print("All client_ids in Transactions exist in Users Data.")
    if missing_cards:
        print(f"Warning: {len(missing_cards)} card_ids in Transactions missing from Cards Data.")
    else:
        print("All card_ids in Transactions exist in Cards Data.")

def detect_outliers_iqr(df, name):
    numeric_cols = df.select_dtypes(include=np.number).columns
    print(f"\n[{name}] Outlier Analysis (IQR Method):")
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        count = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        print(f"  {col}: {count} outliers (Bounds: {lower:.2f}, {upper:.2f})")

def scale_numeric_features(df, cols, name):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols] = scaler.fit_transform(df[cols])
    print(f"\n[{name}] Scaled features: {cols}")
    return df_scaled

# ---
# Merging the data
# ---

def merge_datasets(transactions_df, fraud_labels_df, cards_df, users_df, mcc_codes_df):
    # Rename key columns for clarity
    transactions_df = transactions_df.rename(columns={'id': 'transaction_id'})
    fraud_labels_df = fraud_labels_df.rename(columns={'Transaction_ID': 'transaction_id'})
    cards_df = cards_df.rename(columns={'id': 'card_id'})
    users_df = users_df.rename(columns={'id': 'client_id'})
    
    #  fraud_labels transaction_id is numeric
    fraud_labels_df['transaction_id'] = pd.to_numeric(fraud_labels_df['transaction_id'], errors='coerce')
    
    # Merge fraud labels into transactions 
    merged_df = transactions_df.merge(fraud_labels_df, on='transaction_id', how='left', indicator=True)
    print("\nMerge Indicator for Fraud Labels:")
    print(merged_df['_merge'].value_counts())
    merged_df['Fraud_Label'] = merged_df['Fraud_Label'].fillna("No")
    
    # Merge Cards and Users Data
    merged_df = merged_df.merge(cards_df, on='card_id', how='left', suffixes=('', '_card'))
    merged_df = merged_df.merge(users_df, on='client_id', how='left', suffixes=('', '_user'))
    
    # Merge MCC Codes Data
    mcc_codes_df['MCC_Code'] = pd.to_numeric(mcc_codes_df['MCC_Code'], errors='coerce')
    merged_df = merged_df.merge(mcc_codes_df, left_on='mcc', right_on='MCC_Code', how='left')
    merged_df = merged_df.drop(columns=['MCC_Code'])
    
    return merged_df

# ---
# Main execution to load, clean and execute 
# ---

if __name__ == "__main__":
    
    # --- Download and Load Data ---
    download_transactions_data()
    transactions_df = load_transactions_data()
    mcc_codes_df    = load_mcc_codes()
    fraud_labels_df = load_fraud_labels()
    cards_df        = load_cards_data()
    users_df        = load_users_data()
    
    # Correct check for failed dataset loads using identity check
    if any(x is None for x in [transactions_df, mcc_codes_df, fraud_labels_df, cards_df, users_df]):
        print("One or more datasets failed to load. Exiting.")
        sys.exit(1)
    
    # --- Data Cleaning ---
    transactions_df = clean_transactions_data(transactions_df)
    cards_df = clean_cards_data(cards_df)
    users_df = clean_users_data(users_df)
    print("Data cleaning complete.")
    
    # --- Pre-modeling Checks ---
    check_duplicates(transactions_df, "Transactions Data")
    check_duplicates(users_df, "Users Data")
    check_duplicates(cards_df, "Cards Data")
    detect_outliers_iqr(transactions_df, "Transactions Data")
    detect_outliers_iqr(users_df, "Users Data")
    detect_outliers_iqr(cards_df, "Cards Data")
    transactions_scaled = scale_numeric_features(transactions_df, ['amount'], "Transactions Data")
    
    # --- Merge Datasets ---
    merged_df = merge_datasets(transactions_df, fraud_labels_df, cards_df, users_df, mcc_codes_df)
    print("\nCombined dataset shape:", merged_df.shape)
    print("Combined dataset sample:")
    print(merged_df.head())
    
