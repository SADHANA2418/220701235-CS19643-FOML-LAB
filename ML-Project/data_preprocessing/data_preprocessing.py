import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data():
    main_data_path = 'C:/Users/admin/Desktop/foml/Hybrid-Energy-predictor/data/server_energy_logs_large.csv'  # Absolute path
    simulated_data_path = 'C:/Users/admin/Desktop/foml/Hybrid-Energy-predictor/data/simulated_real_time_data_with_correct_structure.csv'  # Absolute path

    # Load the datasets
    main_df = pd.read_csv(main_data_path)
    simulated_df = pd.read_csv(simulated_data_path)

    return main_df, simulated_df

def preprocess_data(df):
    # Ensure only numeric columns are processed for filling NaN values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Replace column names with underscores instead of spaces
    df.columns = df.columns.str.replace(" ", "_")

    # Map Fan_Status or FanStatus to numerical values
    if 'Fan_Status' in df.columns:
        df['Fan_Status'] = df['Fan_Status'].map({"Working": 1, "Degraded": 0, "Failed": -1})
    
    if 'FanStatus' in df.columns:
        df['FanStatus'] = df['FanStatus'].map({"Working": 1, "Degraded": 0, "Failed": -1})

    # Scale relevant features if they exist in the dataset
    features_to_scale = ['CPU_Usage', 'RAM_Usage', 'Internal_Temperature', 'External_Temperature', 'Energy_Usage']
    scaler = StandardScaler()

    # Check if the expected features for scaling exist in the dataset
    if all(feature in df.columns for feature in features_to_scale):
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    else:
        print("Warning: The following expected features are missing from the data:", [feature for feature in features_to_scale if feature not in df.columns])

    return df

def split_data(df):
    # Split into features (X) and target (y)
    X = df.drop(columns=['Date', 'Server_ID', 'Comments', 'Failure_Probability'])
    y = df['Failure_Probability']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    main_df, simulated_df = load_data()

    main_df = preprocess_data(main_df)
    simulated_df = preprocess_data(simulated_df)

    X_train, X_test, y_train, y_test = split_data(main_df)

    # Create a 'data_preprocessed' folder if it doesn't exist
    output_folder = 'C:/Users/admin/Desktop/foml/Hybrid-Energy-predictor/data_preprocessed'
    os.makedirs(output_folder, exist_ok=True)

    # Save the processed data
    main_df.to_csv(os.path.join(output_folder, 'main_dataset_processed.csv'), index=False)
    simulated_df.to_csv(os.path.join(output_folder, 'simulated_real_time_data_processed.csv'), index=False)

    print(f"Training Data Shape: {X_train.shape}, Test Data Shape: {X_test.shape}")
