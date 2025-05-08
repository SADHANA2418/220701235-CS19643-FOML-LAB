# import pandas as pd
# import joblib
# from tensorflow.keras.models import load_model

# # Load simulated real-time data
# real_time_data_path = 'data_preprocessed/simulated_real_time_data_processed.csv'
# real_time_df = pd.read_csv(real_time_data_path)

# # Load trained models
# ml_model = joblib.load('./ml_model.pkl')           # Load ML model
# dl_model = load_model('./dl_model.h5')             # Load DL model

# # Rename columns if needed (standardization step)
# column_mapping = {
#     'CPU_Usage': 'CPU_Usage(%)',
#     'RAM_Usage': 'RAM_Usage(%)',
#     'Energy_Usage': 'Energy_Usage(kWh)',
#     'External_Temperature': 'External_Temp(C)',
#     'Internal_Temperature': 'Internal_Temp(C)',
#     'FanStatus': 'Fan_Status',
#     'ServerID': 'Server_ID'
# }
# real_time_df.rename(columns=column_mapping, inplace=True)

# # One-hot encode the Server_ID (same as training time)
# real_time_df_encoded = pd.get_dummies(real_time_df, columns=['Server_ID'])

# # Ensure all expected columns from the training set are present in the real-time data
# expected_features = ml_model.feature_names_in_
# for col in expected_features:
#     if col not in real_time_df_encoded.columns:
#         real_time_df_encoded[col] = 0  # Add missing columns with default value

# # Reorder columns to match the model's expectations
# X_real_time = real_time_df_encoded[expected_features]

# # Predict using both models
# ml_preds = ml_model.predict(X_real_time)
# dl_preds = dl_model.predict(X_real_time).flatten()

# # Combine predictions (simple average ensemble)
# ensemble_preds = (ml_preds + dl_preds) / 2

# # Add predictions to the original dataframe
# real_time_df['Predicted_Failure_Probability'] = ensemble_preds

# # Display results
# print(real_time_df[['Date', 'Server_ID', 'Predicted_Failure_Probability']].head())


import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load simulated real-time data
real_time_data_path = 'data_preprocessed/simulated_real_time_data_processed.csv'
real_time_df = pd.read_csv(real_time_data_path)

# Load trained models
ml_model = joblib.load('./ml_model.pkl')           # Load ML model
dl_model = load_model('./dl_model.h5')             # Load DL model

# Rename columns if needed (standardization step)
column_mapping = {
    'CPU_Usage': 'CPU_Usage(%)',
    'RAM_Usage': 'RAM_Usage(%)',
    'Energy_Usage': 'Energy_Usage(kWh)',
    'External_Temperature': 'External_Temp(C)',
    'Internal_Temperature': 'Internal_Temp(C)',
    'FanStatus': 'Fan_Status',
    'ServerID': 'Server_ID'
}
real_time_df.rename(columns=column_mapping, inplace=True)

# One-hot encode the Server_ID (same as training time)
real_time_df_encoded = pd.get_dummies(real_time_df, columns=['Server_ID'])

# Ensure all expected columns from the training set are present in the real-time data
expected_features = ml_model.feature_names_in_
for col in expected_features:
    if col not in real_time_df_encoded.columns:
        real_time_df_encoded[col] = 0  # Add missing columns with default value

# Reorder columns to match the model's expectations
X_real_time = real_time_df_encoded[expected_features]

# Predict using both models
ml_preds = ml_model.predict(X_real_time)
dl_preds = dl_model.predict(X_real_time).flatten()

# Combine predictions (simple average ensemble)
ensemble_preds = (ml_preds + dl_preds) / 2

# Add predictions to the original dataframe
real_time_df['Predicted_Failure_Probability'] = ensemble_preds

# Display results in console
print(real_time_df[['Date', 'Server_ID', 'Predicted_Failure_Probability']].head())

# 📝 Save predictions to a CSV for UI
real_time_df.to_csv('web/live_predictions.csv', index=False)
