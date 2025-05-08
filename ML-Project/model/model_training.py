# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# def load_data():
#     main_data_path = '../data/main_dataset_processed.csv'
#     df = pd.read_csv(main_data_path)
#     return df

# def train_ml_model(X_train, y_train):
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_model.fit(X_train, y_train)
#     return rf_model

# def train_dl_model(X_train, y_train):
#     dl_model = Sequential()
#     dl_model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
#     dl_model.add(Dropout(0.2))
#     dl_model.add(Dense(32, activation='relu'))
#     dl_model.add(Dense(1))

#     dl_model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
#     dl_model.fit(X_train, y_train, epochs=50, batch_size=32)
#     return dl_model

# if __name__ == '__main__':
 
#     df = load_data()
#     X = df.drop(columns=['Date', 'ServerID', 'Comments', 'Failure_Probability'])
#     y = df['Failure_Probability']

#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     ml_model = train_ml_model(X_train, y_train)

#     dl_model = train_dl_model(X_train, y_train)

#     ml_predictions = ml_model.predict(X_test)
#     dl_predictions = dl_model.predict(X_test)

#     ml_mae = mean_absolute_error(y_test, ml_predictions)
#     dl_mae = mean_absolute_error(y_test, dl_predictions)

#     print(f"ML Model MAE: {ml_mae}")
#     print(f"DL Model MAE: {dl_mae}")

# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# def load_data():
#     main_data_path = '../data/main_dataset_processed.csv'
#     df = pd.read_csv(main_data_path)
#     return df

# def train_ml_model(X_train, y_train):
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_model.fit(X_train, y_train)
#     return rf_model

# def train_dl_model(X_train, y_train):
#     dl_model = Sequential()
#     dl_model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
#     dl_model.add(Dropout(0.2))
#     dl_model.add(Dense(32, activation='relu'))
#     dl_model.add(Dense(1))

#     dl_model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
#     dl_model.fit(X_train, y_train, epochs=50, batch_size=32)
#     return dl_model

# if __name__ == '__main__':
#     df = load_data()

#     # Drop columns safely
#     cols_to_drop = ['Date', 'ServerID', 'Comments', 'Failure_Probability']
#     X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
#     y = df['Failure_Probability']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     ml_model = train_ml_model(X_train, y_train)
#     dl_model = train_dl_model(X_train, y_train)

#     ml_predictions = ml_model.predict(X_test)
#     dl_predictions = dl_model.predict(X_test)

#     ml_mae = mean_absolute_error(y_test, ml_predictions)
#     dl_mae = mean_absolute_error(y_test, dl_predictions)

#     print(f"ML Model MAE: {ml_mae}")
#     print(f"DL Model MAE: {dl_mae}")


# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# def load_data():
#     main_data_path = '../data/main_dataset_processed.csv'
#     df = pd.read_csv(main_data_path)
#     return df

# def preprocess_data(df):
#     # Drop irrelevant columns
#     df = df.drop(columns=['Date', 'Comments'], errors='ignore')
    
#     # One-hot encode categorical columns
#     df = pd.get_dummies(df, columns=['ServerID'], drop_first=True)

#     # Separate features and target
#     X = df.drop(columns=['Failure_Probability'])
#     y = df['Failure_Probability']
#     return X, y

# def train_ml_model(X_train, y_train):
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_model.fit(X_train, y_train)
#     return rf_model

# def train_dl_model(X_train, y_train):
#     dl_model = Sequential()
#     dl_model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
#     dl_model.add(Dropout(0.2))
#     dl_model.add(Dense(32, activation='relu'))
#     dl_model.add(Dense(1))

#     dl_model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
#     dl_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
#     return dl_model

# if __name__ == '__main__':
#     df = load_data()
#     X, y = preprocess_data(df)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     ml_model = train_ml_model(X_train, y_train)
#     dl_model = train_dl_model(X_train, y_train)

#     ml_predictions = ml_model.predict(X_test)
#     dl_predictions = dl_model.predict(X_test).flatten()

#     ml_mae = mean_absolute_error(y_test, ml_predictions)
#     dl_mae = mean_absolute_error(y_test, dl_predictions)

#     print(f"ML Model MAE: {ml_mae}")
#     print(f"DL Model MAE: {dl_mae}")



# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# def load_data():
#     main_data_path = '../data/main_dataset_processed.csv'
#     df = pd.read_csv(main_data_path)
#     print(df.columns)
#     return df

# def preprocess_data(df):
#     # Check for required columns
#     if 'Server_ID' not in df.columns or 'Failure_Probability' not in df.columns:
#         raise KeyError("Missing required columns 'Server_ID' or 'Failure_Probability'")
    
#     # Drop irrelevant columns
#     df = df.drop(columns=['Date', 'Comments'], errors='ignore')
    
#     # One-hot encode categorical columns
#     df = pd.get_dummies(df, columns=['Server_ID'], drop_first=True)

#     # Separate features and target
#     X = df.drop(columns=['Failure_Probability'])
#     y = df['Failure_Probability']
    
#     # Remove rows with missing values
#     df = df.dropna()
    
#     return X, y

# def train_ml_model(X_train, y_train):
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_model.fit(X_train, y_train)
#     return rf_model

# def train_dl_model(X_train, y_train):
#     dl_model = Sequential()
#     dl_model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
#     dl_model.add(Dropout(0.2))
#     dl_model.add(Dense(32, activation='relu'))
#     dl_model.add(Dense(1))

#     dl_model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
#     dl_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
#     return dl_model

# if __name__ == '__main__':
#     df = load_data()
#     X, y = preprocess_data(df)

#     # Check the shape of the training data
#     print(f"X_train shape: {X.shape}")

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     ml_model = train_ml_model(X_train, y_train)
#     dl_model = train_dl_model(X_train, y_train)

#     ml_predictions = ml_model.predict(X_test).flatten()  # Flattening to ensure it's 1D
#     dl_predictions = dl_model.predict(X_test).flatten()

#     # ml_mae = mean_absolute_error(y_test, ml_predictions)
#     # dl_mae = mean_absolute_error(y_test, dl_predictions)

#     # print(f"ML Model MAE: {ml_mae}")
#     # print(f"DL Model MAE: {dl_mae}")

#     ml_mae = mean_absolute_error(y_test, ml_predictions)
#     dl_mae = mean_absolute_error(y_test, dl_predictions)

#     print(f"ML Model MAE: {ml_mae}")
#     print(f"DL Model MAE: {dl_mae}")

#     # Save ML model using joblib
#     import joblib
#     joblib.dump(ml_model, '../model/ml_model.pkl')

#     # Save DL model using Keras
#     dl_model.save('../model/dl_model.h5')

#     print("Both models have been saved successfully.")


import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

def load_data():
    main_data_path = '../data/main_dataset_processed.csv'
    df = pd.read_csv(main_data_path)
    print("Columns in dataset:", df.columns)
    return df

def preprocess_data(df):
    if 'Server_ID' not in df.columns or 'Failure_Probability' not in df.columns:
        raise KeyError("Missing required columns 'Server_ID' or 'Failure_Probability'")
    
    df = df.drop(columns=['Date', 'Comments'], errors='ignore')
    df = pd.get_dummies(df, columns=['Server_ID'], drop_first=True)
    
    df = df.dropna()
    X = df.drop(columns=['Failure_Probability'])
    y = df['Failure_Probability']
    
    return X, y

def train_ml_model(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_dl_model(X_train, y_train):
    dl_model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    dl_model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
    dl_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return dl_model

if __name__ == '__main__':
    df = load_data()
    X, y = preprocess_data(df)
    print(f"Feature shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training ML model...")
    ml_model = train_ml_model(X_train, y_train)

    print("Training DL model...")
    dl_model = train_dl_model(X_train, y_train)

    ml_predictions = ml_model.predict(X_test).flatten()
    dl_predictions = dl_model.predict(X_test).flatten()

    ml_mae = mean_absolute_error(y_test, ml_predictions)
    dl_mae = mean_absolute_error(y_test, dl_predictions)

    print(f"ML Model MAE: {ml_mae}")
    print(f"DL Model MAE: {dl_mae}")

    # Ensure model directory exists
    model_dir = './'
    os.makedirs(model_dir, exist_ok=True)

    # Save models
    joblib.dump(ml_model, os.path.join(model_dir, 'ml_model.pkl'))
    dl_model.save(os.path.join(model_dir, 'dl_model.h5'))

    print("✅ Both models have been saved successfully.")
