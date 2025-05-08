
# # import numpy as np
# # from sklearn.metrics import mean_absolute_error
# # from sklearn.ensemble import VotingRegressor

# # from model_training import train_ml_model, train_dl_model

# # def train_ensemble_model(X_train, y_train, X_test, y_test):
# #     ml_model = train_ml_model(X_train, y_train)
# #     dl_model = train_dl_model(X_train, y_train)

# #     ensemble_model = VotingRegressor(estimators=[('rf', ml_model), ('nn', dl_model)])
# #     ensemble_model.fit(X_train, y_train)

# #     predictions = ensemble_model.predict(X_test)

# #     ensemble_mae = mean_absolute_error(y_test, predictions)

# #     print(f"Ensemble Model MAE: {ensemble_mae}")
# #     return ensemble_model

# # if __name__ == '__main__':
 
# #     df = pd.read_csv('../data/main_dataset_processed.csv')
# #     X = df.drop(columns=['Date', 'ServerID', 'Comments', 'Failure_Probability'])
# #     y = df['Failure_Probability']
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
# #     ensemble_model = train_ensemble_model(X_train, y_train, X_test, y_test)

# # import pandas as pd
# # import numpy as np
# # from sklearn.metrics import mean_absolute_error
# # from sklearn.model_selection import train_test_split

# # from model_training import train_ml_model, train_dl_model

# # def train_and_evaluate_ensemble(X_train, y_train, X_test, y_test):
# #     print("Training ML model...")
# #     ml_model = train_ml_model(X_train, y_train)
    
# #     print("Training DL model...")
# #     dl_model = train_dl_model(X_train, y_train)

# #     print("Predicting...")
# #     ml_preds = ml_model.predict(X_test)
# #     dl_preds = dl_model.predict(X_test).flatten()

# #     ensemble_preds = (ml_preds + dl_preds) / 2

# #     ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
# #     print(f"Ensemble Model MAE (manual): {ensemble_mae}")

# #     return ml_model, dl_model

# # if __name__ == '__main__':
# #     df = pd.read_csv('../data/main_dataset_processed.csv')
# #     X = df.drop(columns=['Date', 'Server_ID', 'Comments', 'Failure_Probability'])
# #     y = df['Failure_Probability']
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #     ml_model, dl_model = train_and_evaluate_ensemble(X_train, y_train, X_test, y_test)


# import os
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# import joblib

# from model_training import train_ml_model, train_dl_model

# def train_and_evaluate_ensemble(X_train, y_train, X_test, y_test):
#     print("Training ML model...")
#     ml_model = train_ml_model(X_train, y_train)
    
#     print("Training DL model...")
#     dl_model = train_dl_model(X_train, y_train)

#     print("Predicting...")
#     ml_preds = ml_model.predict(X_test)
#     dl_preds = dl_model.predict(X_test).flatten()

#     ensemble_preds = (ml_preds + dl_preds) / 2
#     ensemble_mae = mean_absolute_error(y_test, ensemble_preds)

#     print(f"Ensemble Model MAE (manual): {ensemble_mae}")
#     return ml_model, dl_model

# if __name__ == '__main__':
#     df = pd.read_csv('../data/main_dataset_processed.csv')

#     # Drop irrelevant columns and prepare features/target
#     df = df.drop(columns=['Date', 'Server_ID', 'Comments'], errors='ignore')
#     df = df.dropna()

#     X = df.drop(columns=['Failure_Probability'])
#     y = df['Failure_Probability']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     ml_model, dl_model = train_and_evaluate_ensemble(X_train, y_train, X_test, y_test)

#     # Ensure the directory exists
#     os.makedirs('../model', exist_ok=True)

#     # Save models
#     joblib.dump(ml_model, '../model/ensemble_ml_model.pkl')
#     dl_model.save('../model/ensemble_dl_model.h5')

#     print("✅ Ensemble models have been saved successfully.")


import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

from model_training import train_ml_model, train_dl_model

def train_and_evaluate_ensemble(X_train, y_train, X_test, y_test):
    print("Training ML model...")
    ml_model = train_ml_model(X_train, y_train)
    
    print("Training DL model...")
    dl_model = train_dl_model(X_train, y_train)

    print("Predicting...")
    ml_preds = ml_model.predict(X_test)
    dl_preds = dl_model.predict(X_test).flatten()

    ensemble_preds = (ml_preds + dl_preds) / 2
    ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
    print(f"Ensemble Model MAE (manual): {ensemble_mae}")

    return ml_model, dl_model

if __name__ == '__main__':
    # Load and preprocess the data
    df = pd.read_csv('./data_preprocessed/main_dataset_processed.csv')
    X = df.drop(columns=['Date', 'Server_ID', 'Comments', 'Failure_Probability'], errors='ignore')
    y = df['Failure_Probability']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train ensemble models
    ml_model, dl_model = train_and_evaluate_ensemble(X_train, y_train, X_test, y_test)
    
    # Save models in the current directory
    try:
        joblib.dump(ml_model, './ensemble_ml_model.pkl')
        dl_model.save('./ensemble_dl_model.h5')
        print(" Ensemble models saved successfully in current directory.")
    except Exception as e:
        print(f" Error saving models: {e}")
