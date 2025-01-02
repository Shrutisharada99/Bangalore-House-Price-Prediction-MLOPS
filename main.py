from pipelines.data_loader import data_loader
from pipelines.data_preprocessor import data_preprocessor
from pipelines.train_test_split import split_data
from pipelines.model_selection import model_selection
from pipelines.data_scaler import  data_scaler
import pickle

if __name__ == "__main__":
   # Step 1: Load Data
   file_path = "Bengaluru_House_Data.csv"
   data = data_loader(file_path)

   # Step 2: Preprocess Data
   X, y,feature_names = data_preprocessor(data)

   # Step 3: Train-Test Split
   X_train, X_test, y_train, y_test = split_data(X, y)

   # Step 4: One hot encoded data

   X_train_scaled,X_test_scaled,scaler=data_scaler(X_train, X_test)
   
   # Step 5: Model Selection
   best_model = model_selection(X_train_scaled, y_train, X_test_scaled, y_test)

   # Save the best model as a pickle file
   with open("best_model.pkl", "wb") as f:
      pickle.dump(best_model, f)

   with open("features.pkl", "wb") as f:
      pickle.dump(feature_names, f)

   with open("scaler.pkl", "wb") as f:
      pickle.dump(scaler, f)