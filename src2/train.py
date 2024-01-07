from flaml import AutoML
from flaml.data import get_output_from_log
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import json
import numpy as np



##Leaving space here to load text data
file_path = 'sampledata_embeddings.xlsx'
data = pd.read_excel(file_path)





#X,y
target_column = 'target'
X=data.drop(columns=[target_column])
y=data[target_column]
y



###############################################

#Before test-train split, we need to reduce the number of vector embedding dimensions
#There are multiple ways such as UMAP, PCA etc 


#################################################


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

automl = AutoML()   
automl_settings = {
    "time_budget": 1000,
    "metric": 'roc_auc',
    "task": 'classification',
    "log_file_name": 'automl.log',
    "model_history": True,
}


# Train the model using the data
automl.fit(X_train=X_train, y_train=y_train, dataframe=data, **automl_settings)

# Save the best model for deployment
best_model = automl.model
best_score = automl.best_loss
print(best_model)
print(best_score)
print(automl.model.estimator)



##Leaving space for leading test data


##########################################

predictions = best_model.predict(X_test)
probabilities = automl.predict_proba(X_test)
probabilities = probabilities[:,1]