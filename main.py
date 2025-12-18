import os
import joblib 
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribs, cat_attribs):
    # For numerical columns
    num_pipline = Pipeline([
        ("scaler", StandardScaler())
    ])

    # For categorical columns
    cat_pipline = Pipeline([ 
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Construct the full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipline, num_attribs), 
        ('cat', cat_pipline, cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Lets train the model
    full_db = pd.read_csv('Social_Media_Advertising.csv')
    shorting_data = full_db.drop(["Campaign_ID", "Location","Language", "Date", "Company", "Target_Audience", "Campaign_Goal", "Duration"], axis=1)

    # Clean currency columns
    currency_cols = ["Acquisition_Cost"]

    for col in currency_cols:
        if col in shorting_data.columns:
            shorting_data[col] = (
                shorting_data[col]
                .astype(str)
                .str.replace(r'[$,]', '', regex=True)
                .astype(float)
            )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(shorting_data, shorting_data['Customer_Segment']):
        shorting_data.loc[test_index].to_csv("input.csv", index=False) 
        training_data = shorting_data.loc[train_index]  
    
    data_labels = training_data["ROI"].copy()
    data_features = training_data.drop("ROI", axis=1)

    num_attribs = data_features.drop(["Channel_Used","Customer_Segment"], axis=1).columns.tolist()
    cat_attribs = ["Channel_Used","Customer_Segment"]

    print("Training the model...")
    pipeline = build_pipeline(num_attribs, cat_attribs) 
    data_prepared = pipeline.fit_transform(data_features)
    print("Data is prepared")

    model = LinearRegression()
    model.fit(data_prepared, data_labels)
    print("Model is trained")

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is trained. Congrats!")
else:
    # Lets do inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')
    input_data.drop("ROI", axis=1, inplace=True)
    input_data.to_csv("test_input.csv", index=False)
    test_input = pd.read_csv('test_input.csv')

    transformed_input = pipeline.transform(test_input)
    predictions = model.predict(transformed_input)
    input_data['ROI'] = predictions

    input_data.to_csv("output.csv", index=False)
    print("Inference is complete, results saved to output.csv Enjoy!")

    #check error
    input_value = pd.read_csv("input.csv")["ROI"]
    output_value = pd.read_csv("output.csv")["ROI"]
    random_forest_rmse = root_mean_squared_error(input_value, output_value)
    print(f"The root mean squared error for Linear Regression is {random_forest_rmse}") 