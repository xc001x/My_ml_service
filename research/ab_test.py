import json # will be needed for saving preprocessing details 
import numpy as np # for data manipulation 
import pandas as pd # for data manipulation 
from sklearn.model_selection import train_test_split # will be used for data split 
import requests 

# load dataset 
df = pd.read_csv('https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv', skipinitialspace=True) 
x_cols = [c for c in df.columns if c != 'income'] 
# set input matrix and target column 
X = df[x_cols] 
y = df['income'] 

# data split train / test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1234) 

# use first 100 rows of test data for A/B test
print("Starting A/B test with 100 sample requests...")
for i in range(100): 
    input_data = dict(X_test.iloc[i]) 
    target = y_test.iloc[i] 
    r = requests.post("http://127.0.0.1:8000/api/v1/income_classifier/predict?status=ab_testing", input_data) 
    response = r.json() 
    # provide feedback 
    requests.put("http://127.0.0.1:8000/api/v1/mlrequests/{}".format(response["request_id"]), {"feedback": target}) 
    print(f"Request {i+1}/100 completed - True value: {target}")

print("\nA/B test complete! You can now stop the test using the stop_ab_test endpoint.")
