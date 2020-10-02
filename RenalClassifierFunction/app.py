import json
import sklearn
import boto3
import os
import pickle

# import requests

s3 = boto3.client('s3')
s3_bucket = os.environ['s3_bucket']
model_name = os.environ['model_name']
temp_file_path = '/tmp/' + model_name

from sklearn.neighbors import KNeighborsClassifier

columns = ['age','bp','sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

def lambda_handler(event, context):

    body = event['body']
    data = json.loads(body)
    print(body)
    input = [data[col] for col in columns]
    # Download pickled model from S3 and unpickle
    s3.download_file(s3_bucket, model_name, temp_file_path)
    with open(temp_file_path, 'rb') as f:
        model = pickle.load(f)

    # Predict class
    prediction = model.predict([input])[0]

    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*", #Required for CORS support to work
            "Access-Control-Allow-Headers": "X-Requested-With,content-type",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Content-Type": "application/json",
            "X-Custom-Header": "application/json"
            },
        "body": json.dumps({
            "prediction": str(prediction),
            # "location": ip.text.replace("\n", "")
        }),
    }
