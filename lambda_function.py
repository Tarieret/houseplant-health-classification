
import boto3
import json
import os
import time
from urllib.parse import unquote_plus

rekognition = boto3.client('rekognition')
s3 = boto3.client('s3')

MODEL_ARN = os.environ['MODEL_ARN']
BUCKET_NAME = os.environ['BUCKET_NAME']
PROJECT_ARN = os.environ['PROJECT_ARN']

def handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = unquote_plus(event['Records'][0]['s3']['object']['key'])
    print(f"Processing image: {key}")
    
    response = rekognition.describe_project_versions(
        ProjectArn=PROJECT_ARN,
        VersionNames=["v1"]
    )
    status = response['ProjectVersionDescriptions'][0]['Status']
    
    if status != 'RUNNING':
        print("Starting model...")
        rekognition.start_project_version(
            ProjectVersionArn=MODEL_ARN,
            MinInferenceUnits=1
        )
        while True:
            response = rekognition.describe_project_versions(
                ProjectArn=PROJECT_ARN,
                VersionNames=["v1"]
            )
            status = response['ProjectVersionDescriptions'][0]['Status']
            print(f"Model status: {status}")
            if status == 'RUNNING':
                break
            elif status == 'FAILED':
                raise Exception("Model failed to start")
            time.sleep(30)
    else:
        print("Model already running")
    
    try:
        s3_object = s3.get_object(Bucket=bucket, Key=key)
        image_bytes = s3_object['Body'].read()
        
        response = rekognition.detect_custom_labels(
            ProjectVersionArn=MODEL_ARN,
            Image={'Bytes': image_bytes},
            MinConfidence=50
        )
        
        if response['CustomLabels']:
            top_label = max(response['CustomLabels'], key=lambda x: x['Confidence'])
            result = {
                'image': key,
                'prediction': top_label['Name'],
                'confidence': top_label['Confidence']
            }
        else:
            result = {
                'image': key,
                'prediction': 'unknown',
                'confidence': 0
            }
        
        result_key = f"results/{key.split('/')[-1].replace('.jpeg', '.json')}"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=result_key,
            Body=json.dumps(result)
        )
        
        print(f"Result saved: {result}")
        return result
        
    finally:
        rekognition.stop_project_version(
            ProjectVersionArn=MODEL_ARN
        )
        print("Model stopped")
