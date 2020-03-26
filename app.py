import boto3
from botocore.config import Config
from flask import Flask, request, redirect, jsonify, render_template
from flask_cors import CORS
import os
import uuid
from gevent.pywsgi import WSGIServer
from chexnet.chexnet import Xray
from util import base64_to_pil, np_to_base64, base64_to_bytes
import numpy as np

S3_ENDPOINT_URL = os.environ['S3_ENDPOINT_URL'] if 'S3_ENDPOINT_URL' in os.environ else 'http://192.168.2.251:9000'
S3_ACCESS_KEY = os.environ['S3_ACCESS_KEY'] if 'S3_ACCESS_KEY' in os.environ else 'minioadmin'
S3_SECRET_KEY = os.environ['S3_SECRET_KEY'] if 'S3_SECRET_KEY' in os.environ else 'minioadmin'
BUCKET_NAME = os.environ['BUCKET_NAME'] if 'BUCKET_NAME' in os.environ else 'covid'

app = Flask(__name__)
CORS(app)

s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    verify=False
)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = base64_to_pil(request.json)
    except Exception:
        return jsonify(error="expected string or bytes-like object"), 422

    rgb_image_np = np.array(img.convert('RGB'))
    stds = rgb_image_np.std(axis=(0, 1))
    means = rgb_image_np.mean(axis=(0, 1))
    if 35 < np.average(stds) < 95 and 60 < np.average(means) < 180:
        img_result = x_ray.predict(img)
        file_name = "%s/%s/%s(%s).jpg" % (
            img_result['result'],
            img_result['type'],
            str(uuid.uuid4()),
            str(round(img_result['probability'], 2)).replace('.', '_')
        )
        s3.upload_fileobj(
            base64_to_bytes(request.json),
            BUCKET_NAME,
            file_name,
            ExtraArgs={
                "ACL": 'public-read',
                "ContentType": "image/jpg"
            }
        )
        return jsonify(
            result=img_result['result'],
            type=img_result['type'],
            probability=str(round(img_result['probability'], 2)) + "%"
        ), 200
    else:
        file_name = "NOT_DETECTED/%s.jpg" % str(uuid.uuid4())
        s3.upload_fileobj(
            base64_to_bytes(request.json),
            BUCKET_NAME,
            file_name,
            ExtraArgs={
                "ACL": 'public-read',
                "ContentType": "image/jpg"
            }
        )
        return jsonify(
            result='NOT DETECTED',
            error='Invalid image'
        ), 200


if __name__ == '__main__':
    x_ray = Xray()
    http_server = WSGIServer(('0.0.0.0', 5000), app, log=None)
    http_server.serve_forever()
