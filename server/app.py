import base64
import numpy as np
from PIL import (Image)
from flask import Flask, render_template, request, jsonify
from models import models
from extract_features import extract
from validate import validate



CLASS_NAMES = ['rock', 'paper', 'scissors']

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024     # 16MB

@app.route('/')
def root():
    return render_template('app.html')



@app.route('/process', methods=['POST', 'GET'])
def process():
    if 'file' not  in request.files:
        return jsonify({'error': 'No file found'}, 404)
    file = request.files['file']
    if not validate(file):
        return jsonify({'error': 'Invalid file'}, 403)
    with Image.open(file.stream) as img:
        img_cpy = img.copy()

    features = extract(images=[img_cpy])

    responses = { model: CLASS_NAMES[models[model].predict(features).flatten()[0]] for model in models }

    return jsonify(responses), 200


if __name__ == "__main__":
    app.run(debug=False)

