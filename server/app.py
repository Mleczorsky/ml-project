import base64
import numpy as np
from PIL import (Image)
from flask import Flask, render_template, request, jsonify
from models import models
from extract_features import extract
# models = ['mock']

CLASS_NAMES = ['rock', 'paper', 'scissors']

app = Flask(__name__)

@app.route('/')
def root():
    list_of_models = "".join([f'<li> {model} <li>' for model in models])
    href = "<a href='./photo'> PHOTO </a>"
    return f'<h1>Hello from Flask & Docker</h1>{href}<ul>{list_of_models}</ul>'

@app.route('/photo')
def photo():
    return render_template('photo.html')


@app.route('/processing', methods=['POST'])
def process():
    # print("Jebanie ruchanie obciąganie")
    # print(request.data)
    # # raise Exception(request.files["file"])
    # # file = request.files["file"]
    # # print(file)
    # # prediction = models["decision_tree"].predict(file)
    # # return jsonify([file, prediction, request.data]), 200
    # return jsonify(list(request.files.keys())), 200

    file = request.files['file']
    with Image.open(file.stream) as img:
        img_cpy = img.copy()

    feautures = extract(images=[img_cpy])

    responses = { model: CLASS_NAMES[models[model].predict(feautures).flatten()[0]] for model in models }

    return jsonify(responses), 200


@app.route('/processing', methods=['GET'])
def chuj():
    print("Obciąganie")
    file = request.files['file']
    img = Image.open(file.stream)

    data = file.stream.read()
    # data = base64.encodebytes(data)
    data = base64.b64encode(data).decode()

    return jsonify({
        'msg': 'success',
        'size': [img.width, img.height],
        'format': img.format,
        'img': data
    })

if __name__ == "__main__":
    app.run(debug=True)

    for model in models:
        print(f'Model: {model} loaded')
