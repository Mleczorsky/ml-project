from flask import Flask, render_template, request, jsonify
from models import models
# models = ['mock']

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
    print("Jebanie ruchanie obciąganie")
    print(request.data)
    # raise Exception(request.files["file"])
    file = request.files["file"]
    print(file)
    prediction = models["decision_tree"].predict(file)
    return jsonify([file, prediction, request.data]), 200

@app.route('/processing', methods=['GET'])
def chuj():
    print("Obciąganie")
    return "Ruchanie"

if __name__ == "__main__":
    app.run(debug=True)

    for model in models:
        print(f'Model: {model} loaded')