from flask import Flask
from models import models

app = Flask(__name__)

@app.route('/')
def hello_geek():
    list_of_models = [f'<li> {model} <li>' for model in models]
    return f'<h1>Hello from Flask & Docker</h1><ul>{list_of_models}</ul>'


if __name__ == "__main__":
    app.run(debug=True)

    for model in models:
        print(f'Model: {model} loaded')