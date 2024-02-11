from flask import Flask
from models import models

app = Flask(__name__)

@app.route('/')
def hello_geek():
    return '<h1>Hello from Flask & Docker</h2>'


if __name__ == "__main__":
    app.run(debug=True)

    for model in models:
        print(f'Model: {model} loaded')