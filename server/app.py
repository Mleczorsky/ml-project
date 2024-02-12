from flask import Flask, render_template
# from models import models
models = ['mock']

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
    file = request.files['image']

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