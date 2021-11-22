from flask import (
    Flask,
    redirect,
    render_template,
    request,
    g,
    session,
    url_for,
    Response,
    jsonify
)
import process

app = Flask(__name__)
app.secret_key = 'secretkey'


@ app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        process.create_csv('BTCUSDT')
        return render_template('graph.html')

    return render_template('index.html')


if __name__ == '__main__':
    # port = int(os.environ.get('PORT', 5000))
    # app.run(host='0.0.0.0', port=port)
    app.run(debug=True)
