import cStringIO

from PIL import Image
from flask import Flask, request, json
import tensorflow as tf
from train import crack_captcha_cnn, convert2gray, X, keep_prop, vec2text
import base64
import numpy as np
import re


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

output = crack_captcha_cnn()
pre = tf.argmax(tf.reshape(output, [-1, 6, 36]), 2)
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "../zf_yzm_train/capcha.model-5200")


@app.route('/yzm', methods=['POST', 'GET'])
def test():
    img_base64 = request.form['img_base64']
    # return img_base64
    img_base64 = re.sub('^data:image/.+;base64,', '', img_base64)
    img_base64 = base64.b64decode(img_base64)
    buffer = cStringIO.StringIO(img_base64)
    img = Image.open(buffer)
    img = np.array(img)
    img = convert2gray(img)
    img = img.flatten() / 255
    text_list = sess.run(pre, feed_dict={X: [img], keep_prop: 1})
    text = text_list[0].tolist()
    vector = np.zeros(6 * 36)
    i = 0
    for n in text:
        vector[i * 36 + n] = 1
        i += 1
    text = vec2text(vector)
    res = {
        "code": 200,
        "data": text,
        "msg": None
    }
    return json.dumps(res)



if __name__ == '__main__':
    app.run()
