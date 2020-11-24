from PIL import Image
import os
import random
import tensorflow as tf
import numpy as np

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 50
MAX_CAPTHCA = 6


char_set = ['0','1','2','3','4','5','6','7','8','9'] + ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
MAX_Len = len(char_set)


def get_text_and_image():
    all_image = os.listdir("../yzm");
    random_file = random.randint(1, 70000)
    base = all_image[random_file][:4]
    image = Image.open("../yzm/%s"% all_image[random_file])
    image = np.array(image)
    # print "text: %s" %base
    return base, image

def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img

def text2vec(name):
    vector = np.zeros(MAX_CAPTHCA * MAX_Len)
    def char2post(c):
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 87

        # print k
        return k
    for i, c in enumerate(name):
        idx = i * MAX_Len + char2post(c)
        # print idx
        vector[idx] = 1
    return vector


def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i
        char_idx = c % MAX_Len
        # print char_idx
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx + 87
        text.append(chr(char_code))
    return "".join(text)

def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTHCA * MAX_Len])
    for i in range(batch_size):
        text, image = get_text_and_image()
        image = convert2gray(image)
        batch_x[i, :] =  image.flatten() / 255
        # print batch_x
        batch_y[i, :] = text2vec(text)
        # print batch_y
    return batch_x, batch_y

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTHCA * MAX_Len])
keep_prop = tf.placeholder(tf.float32)

def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prop)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prop)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prop)

    w_d = tf.Variable(w_alpha * tf.random_normal([7 * 25 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prop)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTHCA * MAX_Len]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_Len * MAX_CAPTHCA]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out

def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTHCA, MAX_Len])
    # pre = tf.argmax(tf.reshape(output, [-1, MAX_CAPTHCA, MAX_Len]), 2)

    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTHCA, MAX_Len]), 2)
    corrext_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(corrext_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prop: 0.75})

            print step, loss_, _
            if step % 100 == 0:
                batch_x_test, batch_y_text = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y:batch_y_text, keep_prop: 1.})

                print step, acc
                if acc > 0.999:
                    saver.save(sess, "capcha.model", global_step=step)
                    break

            step += 1

def crack():
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTHCA, MAX_Len]), 2)
        cor = 0
        step = 0
        while True:
            te, image = get_text_and_image()
            image = convert2gray(image)
            image = image.flatten() / 255
            text_list = sess.run(predict, feed_dict={X:[image], keep_prop: 1})
            text = text_list[0].tolist()
            vector = np.zeros(MAX_CAPTHCA * MAX_Len)
            i = 0
            for n in text:
                vector[i * MAX_Len + n] = 1
                i += 1
            correc =  vec2text(vector)
            if te == correc:
                cor += 1
                print "correct: %d %d" % (cor, step)
            print "%s:%s:%s" % (te, correc, (te == correc))
            step += 1

def zf_yzm_check():
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTHCA, MAX_Len]), 2)
        file = os.listdir('../zf_yzm')
        _text = ''
        for f in file:
            image = Image.open("../zf_yzm/%s" % f)
            image = np.array(image)
            image = convert2gray(image)
            image = image.flatten() / 255
            text_list = sess.run(predict, feed_dict={X:[image], keep_prop: 1})
            text = text_list[0].tolist()
            vector = np.zeros(MAX_CAPTHCA * MAX_Len)
            i = 0
            for n in text:
                vector[i * MAX_Len + n] = 1
                i += 1
            _text = vec2text(vector)
            os.rename("../zf_yzm/%s" % f, "../zf_yzm/%s.jpg" % _text)
            print _text



if __name__ == '__main__':
    # text, image =  get_text_and_image()
    # print text
    # vec =  text2vec(text)
    # print vec
    # train_crack_captcha_cnn()
    # print vec2text(vec)
    # crack()
    zf_yzm_check()



