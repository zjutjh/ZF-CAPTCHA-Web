import base64

import os
import requests
import time
from PIL import Image
from cStringIO import StringIO

if __name__ == '__main__':
    file = os.listdir('../zf_yzm/zf_yzm')
    for f in file:
        img = Image.open('../zf_yzm/zf_yzm/%s' % f)
        buffer = StringIO()
        img.save(buffer, format='JPEG')
        binary_data = buffer.getvalue()
        base64_data = base64.b64encode(binary_data)
        # print base64_data
        begin = time.time()
        res = requests.post('http://127.0.0.1:5000/test', data={
            'img' : base64_data
        })
        end = time.time()
        # print end - begin
        print res.text
