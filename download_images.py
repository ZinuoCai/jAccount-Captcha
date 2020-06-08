import os
import requests

uuid = None
t = None
image_count = 1000

headers = {
    'Accept': 'text/html, */*; q=0.01',
    'Referer': '',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/83.0.4103.61 Safari/537.36',
}
base_url = 'https://jaccount.sjtu.edu.cn/jaccount/captcha?uuid=%s&t=%s' % (uuid, t)
base_dir = 'data/MyData/source_images/'

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for i in range(image_count):
    image_url = '%s%04d' % (base_url, i)
    image_dir = '%s%04d.png' % (base_dir, i)
    resp = requests.get(image_url, headers=headers)
    with open(image_dir, 'wb') as f:
        f.write(resp.content)
