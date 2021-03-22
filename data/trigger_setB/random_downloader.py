from PIL import Image
from StringIO import StringIO
import requests

for i in range(100):
    print i
    res = requests.get('https://source.unsplash.com/random')
    img = Image.open(StringIO(res.content))
    img.save('./pics/' + str(i+1) + '.jpg')
