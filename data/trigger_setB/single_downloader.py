from PIL import Image
from StringIO import StringIO
import requests

i=77
print i
res = requests.get('https://source.unsplash.com/random')
img = Image.open(StringIO(res.content))
img.save('./pics/' + str(i+1) + '.jpg')
