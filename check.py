import numpy
import numpy as np
from PIL import Image


image = 'image.png'
img_array = (255 - np.asarray(Image.open(image).convert('L'))) / 255 - 0.5
img_array = np.expand_dims(numpy.array([img_array]), axis=3)
print(img_array[0].shape)