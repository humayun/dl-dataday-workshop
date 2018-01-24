import numpy as np

from skimage import io
from skimage import data
from skimage.measure import block_reduce

image = io.imread('dog.jpg', as_grey=True)

new_image = block_reduce(image, block_size=(3, 3), func=np.max)

io.imsave('maxpolling_out3.png', new_image)
