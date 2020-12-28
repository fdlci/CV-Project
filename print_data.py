from skimage.io import imread
import matplotlib.pyplot as plt

img1 = imread('all_souls_000002.jpg')
img2 = imread('all_souls_000015.jpg')
plt.figure(figsize=(6,4))

plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.axis('off')
plt.title('All Souls Oxford 1')

plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.axis('off')
plt.title('All Souls Oxford 2')

plt.show()