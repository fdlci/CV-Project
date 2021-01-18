from builduing_DoG_octaves import building_DOG_octaves
import matplotlib.pyplot as plt
from cv2 import imread

def plotting_gaussian_images(gaussian_octaves, layer):
    fig=plt.figure(figsize=(8, 8))
    columns = 3
    rows = 2 
    for i, image in enumerate(gaussian_octaves[layer]):
        fig.add_subplot(rows, columns, i+1)
        plt.title('Image ' + str(i))
        plt.imshow(image)
    plt.suptitle('Layer number ' + str(layer) + ' of the gaussian octaves')
    plt.show()

def plotting_DOG_images(DoG_octaves, layer):
    fig=plt.figure(figsize=(8, 8))
    columns = 3
    rows = 2 
    for i, image in enumerate(DoG_octaves[layer]):
        fig.add_subplot(rows, columns, i+1)
        plt.title('Image ' + str(i))
        plt.imshow(image)
    plt.suptitle('Layer number ' + str(layer) + ' of the DoG octaves')
    plt.show()

if __name__ == "__main__":
    image = imread('all_souls_000002.jpg', 0)
    sigma, assumed_blur, num_sets = 1.6, 0.5, 3
    layer = 3
    DoG_octaves, gaussian_octaves = building_DOG_octaves(image, sigma, assumed_blur, num_sets)
    plotting_gaussian_images(gaussian_octaves, layer)
    plotting_DOG_images(DoG_octaves, layer)
