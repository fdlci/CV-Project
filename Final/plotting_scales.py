from builduing_DoG_octaves import building_DOG_octaves
import matplotlib.pyplot as plt
from cv2 import imread

def plotting_gaussian_images(gaussian_octaves):
    fig=plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3
    for layer in range(len(gaussian_octaves)):
        image = gaussian_octaves[layer][0]
        fig.add_subplot(rows, columns, layer+1)
        plt.title('1st image of layer ' + str(layer+1))
        plt.axis('off')
        plt.imshow(image)
    plt.suptitle('First images of all gaussian octaves')
    plt.show()

def plotting_DOG_images(DoG_octaves):
    fig=plt.figure(figsize=(8, 8))
    columns = 3
    rows = 2
    for layer in range(len(gaussian_octaves)-3):
        image = DoG_octaves[layer][0]
        fig.add_subplot(rows, columns, layer+1)
        plt.title('1st image of layer ' + str(layer+1))
        plt.axis('off')
        plt.imshow(image)
    plt.suptitle('First images of the first 6 DoG octaves')
    plt.show()

if __name__ == "__main__":
    image = imread('all_souls_000002.jpg', 0)
    sigma, assumed_blur, num_sets = 1.6, 0.5, 3
    DoG_octaves, gaussian_octaves = building_DOG_octaves(image, sigma, assumed_blur, num_sets)
    plotting_gaussian_images(gaussian_octaves)
    plotting_DOG_images(DoG_octaves)
