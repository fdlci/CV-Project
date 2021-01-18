import numpy as np
from cv2 import resize, GaussianBlur, subtract, INTER_LINEAR, INTER_NEAREST, imread


def number_of_octaves(image):
    """Defines the number of octaves to compute"""
    return int(np.log(min(image.shape))/np.log(2) -1)

def initial_image(image, sigma, assumed_blur):
    """Sets the blur of the first image to a value of sigma"""

    # Multiply the size of the image by two
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)

    # set a value of initial blur
    sig = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))

    # set initial blur to sigma
    return GaussianBlur(image, (0, 0), sigmaX=sig, sigmaY=sig)

def blur_scale(sigma, num_sets):
    """Computes the blur scale with the initial blur being sigma"""

    # Initializing the blur_scale by sigma
    num_img = num_sets + 3 #we loose one image with the DoG and the 1st and last don't count
    k = 2**(1/num_sets)
    blur_scale = np.zeros(num_img)
    blur_scale[0] = sigma

    # multiplying the blur by 2**(1/s) to achieve twice the blur
    for i in range(1, num_img):
        sigma_previous = k**(i-1) * sigma
        sigma_after = k * sigma_previous
        blur_scale[i] = np.sqrt(sigma_after**2 - sigma_previous**2)

    return blur_scale

def generate_octaves(image, num_octaves, blur_scale):
    """Defines the gaussian octaves that will be used to build the DoG octaves"""
    octaves = []

    # Computing num_octaves, each with num_sets images
    for i in range(num_octaves):
        octave =  []
        octave.append(image)

        for blur in blur_scale[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=blur, sigmaY=blur)
            octave.append(image)
        octaves.append(octave)

        # Resizing the image (division by 2) to use it as base for next octave
        new_base = octave[-3]
        image = resize(new_base, (int(new_base.shape[1] / 2), int(new_base.shape[0] / 2)), interpolation=INTER_NEAREST)

    return np.array(octaves, dtype=object)

def generate_DOG_octaves(gaussian_octaves):
    """Computes the DoG octaves by taking the difference of two consecutive
    images from the gaussian octaves"""

    dog_octaves = []

    for octave in gaussian_octaves:
        dog_oct = []
        for i in range(1,len(octave)):
            dog_oct.append(subtract(octave[i], octave[i-1]))
        dog_octaves.append(dog_oct)

    return np.array(dog_octaves, dtype=object)

def building_DOG_octaves(image, sigma, assumed_blur, num_sets):

    # Initializing the blur of the first image
    init_img = initial_image(image, sigma, assumed_blur)

    # Computing the number of octaves
    num_oct = number_of_octaves(init_img)

    # Computing the blur scale
    blur_scl = blur_scale(sigma, num_sets)

    # Generating Gaussian octaves
    gaussian_octaves = generate_octaves(init_img, num_oct, blur_scl)

    # Generating the DoG octaves
    DoG_octaves = generate_DOG_octaves(gaussian_octaves)

    return DoG_octaves, gaussian_octaves



if __name__ == "__main__":
    image = imread('all_souls_000002.jpg', 0)
    sigma, assumed_blur, num_sets = 1.6, 0.5, 3
    DoG_octaves, gaussian_octaves = building_DOG_octaves(image, sigma, assumed_blur, num_sets)
    print(DoG_octaves[0][0])