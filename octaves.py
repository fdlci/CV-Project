import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve 

def gaussian(x,y,sigma):
    return (1/(2*np.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))

def gaussian_filter(sigma):
    size = 2*np.ceil(3*sigma)+1
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = gaussian(x, y, sigma)
    return g/g.sum()

def generate_octaves(im, num_octave, s, sigma):
    octaves = []
    octave = [im]
    k = 2**(1/s)
    kernel = gaussian_filter(k*sigma)
    for i in range(num_octave):
        octave = [im]
        for i in range(s+2):
            next_level = convolve(octave[-1], kernel)
            octave.append(next_level)
        im = octave[-3][::2,::2]
        octaves.append(octave)
    return octaves

def generate_DOG_octaves(gaussian_octaves):
    DOG_octaves = []
    for octave in gaussian_octaves:
        DOG_oct = []
        for i in range(1, len(octave)):
            DOG_oct.append(octave[i] - octave[i-1])
        DOG_octaves.append(np.concatenate([o[:,:,np.newaxis] for o in DOG_oct], axis=2))
    return DOG_octaves