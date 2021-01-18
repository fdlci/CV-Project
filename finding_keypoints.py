import numpy as np
from numpy.linalg import det, lstsq
from cv2 import KeyPoint

def grad_and_hessian(im):
    """Computes the gradient and the Hessian of a given cube of pixels of size 3x3x3"""

    #computing the gradient
    dx = (im[1, 1, 2] - im[1, 1, 0])/2
    dy = (im[1, 2, 1] - im[1, 0, 1])/2
    ds = (im[2, 1, 1] - im[0, 1, 1])/2

    grad = np.array([dx, dy, ds])

    #computing the Hessian
    cpixel = im[1, 1, 1]
    dxx = im[1, 1, 2] - 2 * cpixel + im[1, 1, 0]
    dyy = im[1, 2, 1] - 2 * cpixel + im[1, 0, 1]
    dss = im[2, 1, 1] - 2 * cpixel + im[0, 1, 1]
    dxy = (im[1, 2, 2] - im[1, 2, 0] - im[1, 0, 2] + im[1, 0, 0])/4
    dxs = (im[2, 1, 2] - im[2, 1, 0] - im[0, 1, 2] + im[0, 1, 0])/4
    dys = (im[2, 2, 1] - im[2, 0, 1] - im[0, 2, 1] + im[0, 0, 1])/4

    hess =  np.array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

    return grad, hess

def is_extremum(im1, im2, im3, thresh):
    """For a given cube of pixels of size 3x3x3, returns True if the center pixel is
    an extremum, False otherwise"""

    # checks if the middle pixel is an extremum
    n, p = im1.shape
    new_sub_im = np.zeros((n,p,3))
    new_sub_im[:,:,0] = im1
    new_sub_im[:,:,1] = im2
    new_sub_im[:,:,2] = im3
    # take the pixel in the middle of the 3x3x3 subimage
    interest_pixel = im2[1,1]

    if abs(interest_pixel) > thresh:
        if interest_pixel == np.amax(new_sub_im):
            return True
        elif interest_pixel == np.amin(new_sub_im):
            return True
    return False

def outside_image(i, j, width, image_shape0, image_shape1, img_ind, num_sets):
    """For a given pixel at place i, j, returns True if pixel is outside of the considered
    frame of the image, False otherwise"""

    if i < width or i >= image_shape0 - width or j < width or j >= image_shape1 - width or img_ind < 1 or img_ind > num_sets:
            return True
    return False

def localize_extremum(i, j, img_ind, oct_ind, num_sets, dog_oct, sigma, thresh, width, r=10, max_iter=5):
    """Checks the two conditions given by the SIFT paper. Gets rid of points such that:
        - the intensity of the extrema is under a certain threshold (low contrast)
        - Edge responses"""


    # starting with the keypoint inside the image
    loc_out = False

    image_shape = dog_oct[0].shape

    for cpt in range(max_iter):

        # Takes three consecutive images of the considered DoG octave
        im1, im2, im3 = dog_oct[img_ind-1:img_ind+2]

        #convert pixel values to values between (0,1) to apply Lowe's thresh
        interest_area = np.stack([im1[i-1:i+2, j-1:j+2], im2[i-1:i+2, j-1:j+2], im3[i-1:i+2, j-1:j+2]]).astype('float32') / 255

        # Compute gradient and Hessian of the pixel cube
        grad, hess = grad_and_hessian(interest_area)

        # Computing the subpixel offset
        subpixel_offset = -lstsq(hess, grad, rcond=None)[0]

        # If one of the absolute values of the coordinates is below 0.5, get rid of kp
        max_abs = np.amax(abs(subpixel_offset))
        if max_abs < 0.5:
            break
        
        # Compute position of new extrema and image_ind
        j += int(round(subpixel_offset[0]))
        i += int(round(subpixel_offset[1]))
        img_ind += int(round(subpixel_offset[2])) 

        # Check if new extrema is in the considered frame of the image
        if outside_image(i, j, width, image_shape[0], image_shape[1], img_ind, num_sets):
            loc_out = True
            break
    
    # Keypoint is outside the frame considered
    if loc_out:
        return None
    
    # Too many attempts without convergence
    if cpt >= max_iter - 1:
        return None

    # Computing value of new extrema
    new_extrema_value = interest_area[1,1,1] + np.dot(grad, subpixel_offset)/2

    # Eliminating Low contrast
    if abs(new_extrema_value)*num_sets >= thresh:

        # Eliminating Edge responses
        trace_hess = np.trace(hess[:2, :2])
        determinant = det(hess[:2, :2])

        if determinant > 0 and r * trace_hess < (r+1)*determinant:
            k = 2**oct_ind
            x = (j + subpixel_offset[0])*k
            y = (i + subpixel_offset[1])*k
            s = subpixel_offset[2]

            # creating a keypoint class
            keypoint = KeyPoint()
            # keypoint coordinates
            keypoint.pt = (x,y)
            # from which octave the keypoint has been extracted
            keypoint.octave = oct_ind + img_ind * (2 ** 8) + int(round((s + 0.5) * 255)) * (2 ** 16)
            # diameter of the meaningful leypoint neighborhood
            keypoint.size = sigma * (2 ** ((img_ind + s) / np.float32(num_sets))) * (2 ** (oct_ind + 1))
            # Value of the intensity of the keypoint
            keypoint.response = abs(new_extrema_value)

            return keypoint, img_ind

    return None