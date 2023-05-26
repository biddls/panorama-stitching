import random
import time
import timeit

import scipy
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

from DLT import dlt


class Stitcher:
    def __init__(self):
        self.colour_pallet = [(255, 0, 0), (255, 170, 0), (0, 255, 0), (0, 255, 0), (127, 0, 255), (0, 255, 255)]

    def stitch(self, _img_left, _img_right, show=False, lowesThresh=0.75):  # Add input arguments as you deem fit
        """
            The main method for stitching two images
        """

        # Step 1 - extract the keypoints and features with a suitable feature
        # detector and descriptor
        keypoints_l, descriptors_l = self.compute_descriptors(_img_left)
        keypoints_r, descriptors_r = self.compute_descriptors(_img_right)

        # Step 2 - Feature matching. You will have to apply a selection technique
        # to choose the best matches
        matches = self.matching(descriptors_l, descriptors_r, lowesThresh)
        if show:
            print("Number of matching correspondences selected:", len(matches))

        # Step 3 - Draw the matches connected by lines
        if show:
            self.draw_matches(_img_left, _img_right, matches, keypoints_l, keypoints_r)

        # Step 4 - fit the homography model with the RANSAC algorithm
        homography = self.find_homography(matches, keypoints_l, keypoints_r)

        # Step 5 - Warp images to create the panoramic image
        _result = self.warping(_img_left, _img_right, homography)

        # Step 6 - Remove black boarders
        _result = self.remove_black_border(_result, _img_left.shape)

        return _result

    def compute_descriptors(self, img):
        """
        The feature detector and descriptor
        GPT filled in
        """
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Create a SIFT object
        sift = cv2.SIFT_create()
        # Detect keypoints and compute descriptors
        keypoints, features = sift.detectAndCompute(img, None)

        return keypoints, features

    def matching(self, descriptors_l, descriptors_r, lowesThresh):
        """
        Find the matching correspondences between the two images
        I have 0 idea why this is doing horrible things to the rest of my code
        """
        # get dist
        matches = scipy.spatial.distance.cdist(descriptors_l, descriptors_r)
        # get best 2 distances
        indices = np.argsort(matches, axis=1)

        temp = []
        for i, indice in enumerate(indices):
            # get the values from the indicies of the best 2
            matchRow = matches[i, indice]
            # apply lowes filter
            if matchRow[0] < lowesThresh * matchRow[1]:
                # append the indicie
                temp.append([i, indice[0]])

        return temp

    def draw_matches(self, img_left, img_right, matches, keypoints_l, keypoints_r):
        """
        Connect correspondences between images with lines and draw these lines
        """
        # Display the image with correspondences
        img = np.concatenate((img_left, img_right), axis=1)

        for match in matches:
            a = keypoints_l[match[0]]
            b = keypoints_r[match[1]]

            a = [int(p) for p in keypoints_l[match[0]].pt]
            b = [int(p) for p in keypoints_r[match[1]].pt]

            b[0] = b[0] + img_left.shape[1]

            colour = random.choice(self.colour_pallet)

            cv2.line(img, a, b, colour, 1)
            cv2.circle(img, a, 4, colour, 2)
            cv2.circle(img, b, 4, colour, 2)

        img = np.flip(img, axis=2)
        plt.imshow(img)
        plt.show()

    def find_homography(self, matches, keypoints_l, keypoints_r):
        """
        Fit the best homography model with the RANSAC algorithm.
        """

        dst_temp = np.ones((len(matches), 3))
        src_temp = np.ones((len(matches), 3))

        # # Extract source and destination points from the matches
        dst_temp[:, :2] = np.array([keypoints_l[m[0]].pt for m in matches]).astype(float)
        src_temp[:, :2] = np.array([keypoints_r[m[1]].pt for m in matches]).astype(float)

        bestErr = float('inf')
        for _ in range(600):
            choices = [random.randint(0, len(matches) - 1) for _ in range(20)]

            homography = solve_homography(dst_temp[choices], src_temp[choices]).reshape((3, 3))
            homography = homography / homography[2, 2]

            reproj = homography @ src_temp.T
            reproj = reproj.T
            reproj = np.divide(reproj.T, reproj[:, 2]).T
            error = np.sum(np.square(dst_temp - reproj))
            if float(error) < bestErr:
                best = homography
                bestErr = error

        return best

    def warping(self, img_left, img_right, homography):
        """
        Warp images to create a panoramic image
        Gpt filled in (heavily edited by me)
        """

        # Get the dimensions of the right image
        h, w = [l + r for l, r in zip(img_left.shape[:2], img_right.shape[:2])]

        boundingBoxR = np.array([[0, 0, 1], [img_right.shape[1], 0, 1], [0, img_right.shape[0], 1], [*img_right.shape[:2][::-1], 1]])
        boundingBoxR = homography @ boundingBoxR.T
        boundingBoxR = (boundingBoxR / boundingBoxR.T[:, 2]).T
        boundingBoxR = boundingBoxR[:, :2]

        # Warp the left image to align with the right image
        warped_img = cv2.warpPerspective(img_right, homography, (w, h))
        warped_img = linear_blending(img_left, warped_img, boundingBoxR)

        return warped_img

    def remove_black_border(self, img, l_shape):
        """
        Remove black border after stitching
        """

        # Find the non-zero pixels in the warped image
        mask = img[:, :, 0] > 0

        # Find the bounding box of the non-zero pixels
        coords = np.argwhere(mask)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        # Crop the warped image based on the bounding box
        cropped_image = img[x_min:l_shape[0], y_min:y_max]

        return cropped_image


def linear_blending(l_img, r_img, boundingBoxR):
    """
    linear blending (also known as feathering)
    """
    mask = r_img[:, :, 0] > 0
    coords = np.argwhere(mask)
    x_min, y_min = coords.min(axis=0)

    l_shape = l_img.shape
    r_img = r_img[:l_shape[0]]
    r_img[:, :int(boundingBoxR[2, 0])] = [0, 0, 0]
    r_shape = r_img.shape

    fade = np.linspace(0, 1, num=abs(l_shape[1] - int(boundingBoxR[2, 0]))).reshape((1, -1)).astype(np.float64)
    fade = np.repeat(fade, repeats=l_shape[0], axis=0)
    fade = np.stack((fade, fade, fade), axis=2)
    l_img[:, int(boundingBoxR[2, 0]):] = l_img[:, int(boundingBoxR[2, 0]):] * fade

    fade = np.linspace(1, 0, num=abs(l_shape[1] - y_min)).reshape((1, -1)).astype(np.float64)
    fade = np.repeat(fade, repeats=r_shape[0], axis=0)
    fade = np.stack((fade, fade, fade), axis=2)
    r_img[:, y_min:l_shape[1]] = r_img[:, y_min:l_shape[1]] * fade

    r_img[:l_shape[0], :l_shape[1]] += l_img

    r_img = r_img[:int(boundingBoxR[2, 1]), :]
    r_img = r_img[int(boundingBoxR[0, 1]):, :]
    r_img = r_img[:, :min(int(boundingBoxR[1, 0]), int(boundingBoxR[3, 0]))]

    return r_img


def solve_homography(_x: np.array, _X: np.array) -> np.array:
    """
    Find the homography matrix between a set of S points and a set of
    D points
    """
    M = np.zeros((2 * len(_x), 9), dtype=np.int32)

    for i, r in enumerate(_x):
        M[i * 2] = [*-_X[i], 0, 0, 0, *(r[0] * _X[i])]
        M[(i * 2) + 1] = [0, 0, 0, *-_X[i], *(r[1] * _X[i])]

    _, _, _M = np.linalg.svd(M, full_matrices=True)

    __M = _M[-1][:]

    return __M


if __name__ == "__main__":
    # Read the image files
    img_left = cv2.imread("l2.jpg")
    img_right = cv2.imread("r2.jpg")

    stitcher = Stitcher()
    result = stitcher.stitch(img_left.copy(), img_right.copy(), show=True, lowesThresh=0.5)  # Add input arguments as you deem fit

    img = np.flip(result, axis=2)
    plt.imshow(img)
    plt.show()

    t = timeit.Timer(lambda: stitcher.stitch(img_left.copy(), img_right.copy(), lowesThresh=0.5))
    t = t.timeit(5)
    print(f"{round(t, 2)} seconds to run")
    print(f"That's {round(t/5, 2)} seconds per run")

