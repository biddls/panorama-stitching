import random
import scipy
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import time


class Stitcher:
    def __init__(self):
        self.colour_pallet = [(255, 0, 0), (255, 170, 0), (0, 255, 0), (0, 255, 0), (127, 0, 255), (0, 255, 255)]

    def stitch(self, _img_left, _img_right, show=False):  # Add input arguments as you deem fit
        """
            The main method for stitching two images
        """

        # _img_left[:, _img_left.shape[1] // 2:] = [0, 0, 0]
        # _img_right[:, _img_right.shape[1] // 2:] = [0, 0, 0]

        # Step 1 - extract the keypoints and features with a suitable feature
        # detector and descriptor
        keypoints_l, descriptors_l = self.compute_descriptors(_img_left)
        keypoints_r, descriptors_r = self.compute_descriptors(_img_right)

        # Step 2 - Feature matching. You will have to apply a selection technique
        # to choose the best matches
        matches = self.matching(descriptors_l, descriptors_r)  # Add input arguments as you deem fit

        print("Number of matching correspondences selected:", len(matches))

        # Step 3 - Draw the matches connected by lines
        if show:
            self.draw_matches(_img_left, _img_right, matches, keypoints_l, keypoints_r)

        # Step 4 - fit the homography model with the RANSAC algorithm
        homography = self.find_homography(matches, keypoints_l, keypoints_r)

        # Step 5 - Warp images to create the panoramic image
        _result = self.warping(_img_left, _img_right, homography)  # Add input arguments as you deem fit

        # Step 6 - Remove black boarders
        _result = self.remove_black_border(_result)

        return _result

    def compute_descriptors(self, img):
        """
        The feature detector and descriptor
        # """
        # Create a SIFT object
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        keypoints, features = sift.detectAndCompute(img, None)

        return keypoints, features

    def matching(self, descriptors_l, descriptors_r):
        """
        Find the matching correspondences between the two images
        """
        # get dist
        matches = scipy.spatial.distance.cdist(descriptors_l, descriptors_r)

        # get best 2 distances
        indices = np.argsort(matches, axis=0)[:, :2]
        temp = []
        for i, indice in enumerate(indices):
            # get the values from the indicies of the best 2
            matchRow = matches[i, indice]
            # apply lowes filter
            if matchRow[0] < 0.75 * matchRow[1]:
                # append the indicie
                temp.append([i, indice[0]])

        return temp

    def draw_matches(self, img_left, img_right, matches, keypoints_l, keypoints_r):
        """
        Connect correspondences between images with lines and draw these lines
        """
        # Display the image with correspondences
        img = np.concatenate((img_left, img_right), axis=1)

        for temp in matches:
            a = keypoints_l[temp[0]]
            b = keypoints_r[temp[1]]

            a = [int(p) for p in a.pt]
            b = [int(p) for p in b.pt]

            b[0] = b[0] + img_left.shape[1]

            colour = random.choice(self.colour_pallet)

            cv2.line(img, a, b, colour, 1)
            cv2.circle(img, a, 4, colour, 2)
            cv2.circle(img, b, 4, colour, 2)

        # img = np.flip(img, axis=2)
        # plt.imshow(img)
        # plt.show()

    def find_homography(self, matches, keypoints_l, keypoints_r):
        """
        Fit the best homography model with the RANSAC algorithm.
        """

        # # Extract source and destination points from the matches
        dst_pts = [keypoints_l[m[0]].pt for m in matches]
        src_pts = [keypoints_r[m[1]].pt for m in matches]

        dst_temp = np.ones((len(dst_pts), 3))
        src_temp = np.ones((len(src_pts), 3))

        dst_temp[:, :2] = np.array(dst_pts)
        src_temp[:, :2] = np.array(src_pts)

        dst_temp = dst_temp.astype(float)
        src_temp = src_temp.astype(float)

        # Run RANSAC to estimate the homography
        homography = solve_homography(dst_temp, src_temp)

        homography = homography.reshape((3, 4))

        return homography

    def warping(self, img_left, img_right, homography, *args):  # Add input arguments as you deem fit
        """
        Warp images to create a panoramic image
        """

        # Get the dimensions of the right image
        h, w = [l + r for l, r in zip(img_left.shape[:2], img_right.shape[:2])]

        # Warp the left image to align with the right image
        warped_img = cv2.warpPerspective(img_right, homography, (w, h))

        warped_img[0:0 + img_left.shape[0], 0:img_left.shape[1]] = img_left
        # Combine the warped left image with the right image
        # result = cv2.addWeighted(img_right, 0.5, warped_img, 0.5, 0)

        return warped_img

    def remove_black_border(self, img):
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
        cropped_image = img[x_min:x_max, y_min:y_max]

        return cropped_image


class Blender:
    def linear_blending(self, *args):
        """
        linear blending (also known as feathering)
        """

        return linear_blending_img

    def customised_blending(self, *args):
        """
        Customised blending of your choice
        """
        return customised_blending_img


def solve_homography(S, D):
    from DLT import dlt
    """
    Find the homography matrix between a set of S points and a set of
    D points
    """

    H = dlt(S, D)

    return H


if __name__ == "__main__":
    # Read the image files
    img_left = cv2.imread("s1.jpg")  # your code here
    img_right = cv2.imread("s2.jpg")  # your code here

    stitcher = Stitcher()
    result = stitcher.stitch(img_left, img_right, show=True)  # Add input arguments as you deem fit

    img = np.flip(result, axis=2)
    plt.imshow(img)
    plt.show()
