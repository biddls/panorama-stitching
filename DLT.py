import os.path
import numpy as np
import cv2 as cv
from time import time_ns

# Implementation of the DLT method to compute the camera projection matrix
logging = True

p = print if logging else lambda _: None


def decompose_P(_P: np.array) -> (np.array, np.array, np.array):
    # Decompose Camera projection matrix P into intrinsic (c, s, m , x_0, y_0)
    # and extrinsic  parameters (R, X0).
    # Input:  Camera projection matrix (P:= 3x4)
    # Output: Intrinsics (K:= 3x3)
    #         Origin of camera in world frame (X0:= 3x1)
    #         Orientation of the camera in the world frame (R:=3x3)
    # Notes:
    # Camera projection equation
    # X (3D point in world) -> x (2d position in image)
    # x = KR[I|-X0]X

    # compute the intrinsic matrix K
    _P = np.reshape(_P, (3, 4))[:, :-1]
    _K, _R = np.linalg.qr(np.linalg.inv(_P))

    # compute the extrinsic matrix R and projection center X0
    _T = np.linalg.inv(_K) @ _P
    _X0 = -np.linalg.inv(_R) @ _T
    return _K / _K[2, 2], _R, _X0


def dlt(_x: np.array, _X: np.array) -> np.array:
    # 1. Build linear system Mp = 0

    M = np.zeros((2 * len(_x), 12), dtype=np.int32)
    # [x, y]
    for i, r in enumerate(_x):
        M[i * 2] = [*_X[i], 1, 0, 0, 0, 0, *(-r[0] * _X[i]), -r[0]]
        M[(i * 2) + 1] = [0, 0, 0, 0, *_X[i], 1, *(-r[1] * _X[i]), -r[1]]

    # 2. Perform SVD => USV
    _, _, _M = np.linalg.svd(M, full_matrices=True)

    __M = _M[-1][:]
    # p(M.shape, _M.shape, __M.shape)
    # temp = _M @ __M
    # p(np.sum(temp))

    return __M


def reproject(_P: np.array, _x: np.array, _X: np.array):
    # 1. Convert 3D points to homogeneous coordinates
    __X = np.ones((len(_X), 4))
    __X[:, :-1] = _X

    # 2. Project 3D points to 2D points
    _P = np.reshape(_P, (3, 4))
    # p(f"{_P.shape=} {__X.T.shape=}")
    __x = _P @ __X.T

    # 3. Convert 2D points to inhomogeneous coordinates
    # p(f"{__x.shape=}")
    __x = __x / __x[-1]

    # 4. Compare with original 2D points
    p(f"\nOriginal 2D points:\n{_x}\n")
    p(f"Reprojected 2D points:\n{np.round(__x.T[:, :-1], 3)}\n")

    # 5. Compute the re-projection error
    # p(f"{_x.shape=}, {__x.T[:,:-1].shape=}")
    error = np.linalg.norm(_x - __x.T[:, :-1], axis=1)
    p(f"Reprojection error:\n{np.average(error):.2f} Pixels")


class Points:
    def __init__(self, _img, _X):
        self.__img = _img
        self.__X = _X
        self.__points = np.zeros((len(_X), 2), dtype=np.int32)
        self.__selected = 0

    def get_points(self):
        if os.path.exists('points.npy'):
            return np.load('points.npy')
        print(" -> is X | depth is Y | ^ is Z")
        print("Select the points in the image in the following order:")
        print(f"{self.__X[0]}", end='')
        # Set window size
        cv.namedWindow('labeling', cv.WINDOW_NORMAL)  # Can be resized
        cv.resizeWindow('labeling', img.shape[1], img.shape[0])  # Reasonable size window

        # displaying the image
        cv.imshow('labeling', self.__img)

        # setting mouse handler for the image
        # and calling the __mouse_callback() function
        cv.setMouseCallback('labeling', self.__mouse_callback)
        cv.windowName = 'labeling'
        # wait for a key to be pressed to exit
        cv.waitKey(0)

        # close the window
        cv.destroyAllWindows()

        np.save('points.npy', self.__points)
        return self.__points

    # mouse callback function
    def __mouse_callback(self, event, _x, _y, *args):
        # checking for left mouse clicks
        if event == cv.EVENT_LBUTTONDOWN:
            self.__points[self.__selected] = np.array([_x, _y])
            self.__selected += 1

            # Output the selected points
            p(f" -> ({_x},{_y})")

            # Check if all points are selected
            if self.__selected == len(self.__X):
                cv.destroyAllWindows()
                return

            # Output the next point to be selected
            print(f"{self.__X[self.__selected]}", end='')

            # displaying the coordinates on the image
            cv.drawMarker(self.__img, (_x, _y), (0, 255, 0), cv.MARKER_CROSS, 10, 2)
            cv.putText(
                self.__img,
                str(self.__X[self.__selected - 1]),
                (_x, _y),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

            # displaying the image with the new points and text
            cv.imshow('labeling', self.__img)


def reprojectOntoCube(_P: np.array, _img: np.array):
    # generate all possible cordinate points of the visible faces of the cube
    shape = (7, 9, 7)
    _x = []
    for __x in range(0, shape[0] + 1):
        for _z in range(0, shape[2] + 1):
            _x.append([__x, 0, _z])

    for _y in range(0, shape[1] + 1):
        for _z in range(0, shape[2] + 1):
            _x.append([0, _y, _z])

    for __x in range(0, shape[0] + 1):
        for _y in range(0, shape[1] + 1):
            _x.append([__x, _y, 7])

    _x = np.array(_x)

    # 1. Convert 3D points to homogeneous coordinates
    __X = np.ones((len(_x), 4))
    __X[:, :-1] = _x

    # 2. Project 3D points to 2D points
    _P = np.reshape(_P, (3, 4))
    __x = _P @ __X.T

    # 3. Convert 2D points to inhomogeneous coordinates
    __x = __x / __x[-1]
    __x = __x.T[:, :-1]

    # 4. Draw the cube
    # Set window size
    cv.namedWindow('image', cv.WINDOW_NORMAL)  # Can be resized
    cv.resizeWindow('image', img.shape[1], img.shape[0])  # Reasonable size window

    # plot the cube
    backtorgb = cv.cvtColor(_img, cv.COLOR_GRAY2RGB)
    cv.imshow('image', backtorgb)

    # plot the points
    __x = __x.astype(int)
    for i in range(len(__x)):
        cv.drawMarker(backtorgb, (__x[i][0], __x[i][1]), (0, 255, 0), cv.MARKER_CROSS, 1, 2)

    # displaying the image with the new points and text
    cv.imshow('image', backtorgb)

    # wait for a key to be pressed to exit
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # load the input image (cube0.jpg)
    img = cv.imread('cube0.jpg', cv.COLOR_BGR2GRAY)

    # Define 3D control points (Minimum of 6 Points)
    X = np.array([
        [0, 0, 0],
        [7, 0, 0],
        [7, 0, 7],
        [0, 0, 7],
        [7, 9, 7],
        [0, 9, 7],
        [0, 9, 0]
    ])

    a = Points(img, X)
    x = a.get_points()

    # measure performance
    _time = time_ns()

    # perform dlt
    P = dlt(x, X)

    # # decompose projection matrix to get instrinsics and extrinsics
    [K, R, X0] = decompose_P(P)

    # end performance measurement
    end = (time_ns() - _time) / 1e9
    print(f"\nTime taken:\n\t{end:.8f} seconds\n")

    p(f"The estimated projection matrix:\n{P}\n")
    p(f"Intrinsic matrix:\n{K}\n")
    p(f"Extrinsic matrix:\n\tR:\n{R}\n\tX0:\n{X0}")

    reproject(P, x, X)

    # Show the cube reprojected back onto the image
    reprojectOntoCube(P, img)
