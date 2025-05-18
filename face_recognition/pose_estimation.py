import numpy as np
import cv2 as cv


# Define 3D model points of facial landmarks
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

LANDMARK_IDXS = [1, 152, 33, 263, 61, 291]


def get_head_pose(landmarks, image_shape):
    # Extract the 2D image points
    image_points = []
    for idx in LANDMARK_IDXS:
        lm = landmarks[idx]
        x = int(lm.x * image_shape[1])
        y = int(lm.y * image_shape[0])
        image_points.append((x, y))

    image_points = np.array(image_points, dtype=np.float64)

    # Approximate camera internals
    focal_length = image_shape[1]
    center = (image_shape[1] / 2, image_shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs)

    # get degrees
    rmat, _ = cv.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rmat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(proj_matrix)

    yaw = euler_angles[2][0]
    pitch = euler_angles[1][0]
    return yaw, pitch
