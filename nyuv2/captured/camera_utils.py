import scipy.io as sio
import numpy as np
import cv2

def undistort_img(img, fc, pc, rdc, tdc):
    if len(rdc) == 2:
        rdc = np.append(rdc, 0.)
    distortionCoefficients = np.concatenate((rdc[:2], tdc, rdc[2:]))
    cameraMatrix = np.array([[fc[0], 0.,     pc[0]],
                             [0.,     fc[1], pc[1]],
                             [0.,     0.,    1.]])
    img_undist = cv2.undistort(img, cameraMatrix, distortionCoefficients)
    return img_undist


def extract_camera_intrinsics(CameraParameters):
    """
    Get intrinsics from param struct
    """
    focalLength = CameraParameters["FocalLength"][0]
    principalPoint = CameraParameters["PrincipalPoint"][0]
    radialDistortion = CameraParameters["RadialDistortion"][0]
    tangentialDistortion = CameraParameters["TangentialDistortion"][0]
    return focalLength, principalPoint, radialDistortion, tangentialDistortion


def extract_camera_params(filepath):
    arr = sio.loadmat(filepath)
    param_struct = arr["param_struct"]
    RotationOfCamera2 = param_struct["RotationOfCamera2"][0][0]
    TranslationOfCamera2 = param_struct["TranslationOfCamera2"][0][0][0]
    cp1 = param_struct["CameraParameters1"][0][0][0][0]
    cp2 = param_struct["CameraParameters2"][0][0][0][0]

    fc1, pc1, rdc1, tdc1 = extract_camera_intrinsics(cp1)
    fc2, pc2, rdc2, tdc2 = extract_camera_intrinsics(cp2)
    return fc1, fc2, pc1, pc2, rdc1, rdc2, tdc1, tdc2, RotationOfCamera2, TranslationOfCamera2


def project_depth(z, z_mask, imagesize2, fc1, fc2, pc1, pc2, RotationOfCamera2, TranslationOfCamera2):
    """
    Project a depth map z from camera 1 to camera 2, given the rotation matrix and translation vectors.

    Does not account for any distortion.

    Pay attention that the units of z and the units of the translation vector match.

    :param z: Input depth map
    :param imagesize2: Size of output image
    :param fc1: Focal Length in pixels ([x, y] order) of camera 1.
    :param fc2: Focal Length in pixels ([x, y] order) of camera 2.
    :param pc1: pair (y, x) of the center pixel in input depth map
    :param pc2: pair (y, x) of the center pixel in the output depth map
    :param RotationOfCamera2: Rotation of camera 2 relative to camera 1.
    :param TranslationOfCamera2: Translation of camera 2 relative to camera 1.
        # NOTE: To transform, use x^T*R + t^T = x_new^T
    """
    new_image = np.zeros(imagesize2)
    new_mask = np.zeros(imagesize2)
    yy, xx = np.meshgrid(range(z.shape[0]), range(z.shape[1]), indexing="ij")

    # Get coordinates in world units
    x = ((xx - pc1[0]) * z) / fc1[0]
    y = ((yy - pc1[1]) * z) / fc1[1]

    # Use rotation and translation to get (x', y', z')
    xyz = np.array([x, y, z]).transpose(1, 2, 0)
    xyz_new = np.matmul(xyz, RotationOfCamera2) + TranslationOfCamera2
    x_new = xyz_new[..., 0]
    y_new = xyz_new[..., 1]
    z_new = xyz_new[..., 2]

    # Get pixel coords from world coords
    xx_new = np.floor(x_new * fc2[0] / z_new + pc2[0]).astype('int')
    yy_new = np.floor(y_new * fc2[1] / z_new + pc2[1]).astype('int')
    mask = (xx_new >= 0) & (xx_new < imagesize2[1]) & (yy_new >= 0) & (yy_new < imagesize2[0])
    new_image[yy_new[mask], xx_new[mask]] = z_new[mask]
    new_mask[yy_new[mask], xx_new[mask]] = z_mask[mask]
    return new_image, new_mask

