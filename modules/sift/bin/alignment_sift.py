import cv2
import numpy as np
import os
import typing
from slide_image import get_slide_image


def _get_sift_homography(
    img_fxd: np.ndarray,
    img_mvg: np.ndarray,
    min_match_count: int = 10,
) -> np.ndarray:
    """
    Get 3x3 homography matrix between two images using the SIFT algorithm.

    Developer notes: Some parameters are hard-coded and work well-enough (e.g.,
    'trees' or 'checks' below), but may need to evaluated.

    Parameters
    ----------
    img_fxd : np.ndarray
        Reference image to align to
    img_mvg : np.ndarray
        Image to transform
    min_match_count : int, optional
        Number of good matches to find between the fixed and moving images to
        perform image reigstration, by default 10

    Returns
    -------
    np.ndarray
        3x3 homography matrix

    Raises
    ------
    AttributeError
        If no. matches detected by SIFT is insufficient to perform image
        registration (10 by default)
    """
    # Find matches between images with SIFT
    sift = cv2.SIFT_create()
    kp_fxd, dsc_fxd = sift.detectAndCompute(img_fxd, None)
    kp_mvg, dsc_mvg = sift.detectAndCompute(img_mvg, None)
    flann = cv2.FlannBasedMatcher(
        index_params={'algorithm': 1, 'trees': 5},
        search_params={'checks': 50},
    )
    matches = flann.knnMatch(dsc_fxd, dsc_mvg, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good_matches = list(filter(
        lambda m: m[0].distance < 0.7 * m[1].distance, 
        matches
    ))

    # Compute 3x3 homography matrix
    if len(good_matches) > min_match_count:
        fxd_pts = np.array([kp_fxd[m.queryIdx].pt for m, _ in good_matches])
        mvg_pts = np.array([kp_mvg[m.trainIdx].pt for m, _ in good_matches])
        M, mask = cv2.findHomography(fxd_pts, mvg_pts, cv2.RANSAC, 5.0)
        return M
    else:
        msg = (
            f'Insufficient no. matches detected to perform image '
            f'registration: {len(good_matches)} matches'
        )
        raise AssertionError(msg)


def align_sift(
    path_fxd: typing.Union[str, os.PathLike],
    path_mvg: typing.Union[str, os.PathLike],
    channel_fxd: int,
    channel_mvg: int,
    path_out: typing.Union[str, os.PathLike],
):
    """
    Align an "moving" image to a "fixed" reference image using:
    1. SIFT to identify keypoints
    2. FLANN to match keypoints between images
    3. RANSAC to optimize the homography matrix to align matched keypoints

    Parameters
    ----------
    path_fxd : typing.Union[str, os.PathLike]
        Path to fixed image, i.e., reference image to align to. Fixed image
        can be of types supported by the SlideImage module
    path_mvg : typing.Union[str, os.PathLike]
        Path to moving image, i.e., image to transform
    channel_fxd : int
        0-indexed channel number in the fixed image to use for alignment
    channel_mvg : int
        0-indexed channel number in the moving image to use for alignment
    path_out : typing.Union[str, os.PathLike]
        Path to save the aligned moving image to (only .tiff output images are
        currently supported)
    """
    img_fxd = get_slide_image(path_fxd)
    img_mvg = get_slide_image(path_mvg)

    # Get homography matrix of moving to fixed images
    M = _get_sift_homography(
        img_fxd=img_fxd._get_slice(C=channel_fxd),
        img_mvg=img_mvg._get_slice(C=channel_mvg),
    )

    # Transform moving image to match fixed image
    dim = img_fxd[channel_fxd].shape[::-1]
    img_wrp = np.stack(
        [cv2.warpPerspective(img, M, dim) for img in img_mvg]  # each channel
    )
    # TODO: save output as .tiff file
    np.save(path_out, img_wrp)