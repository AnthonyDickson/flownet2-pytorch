from typing import Optional

import matplotlib.pyplot as plt
import cv2

import numpy as np
import plac


def align_images(query_img, train_img, draw_matches=False):
    """
    Align two images using SURF features and a homography warp.

    :param query_img: The 'template' image.
    :param train_img: The image to align with the template image.
    :param draw_matches: Visualise the two images and the matching SURF descriptors.
    :return: A copy of `train_img` that is aligned with `query_img`.
    """
    surf = cv2.xfeatures2d.SURF_create(400)
    query_key_points, query_descriptors = surf.detectAndCompute(query_img, None)
    train_key_points, train_descriptors = surf.detectAndCompute(train_img, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(query_descriptors, train_descriptors, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    good_matches = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append(m)

    if draw_matches:
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)

        match_viz = cv2.drawMatchesKnn(query_img, query_key_points, train_img, train_key_points, matches, None,
                                       **draw_params)
        plt.imshow(match_viz)
        plt.show()

    src_pts = np.float32([query_key_points[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([train_key_points[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    return cv2.warpPerspective(train_img, H, tuple(reversed(train_img.shape[:2])))


def main(query_img_path: Optional[str] = None, train_img_path: Optional[str] = None):
    """
    Align two images (ideally video frames) via a homography warp.

    :param query_img_path: The path to the first image.
    :param train_img_path: The path to the second image.
    """
    query_img = cv2.imread(query_img_path, cv2.IMREAD_UNCHANGED)
    train_img = cv2.imread(train_img_path, cv2.IMREAD_UNCHANGED)

    aligned_img = align_images(query_img, train_img, draw_matches=True)

    plt.imshow(query_img)
    plt.show()
    plt.imshow(train_img)
    plt.show()
    plt.imshow(aligned_img)
    plt.show()


if __name__ == '__main__':
    plac.call(main)
