import numpy as np
import random
import cv2 as cv
import math


# find all matches with given images
def find_matches(query, img, ratio):
    sift = cv.xfeatures2d.SIFT_create()

    # find SIFT keypoints and descriptors
    keypoints_1, descriptors_1 = sift.detectAndCompute(query, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img, None)

    # matching with knn matching
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Filter matches with ratio, store good matches
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return keypoints_1, keypoints_2, good_matches


# view all matches with a given threshold
def view_matches(img_1, img_2):
    threshold = 0.56
    kp_1, kp_2, good_matches = find_matches(img_1, img_2, threshold)

    # Draw matches
    matched_img = np.empty((max(img_1.shape[0], img_2.shape[0]), img_1.shape[1] + img_2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img_1, kp_1, img_2, kp_2, good_matches, matched_img,
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return kp_1, kp_2, good_matches, matched_img


def view_random_matches(kp_1, kp_2, matches, img_1, img_2):
    matched_img = np.empty((max(img_1.shape[0], img_2.shape[0]), img_1.shape[1] + img_2.shape[1], 3), dtype=np.uint8)
    matched_img[0: img_1.shape[0], 0: img_1.shape[1]] = img_1
    matched_img[0: img_2.shape[0], img_1.shape[1]: img_1.shape[1] + img_2.shape[1]] = img_2

    pts_1 = np.array([np.round(kp_1[m.queryIdx].pt) for m in matches])
    pts_2 = np.array([np.round(kp_2[m.trainIdx].pt) for m in matches])

    # randomly choose 8 random int
    # print(len(matches))
    # eight_matches = [i for i in range(90, 130)]
    eight_matches = [random.randint(0, len(matches) - 1) for _ in range(8)]
    print(eight_matches)
    for i in eight_matches:
        end1 = (int(pts_1[i, 0]), int(pts_1[i, 1]))
        end2 = (int(pts_2[i, 0]) + img_1.shape[1], int(pts_2[i, 1]))
        cv.line(matched_img, end1, end2, (0, 255, 255), 2)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(matched_img, str(i), end1, font, .5, (255, 255, 255), 2, cv.LINE_AA)
    return eight_matches, matched_img


def view_reliable_matches(kp_1, kp_2, matches, indexs, img_1, img_2):
    matched_img = np.empty((max(img_1.shape[0], img_2.shape[0]), img_1.shape[1] + img_2.shape[1], 3), dtype=np.uint8)
    matched_img[0: img_1.shape[0], 0: img_1.shape[1]] = img_1
    matched_img[0: img_2.shape[0], img_1.shape[1]: img_1.shape[1] + img_2.shape[1]] = img_2

    pts_1 = np.array([np.round(kp_1[m.queryIdx].pt) for m in matches])
    pts_2 = np.array([np.round(kp_2[m.trainIdx].pt) for m in matches])

    # print(indexs)
    reliable_pts1 = []
    reliable_pts2 = []
    for i in indexs:
        end1 = (int(pts_1[i, 0]), int(pts_1[i, 1]))
        end2 = (int(pts_2[i, 0]) + img_1.shape[1], int(pts_2[i, 1]))
        reliable_pts1.append([int(pts_1[i, 0]), int(pts_1[i, 1])])
        reliable_pts2.append([int(pts_2[i, 0]), int(pts_2[i, 1])])
        cv.line(matched_img, end1, end2, (0, 255, 255), 2)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(matched_img, str(i), end1, font, 1., (255, 255, 255), 2, cv.LINE_AA)
    return reliable_pts1, reliable_pts2, matched_img


# b)
def std_8_points_algo(kp1, kp2):
    # p_r.T * F * p_l = 0
    # construct matrix A
    print("*****std 8 points algo*****")
    A = np.zeros((len(kp1), 9))
    print(A.shape)
    for i in range(len(kp1)):
        x_l, y_l = kp1[i]
        x_r, y_r = kp2[i]
        # A[i, :] = np.array([x_l * x_r, x_l * y_r, x_l, y_l * x_r, y_l * y_r, y_l, x_r, y_r, 1]).astype(np.float)
        A[i, :] = np.array([x_r * x_l, x_r * y_l, x_r, y_r * x_l, y_r * y_l, y_r, x_l, y_l, 1])

    # compute svd of A
    u_a, s_a, v_a = np.linalg.svd(A)

    # find f as the column corresponding to the smallest value in D
    index = np.argmin(s_a)
    # print(index)
    f = v_a[index]

    # reshape f to 3x3
    F = f.reshape((3, 3))

    # compute svd of F
    u_f, s_f, v_f = np.linalg.svd(F)
    # set smallest value in v to 0
    s_f[-1] = 0
    # compute F again
    # print(s_f)
    F_rank_2 = np.matmul(u_f, np.matmul(np.diag(s_f), v_f))
    # make last element to 1.
    # F_rank_2 /= np.linalg.norm(F_rank_2)
    F_rank_2 /= F_rank_2[2, 2]
    return F_rank_2


# c)
def draw_epipolar_line(F, pt_other, pt_in, right_img):
    for i in range(len(pt_other)):
        x_l, y_l = pt_other[i]
        x_r, y_r = pt_in[i]
        left_pt = np.array([x_l, y_l, 1.]).astype(np.float)
        line_r = np.dot(F, left_pt)
        cv.circle(right_img, (x_r, y_r), 10, (255, 0, 255), -1)
        # find 2 points one line out of image
        x1 = 0
        y1 = (- line_r[2]) / (line_r[1])
        x2 = float(right_img.shape[1])
        y2 = (-1 * x2 * line_r[0] - line_r[2]) / (line_r[1])
        # print(line_r)
        # print((x1, y1), (x2, y2))
        cv.line(right_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    return right_img


# d)
def rectify_img(img_1, img_2, pts1, pts2, F):
    h, w = img_1.shape[0:2]
    _, h_1, h_2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, (w, h))

    print('***********rectify*****************')
    print(h_1)
    print(h_2)

    corners = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    corners = np.array([corners])

    dst_corners = cv.perspectiveTransform(corners, h_1)[0]
    dst_corners = np.asarray(dst_corners, dtype=np.float32)

    bbox = cv.boundingRect(dst_corners)
    w_after_l = bbox[2]
    h_after_l = bbox[3]

    # T2 is used to move left-top corner of bbox to (0,0)
    t_l = np.matrix([[1., 0., -1. * bbox[0]], [0., 1., -1. * bbox[1]], [0., 0., 1.]])

    dst_corners = cv.perspectiveTransform(corners, h_2)[0]
    dst_corners = np.asarray(dst_corners, dtype=np.float32)
    bbox_r = cv.boundingRect(dst_corners)

    w_after_r = bbox_r[2]
    h_after_r = bbox_r[3]

    t_r = np.matrix([[1., 0., -1. * bbox_r[0]], [0., 1., -1. * bbox_r[1]], [0., 0., 1.]])

    left_rect = cv.warpPerspective(img_1, t_l * h_1, (w_after_l, h_after_l))
    right_rect = cv.warpPerspective(img_2, t_r * h_2, (w_after_r, h_after_r))

    # return left_rect, cv.resize(right_rect, (left_rect.shape[1], left_rect.shape[0]))
    return left_rect, right_rect
    # return img_1, img_2


def drawlines(right_img, lines, pts2):
    _, c, _ = right_img.shape
    for r, pt2 in zip(lines, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        cv.line(right_img, (x0, y0), (x1, y1), color, 2)
        cv.circle(right_img, tuple(pt2), 8, color, -1)
    return right_img


if __name__ == '__main__':
    l_1 = cv.imread('../prof/I1.jpg')
    l_2 = cv.imread('../prof/I2.jpg')
    l_3 = cv.imread('../prof/I3.jpg')
    # print(l_1.shape)

    # q3 a)
    kp_1_12, kp_2_12, good_matches_12, match_img_12 = view_matches(l_1, l_2)
    kp_1_13, kp_2_13, good_matches_13, match_img_13 = view_matches(l_1, l_3)
    print(len(good_matches_12))
    print(len(good_matches_13))
    cv.imwrite("../results/q3a_all_matches_12.jpg", match_img_12)
    cv.imwrite("../results/q3a_all_matches_13.jpg", match_img_13)


    eight_matches_12, eight_matched_img_12 = view_random_matches(kp_1_12, kp_2_12, good_matches_12, l_1, l_2)
    eight_matches_13, eight_matched_img_13 = view_random_matches(kp_1_13, kp_2_13, good_matches_13, l_1, l_3)
    cv.imwrite("../results/q3a_random_matches_12.jpg", eight_matched_img_12)
    cv.imwrite("../results/q3a_random_matches_13.jpg", eight_matched_img_13)


    # reliable matches for cat
    # reliable_match_12 = [13, 21, 23, 25, 36, 32, 612, 742]
    # reliable_match_13 = [52, 36, 47, 73, 83, 105, 118, 114]
    # for image jimin
    # reliable_match_12 = [21, 612, 742, 633, 842, 688, 644, 607]
    # # first, 0 for homography
    # reliable_match_13 = [47, 83, 234, 87, 288, 221, 200, 158]
    # epipole in img
    # reliable_match_13 = [288, 158, 191, 269, 260, 161, 236, 72, 183]
    # cv epipole in img
    # reliable_match_13 = [221, 234, 87, 83, 288, 201, 47, 200, 86]
    # better
    # reliable_match_13 = [234, 221, 158, 87, 288, 201, 284, 200, 226, 107]

    # for prof's images
    reliable_match_12 = [6514, 1077, 3608, 6444, 4054, 5749, 1911, 2811]
    reliable_match_13 = [133, 239, 398, 532, 466, 315, 507, 380]

    reliable_pts1_12, reliable_pts2_12, reliable_matched_img_12 = \
        view_reliable_matches(kp_1_12, kp_2_12, good_matches_12, reliable_match_12, l_1, l_2)
    reliable_pts1_13, reliable_pts2_13, reliable_matched_img_13 = \
        view_reliable_matches(kp_1_13, kp_2_13, good_matches_13, reliable_match_13, l_1, l_3)

    cv.imwrite("../results/q3a_reliable_matches_12.jpg", reliable_matched_img_12)
    cv.imwrite("../results/q3a_reliable_matches_13.jpg", reliable_matched_img_13)

    # print("reliable matches: ")
    # print("########")
    # print(reliable_pts1_12)
    # print("########")
    # print(reliable_pts2_12)
    # print("########")
    # print(reliable_pts1_13)
    # print("########")
    # print(reliable_pts2_13)


    # q3 b) compute fundamental matrix
    F_12 = std_8_points_algo(reliable_pts1_12, reliable_pts2_12)
    F_13 = std_8_points_algo(reliable_pts1_13, reliable_pts2_13)


    # q3 c) calculate the epipolar lines in the right image
    ################ for (l1, l2) pair ####################
    right_img = l_2.copy()
    r_epipolar_line_img = draw_epipolar_line(F_12, reliable_pts1_12, reliable_pts2_12, right_img)
    cv.imwrite("../results/q3c_epipolar_line_r_12.jpg", r_epipolar_line_img)

    left_img = l_1.copy()
    l_epipolar_line_img = draw_epipolar_line(F_12.T, reliable_pts2_12, reliable_pts1_12, left_img)
    both_epipolar_line = np.empty(
        (max(left_img.shape[0], right_img.shape[0]), left_img.shape[1] + right_img.shape[1], 3), dtype=np.uint8)
    both_epipolar_line[0: left_img.shape[0], 0: left_img.shape[1]] = l_epipolar_line_img
    both_epipolar_line[0: right_img.shape[0],
    left_img.shape[1]: left_img.shape[1] + right_img.shape[1]] = r_epipolar_line_img
    cv.imwrite("../results/q3c_epipolar_line_both_12.jpg", both_epipolar_line)

    ################ for (l1, l3) pair ####################
    right_img = l_3.copy()
    r_epipolar_line_img_13 = draw_epipolar_line(F_13, reliable_pts1_13, reliable_pts2_13, right_img)
    cv.imwrite("../results/q3c_epipolar_line_r_13.jpg", r_epipolar_line_img_13)

    left_img = l_1.copy()
    l_epipolar_line_img_13 = draw_epipolar_line(F_13.T, reliable_pts2_13, reliable_pts1_13, left_img)
    both_epipolar_line_13 = np.empty(
        (max(left_img.shape[0], right_img.shape[0]), left_img.shape[1] + right_img.shape[1], 3), dtype=np.uint8)
    both_epipolar_line_13[0: left_img.shape[0], 0: left_img.shape[1]] = l_epipolar_line_img_13
    both_epipolar_line_13[0: right_img.shape[0],
    left_img.shape[1]: left_img.shape[1] + right_img.shape[1]] = r_epipolar_line_img_13
    cv.imwrite("../results/q3c_epipolar_line_both_13.jpg", both_epipolar_line_13)


    # # q3 d) rectify image with computed fundamental matrix
    # ################ for (l1, l2) pair ####################
    left_rect, right_rect = rectify_img(l_epipolar_line_img.copy(), r_epipolar_line_img.copy(),
                                        np.array(reliable_pts1_12), np.array(reliable_pts2_12), F_12)

    rectified_imgs_12 = np.empty(
        (max(left_rect.shape[0], right_rect.shape[0]), left_rect.shape[1] + right_rect.shape[1], 3), dtype=np.uint8)
    rectified_imgs_12[0: left_rect.shape[0], 0: left_rect.shape[1]] = left_rect
    rectified_imgs_12[0: right_rect.shape[0], left_rect.shape[1]: left_rect.shape[1] + right_rect.shape[1]] = right_rect
    cv.imwrite("../results/q3d_rectified_12.jpg", rectified_imgs_12)

    # ################ for (l1, l3) pair ####################
    left_rect_13, right_rect_13 = rectify_img(l_epipolar_line_img_13.copy(), r_epipolar_line_img_13.copy(),
                                        np.array(reliable_pts1_13), np.array(reliable_pts2_13), F_13)

    rectified_imgs_13 = np.empty(
        (max(left_rect_13.shape[0], right_rect_13.shape[0]), left_rect_13.shape[1] + right_rect_13.shape[1], 3), dtype=np.uint8)
    rectified_imgs_13[0: left_rect_13.shape[0], 0: left_rect_13.shape[1]] = left_rect_13
    rectified_imgs_13[0: right_rect_13.shape[0], left_rect_13.shape[1]: left_rect_13.shape[1] + right_rect_13.shape[1]] = right_rect_13
    cv.imwrite("../results/q3d_rectified_13.jpg", rectified_imgs_13)
    #
    #
    # # q3 e) compute fundamental matrix with opencv functions
    F_12_cv = cv.findFundamentalMat(np.array(reliable_pts1_12), np.array(reliable_pts2_12), cv.FM_LMEDS)[0]
    F_13_cv = cv.findFundamentalMat(np.array(reliable_pts1_13), np.array(reliable_pts2_13), cv.FM_LMEDS)[0]
    print("**********12************")
    print(F_12)
    print(np.linalg.norm(F_12))
    print('=============================')
    print(F_12_cv)
    print(np.linalg.norm(F_12_cv))
    print("**********13************")
    print(F_13)
    print(np.linalg.norm(F_13))
    print('=============================')
    print(F_13_cv)
    print(np.linalg.norm(F_13_cv))
    #
    # ################ for (l1, l2) pair ####################
    lines2 = cv.computeCorrespondEpilines(np.array(reliable_pts1_12).reshape(-1, 1, 2), 1, F_12_cv)
    lines2 = lines2.reshape(-1, 3)
    r_epipolar_line_img_cv_12 = drawlines(r_epipolar_line_img, lines2, np.array(reliable_pts2_12))
    cv.imwrite("../results/q3c_epipolar_line_r_cv_12.jpg", r_epipolar_line_img_cv_12)
    #
    # # ################ for (l1, l3) pair ####################
    lines3 = cv.computeCorrespondEpilines(np.array(reliable_pts1_13).reshape(-1, 1, 2), 1, F_13_cv)
    lines3 = lines3.reshape(-1, 3)
    r_epipolar_line_img_cv_13 = drawlines(r_epipolar_line_img_13, lines3, np.array(reliable_pts2_13))
    cv.imwrite("../results/q3c_epipolar_line_r_cv_13.jpg", r_epipolar_line_img_cv_13)


    # # q3 e) rectify image with computed fundamental matrix by opencv functions
    # ################ for (l1, l2) pair ####################
    right_img = l_2.copy()
    lines2_cv = cv.computeCorrespondEpilines(np.array(reliable_pts1_12).reshape(-1, 1, 2), 1, F_12_cv)
    lines2_cv = lines2_cv.reshape(-1, 3)
    Qf_r_epipolar_line_img_cv_12 = drawlines(right_img, lines2_cv, np.array(reliable_pts2_12))

    left_img = l_1.copy()
    lines1_cv = cv.computeCorrespondEpilines(np.array(reliable_pts2_12).reshape(-1, 1, 2), 1, F_12_cv.T)
    lines1_cv = lines1_cv.reshape(-1, 3)
    Qf_l_epipolar_line_img_cv_12 = drawlines(left_img, lines1_cv, np.array(reliable_pts1_12))

    left_rect_cv_12, right_rect_cv_12 = rectify_img(Qf_l_epipolar_line_img_cv_12, Qf_r_epipolar_line_img_cv_12,
                                        np.array(reliable_pts1_12), np.array(reliable_pts2_12), F_12_cv)

    rectified_imgs_cv_12 = np.empty(
        (max(left_rect_cv_12.shape[0], right_rect_cv_12.shape[0]), left_rect_cv_12.shape[1] + right_rect_cv_12.shape[1], 3), dtype=np.uint8)
    rectified_imgs_cv_12[0: left_rect_cv_12.shape[0], 0: left_rect_cv_12.shape[1]] = left_rect_cv_12
    rectified_imgs_cv_12[0: right_rect_cv_12.shape[0],
    left_rect_cv_12.shape[1]: left_rect_cv_12.shape[1] + right_rect_cv_12.shape[1]] = right_rect_cv_12
    cv.imwrite("../results/q3f_rectified_cv_12.jpg", rectified_imgs_cv_12)

    # ################ for (l1, l3) pair ####################
    right_img = l_3.copy()
    lines2_cv = cv.computeCorrespondEpilines(np.array(reliable_pts1_13).reshape(-1, 1, 2), 1, F_13_cv)
    lines2_cv = lines2_cv.reshape(-1, 3)
    Qf_r_epipolar_line_img_cv_13 = drawlines(right_img, lines2_cv, np.array(reliable_pts2_13))

    left_img = l_1.copy()
    lines1_cv = cv.computeCorrespondEpilines(np.array(reliable_pts2_13).reshape(-1, 1, 2), 1, F_13_cv.T)
    lines1_cv = lines1_cv.reshape(-1, 3)
    Qf_l_epipolar_line_img_cv_13 = drawlines(left_img, lines1_cv, np.array(reliable_pts1_13))

    left_rect_cv_13, right_rect_cv_13 = rectify_img(Qf_l_epipolar_line_img_cv_13, Qf_r_epipolar_line_img_cv_13,
                                        np.array(reliable_pts1_13), np.array(reliable_pts2_13), F_13_cv)

    rectified_imgs_cv_13 = np.empty(
        (max(left_rect_cv_13.shape[0], right_rect_cv_13.shape[0]), left_rect_cv_13.shape[1] + right_rect_cv_13.shape[1], 3), dtype=np.uint8)
    rectified_imgs_cv_13[0: left_rect_cv_13.shape[0], 0: left_rect_cv_13.shape[1]] = left_rect_cv_13
    rectified_imgs_cv_13[0: right_rect_cv_13.shape[0],
    left_rect_cv_13.shape[1]: left_rect_cv_13.shape[1] + right_rect_cv_13.shape[1]] = right_rect_cv_13
    cv.imwrite("../results/q3f_rectified_cv_13.jpg", rectified_imgs_cv_13)