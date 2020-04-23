import numpy as np
import scipy.linalg as sp
import cv2 as cv


def two_norm(x):
    return np.sqrt(x.dot(x))


# Compute normalized correlation as matching loss and find the match with the max nc value
def nc_matching_loss(left_p, right_p):
    return np.dot(left_p, right_p) / (two_norm(left_p) * two_norm(right_p))


def ssd_matching_loss(left_p, right_p):
    return np.sum((left_p - right_p) ** 2)


# Q2 a)
# Compute depth of given image
# Z = f * T / (x_l - x_r)
def compute_depth(left_img, right_img, pt1, pt2, focal_len, baseline):
    w = pt2[0] - pt1[0] # x-direction
    h = pt2[1] - pt1[1] # y-direction
    depth_matrix = np.zeros((h, w))
    patch_size = 19
    k = patch_size // 2
    step = 1 # for sample
    print(h, w)

    max_distance = 70

    for r in range(h):
        for c in range(w):
            left_x = pt1[0] + c
            left_y = pt1[1] + r
            # For each (pixels), get a small patch
            left_p = left_img[left_y - k: left_y + k + 1, left_x - k: left_x + k + 1].astype(np.float)
            # scan the line and compare matches
            max_col = min(left_x + 1, right_img.shape[1] - k - 1)
            right_x = list(range(max_col - max_distance, max_col, step))
            # print(np.array(right_x).shape)
            right_p = np.array([right_img[left_y - k: left_y + k + 1, x - k: x + k + 1] for x in right_x]).astype(np.float)
            loss_matrix = np.array([ssd_matching_loss(left_p.flatten(), right_p[p].flatten()) for p in range(len(right_p))])
            # find most similar one
            best_matching = np.argmin(loss_matrix)

            # Compute Z by f * Z / (x_l - x_r)
            depth_matrix[r, c] = float(left_x - right_x[best_matching])

    print depth_matrix.max()
    print depth_matrix.min()

    # depth_matrix /= max_distance
    depth_matrix = np.full(depth_matrix.shape, focal_len * baseline) / (depth_matrix + 1.)
    print depth_matrix.max()
    print depth_matrix.min()
    depth_matrix /= depth_matrix.max()
    depth_matrix = depth_matrix * 255
    depth_matrix = depth_matrix.astype(np.uint8)
    depth_matrix = cv.applyColorMap(cv.GaussianBlur(depth_matrix, (3, 3), 0), cv.COLORMAP_JET)
    print depth_matrix.shape

    return depth_matrix


def compute_depth_full(left_img, right_img, focal_len, baseline):
    h, w, _ = left_img.shape
    depth_matrix = np.zeros((h, w))
    patch_size = 15
    k = patch_size // 2
    step = 1 # for sample
    print(h, w)

    max_distance = 70

    for r in range(k, h - k - 1):
        for c in range(max_distance + k, w - max_distance - k - 1):
            left_x = c
            left_y = r
            left_p = left_img[left_y - k: left_y + k + 1, left_x - k: left_x + k + 1].astype(np.float)
            # scan the line and compare matches
            max_col = min(left_x + 1, right_img.shape[1] - k - 1)
            right_x = list(range(max_col - max_distance, max_col, step))
            right_p = np.array([right_img[left_y - k: left_y + k + 1, x - k: x + k + 1].astype(np.float) for x in right_x])
            loss_matrix = np.array(
                [ssd_matching_loss(left_p.flatten(), right_p[p].flatten()) for p in range(len(right_p))])
            best_matching = np.argmin(loss_matrix)

            # Compute Z by f * T / (x_l - x_r)
            depth_matrix[r, c] = float(left_x - right_x[best_matching])

    print depth_matrix.max()
    print depth_matrix.min()
    # depth_matrix = cv.normalize(np.abs(depth_matrix), depth_matrix, 0, 1., cv.NORM_MINMAX)
    # depth_matrix /= max_distance
    depth_matrix = np.full(depth_matrix.shape, focal_len * baseline) / (depth_matrix + 1.)
    # depth_matrix = depth_matrix ** (1. / 2.)
    # depth_matrix = cv.normalize(depth_matrix, depth_matrix, 0, 255., cv.NORM_MINMAX)
    # depth_matrix = np.full(depth_matrix.shape, focal_len * baseline) / (depth_matrix + 1.)
    # depth_matrix = cv.normalize(depth_matrix, depth_matrix, 0, 255., cv.NORM_MINMAX)
    print depth_matrix.max()
    print depth_matrix.min()
    depth_matrix /= depth_matrix.max()
    depth_matrix = depth_matrix * 255
    depth_matrix = depth_matrix.astype(np.uint8)
    depth_matrix = cv.applyColorMap(cv.GaussianBlur(depth_matrix, (3, 3), 0), cv.COLORMAP_JET)
    return depth_matrix


def predict_car(depth, pt1, pt2, f, baseline, px, py):
    w = pt2[0] - pt1[0] # x-direction
    h = pt2[1] - pt1[1] # y-direction
    seg_map = np.zeros((h, w))
    print(h, w)
    center_x = pt1[0] + w // 2
    center_y = pt1[1] + h // 2
    center_Z = (f * baseline) / (depth[center_y, center_x])
    center_3d = compute_3d_position(center_x, center_y, px, py, center_Z, f)
    print('2d center point: ({}, {})'.format(center_x, center_y))
    print('3d center point: {}'.format(center_3d))
    t = 1.3
    print('t: {}'.format(t))
    min_X = np.inf
    min_Y = np.inf
    min_Z = np.inf
    max_X = 0
    max_Y = 0
    max_Z = 0
    for r in range(h):
        for c in range(w):
            x = pt1[0] + c
            y = pt1[1] + r
            d = (f * baseline) / (depth[y, x] + 0.1)
            pos_3d = compute_3d_position(x, y, px, py, d, f)
            if compute_distance(pos_3d, center_3d) < t:
                # this point is in car
                seg_map[r, c] = 1
                if pos_3d[0] < min_X: min_X = pos_3d[0]
                if pos_3d[1] < min_Y: min_Y = pos_3d[1]
                if pos_3d[2] < min_Z: min_Z = pos_3d[2]
                if pos_3d[0] > max_X: max_X = pos_3d[0]
                if pos_3d[1] > max_Y: max_Y = pos_3d[1]
                if pos_3d[2] > max_Z: max_Z = pos_3d[2]

    return seg_map, (min_X, min_Y, min_Z), (max_X, max_Y, max_Z)


# predict car segmentation for the full image
def predict_car_full(depth, pt1, pt2, f, baseline, px, py):
    w = pt2[0] - pt1[0] # x-direction
    h = pt2[1] - pt1[1] # y-direction
    seg_map = np.zeros((depth.shape[0], depth.shape[1]))
    print(h, w)
    center_x = pt1[0] + w // 2
    center_y = pt1[1] + h // 2
    center_Z =(f * baseline) / (depth[center_y, center_x] + 0.1)
    center_3d = compute_3d_position(center_x, center_y, px, py, center_Z, f)
    print('2d center point: ({}, {})'.format(center_x, center_y))
    print('3d center point: {}'.format(center_3d))
    t = 1.3
    print('t: {}'.format(t))
    min_X = np.inf
    min_Y = np.inf
    min_Z = np.inf
    max_X = 0
    max_Y = 0
    max_Z = 0

    for r in range(depth.shape[0]):
        for c in range(depth.shape[1]):
            x = c
            y = r
            d = (f * baseline) / (depth[y, x] + 0.1)
            pos_3d = compute_3d_position(x, y, px, py, d, f)
            if compute_distance(pos_3d, center_3d) < t:
                # this point is in car
                seg_map[r, c] = 1
                if pos_3d[0] < min_X: min_X = pos_3d[0]
                if pos_3d[1] < min_Y: min_Y = pos_3d[1]
                if pos_3d[2] < min_Z: min_Z = pos_3d[2]
                if pos_3d[0] > max_X: max_X = pos_3d[0]
                if pos_3d[1] > max_Y: max_Y = pos_3d[1]
                if pos_3d[2] > max_Z: max_Z = pos_3d[2]

    return seg_map, (min_X, min_Y, min_Z), (max_X, max_Y, max_Z)


def compute_3d_position(x, y, px, py, Z, f):
    X = float(Z * (x - px)) / f
    Y = float(Z * (y - py)) / f
    return np.array([X, Y, Z])


def compute_2d_position(X, Y, Z, px, py, f):
    x = int((f * X) / Z + px)
    y = int((f * Y) / Z + py)
    return (x, y)


def compute_distance(pt1, pt2):
    return np.sqrt(np.sum((pt1 - pt2) ** 2))


if __name__ == '__main__':

    # Q2 a)
    left = cv.imread('../A4_files/000020_left.jpg')
    right = cv.imread('../A4_files/000020_right.jpg')

    # read bounding box info
    f_box = open("../A4_files/000020.txt", "r")
    fl_box = f_box.readlines()
    _, x1, y1, x2, y2 = fl_box[0].split()
    print(x1, y1, x2, y2)
    pt1 = (int(float(x1)), int(float(y1)))
    pt2 = (int(float(x2)), int(float(y2)))

    # read camera calibration info
    f = 0
    px = 0
    py = 0
    baseline = 0
    f_cal = open("../A4_files/000020_allcalib.txt", "r")
    fl_cal = f_cal.readlines()
    for l in fl_cal:
        print(l.split())
        label, n = l.split()
        if label == "f:":
            f = float(n)
        elif label == "px:":
            px = float(n)
        elif label == "py:":
            py = float(n)
        elif label == "baseline:":
            baseline = float(n)

    print(f, px, py, baseline)

    # Draw bounding box onto image
    left_box = left.copy()
    cv.rectangle(left_box, pt1, pt2, (255, 0, 255), 2)
    cv.imwrite("../results/left_box.jpg", left_box)
    box_img = left.copy()
    box_img = left[pt1[1]: pt2[1], pt1[0]: pt2[0]]
    cv.imwrite("../results/left_box_img.jpg", box_img)
    # print(pt1, pt2)

    # depth = compute_depth_full(left, right, f, baseline)
    # cv.imwrite("../results/q2a_full_depth_matrix.jpg", depth)

    depth_box = compute_depth(left, right, pt1, pt2, f, baseline)
    depth_img = np.empty((box_img.shape[0], box_img.shape[1] + depth_box.shape[1], 3), dtype=np.uint8)
    depth_img[0: box_img.shape[0], 0: box_img.shape[1]] = box_img
    depth_img[0: box_img.shape[0], box_img.shape[1]: box_img.shape[1] + depth_box.shape[1]] = depth_box
    cv.imwrite("../results/q2a_depth_compare.jpg", depth_img)


    # Q2 d)
    print('##########################################')
    depth = cv.imread('../results/000020_GANet.jpg', 0)
    print depth.max()
    print depth
    depth_c = depth.copy()
    depth_c = cv.normalize(depth_c.astype(float), depth_c, 0, 1., cv.NORM_MINMAX)
    depth_c = depth_c ** (1. / 0.5)
    depth_c = cv.normalize(depth_c, depth_c, 0, 255., cv.NORM_MINMAX).astype(np.uint8)
    cv.imwrite("../results/q2d_depth_copy.jpg", depth_c)

    # seg, min_pt, max_pt = predict_car(depth, pt1, pt2, f, baseline, px, py)
    # seg_img = np.zeros((seg.shape[0], seg.shape[1], 3))
    # seg_img[:, :, 0] = seg * 255
    # seg_img[:, :, 1] = seg * 255
    # seg_img[:, :, 2] = seg * 255
    # print min_pt
    # print max_pt
    # car_seg_predict = np.empty((box_img.shape[0], box_img.shape[1] + seg_img.shape[1], 3), dtype=np.uint8)
    # car_seg_predict[0: box_img.shape[0], 0: box_img.shape[1]] = box_img
    # car_seg_predict[0: box_img.shape[0], box_img.shape[1]: box_img.shape[1] + seg_img.shape[1]] = seg_img
    # cv.imwrite("../results/q2d_car_predict_segmentation.jpg", car_seg_predict)


    seg, min_pt, max_pt = predict_car_full(depth, pt1, pt2, f, baseline, px, py)
    car_seg_predict_full = np.zeros((seg.shape[0], seg.shape[1], 3))
    car_seg_predict_full[:, :, 0] = seg * 255
    car_seg_predict_full[:, :, 1] = seg * 255
    car_seg_predict_full[:, :, 2] = seg * 255
    cv.imwrite("../results/q2d_car_predict_segmentation_full.jpg", car_seg_predict_full)

    bounding_box_3d = left.copy()

    x1 = compute_2d_position(min_pt[0], min_pt[1], min_pt[2], px, py, f)
    x2 = compute_2d_position(max_pt[0], min_pt[1], min_pt[2], px, py, f)
    x3 = compute_2d_position(max_pt[0], max_pt[1], min_pt[2], px, py, f)
    x4 = compute_2d_position(min_pt[0], max_pt[1], min_pt[2], px, py, f)
    x5 = compute_2d_position(min_pt[0], min_pt[1], max_pt[2], px, py, f)
    x6 = compute_2d_position(max_pt[0], min_pt[1], max_pt[2], px, py, f)
    x7 = compute_2d_position(max_pt[0], max_pt[1], max_pt[2], px, py, f)
    x8 = compute_2d_position(min_pt[0], max_pt[1], max_pt[2], px, py, f)
    print('x1: ({}, {})'.format(x1[0], x1[1]))
    print('x4: ({}, {})'.format(x4[0], x4[1]))
    print('x7: ({}, {})'.format(x7[0], x7[1]))
    color = (255, 255, 0)
    cv.line(bounding_box_3d, x1, x2, color, 2)
    cv.line(bounding_box_3d, x2, x3, color, 2)
    cv.line(bounding_box_3d, x1, x4, color, 2)
    cv.line(bounding_box_3d, x3, x4, color, 2)
    cv.line(bounding_box_3d, x1, x5, color, 2)
    cv.line(bounding_box_3d, x2, x6, color, 2)
    cv.line(bounding_box_3d, x4, x8, color, 2)
    cv.line(bounding_box_3d, x5, x6, color, 2)
    cv.line(bounding_box_3d, x5, x8, color, 2)

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(bounding_box_3d, str(1), x1, font, .5, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(bounding_box_3d, str(2), x2, font, .5, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(bounding_box_3d, str(3), x3, font, .5, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(bounding_box_3d, str(4), x4, font, .5, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(bounding_box_3d, str(5), x5, font, .5, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(bounding_box_3d, str(6), x6, font, .5, (0, 0, 255), 2, cv.LINE_AA)
    # cv.putText(bounding_box_3d, str(7), x7, font, .5, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(bounding_box_3d, str(8), x8, font, .5, (0, 0, 255), 2, cv.LINE_AA)
    cv.imwrite("../results/q2d_car_predict_3d_bounding_box.jpg", bounding_box_3d)

