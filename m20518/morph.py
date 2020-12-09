import numpy as np
kernel = np.ones((3, 3))  # 기본 커널


def idx_check(index):
    if index < 0:
        return 0
    else:
        return index


def dilate(img=None, kernel=kernel):
    img = np.asarray(img)
    kernel = np.asarray(kernel)
    krnl_shp = kernel.shape
    img_out = np.zeros((img.shape[0], img.shape[1]))
    krnl_shp_half = ((kernel.shape[0] - 1) // 2, (kernel.shape[1] - 1) // 2)
    for i in range(len(img)):
        for j in range(len(img[0])):
            overlap = img[idx_check(i - krnl_shp_half[0]):i +
                          (krnl_shp[0] - krnl_shp_half[0]),
                          idx_check(j - krnl_shp_half[1]):j +
                          (krnl_shp[1] - krnl_shp_half[1])]
            shp = overlap.shape

            ste_first_row_idx = int(
                np.fabs(i -
                        krnl_shp_half[0])) if i - krnl_shp_half[0] < 0 else 0
            ste_first_col_idx = int(
                np.fabs(j -
                        krnl_shp_half[1])) if j - krnl_shp_half[1] < 0 else 0

            ste_last_row_idx = krnl_shp[0] - 1 - (
                i + (krnl_shp[0] - krnl_shp_half[0]) -
                img.shape[0]) if i + (krnl_shp[0] - krnl_shp_half[0]
                                      ) > img.shape[0] else krnl_shp[0] - 1
            ste_last_col_idx = krnl_shp[1] - 1 - (
                j + (krnl_shp[1] - krnl_shp_half[1]) -
                img.shape[1]) if j + (krnl_shp[1] - krnl_shp_half[1]
                                      ) > img.shape[1] else krnl_shp[1] - 1

            if shp[0] != 0 and shp[1] != 0 and np.logical_and(
                    kernel[ste_first_row_idx:ste_last_row_idx + 1,
                           ste_first_col_idx:ste_last_col_idx + 1],
                    overlap).any():
                img_out[i, j] = 1
    return img_out


def erode(img=None, kernel=kernel):
    img = np.asarray(img)
    kernel = np.asarray(kernel)
    krnl_shp = kernel.shape
    img_out = np.zeros((img.shape[0], img.shape[1]))
    krnl_shp_half = (int(np.ceil(
        (kernel.shape[0] - 1) / 2.0)), int(np.ceil(
            (kernel.shape[1] - 1) / 2.0)))
    for i in range(len(img)):
        for j in range(len(img[0])):
            overlap = img[idx_check(i - krnl_shp_half[0]):i +
                          (krnl_shp[0] - krnl_shp_half[0]),
                          idx_check(j - krnl_shp_half[1]):j +
                          (krnl_shp[1] - krnl_shp_half[1])]
            shp = overlap.shape
            ste_first_row_idx = int(
                np.fabs(i -
                        krnl_shp_half[0])) if i - krnl_shp_half[0] < 0 else 0
            ste_first_col_idx = int(
                np.fabs(j -
                        krnl_shp_half[1])) if j - krnl_shp_half[1] < 0 else 0

            ste_last_row_idx = krnl_shp[0] - 1 - (
                i + (krnl_shp[0] - krnl_shp_half[0]) -
                img.shape[0]) if i + (krnl_shp[0] - krnl_shp_half[0]
                                      ) > img.shape[0] else krnl_shp[0] - 1
            ste_last_col_idx = krnl_shp[1] - 1 - (
                j + (krnl_shp[1] - krnl_shp_half[1]) -
                img.shape[1]) if j + (krnl_shp[1] - krnl_shp_half[1]
                                      ) > img.shape[1] else krnl_shp[1] - 1

            if shp[0] != 0 and shp[1] != 0 and np.array_equal(
                    np.logical_and(
                        overlap,
                        kernel[ste_first_row_idx:ste_last_row_idx + 1,
                               ste_first_col_idx:ste_last_col_idx + 1]),
                    kernel[ste_first_row_idx:ste_last_row_idx + 1,
                           ste_first_col_idx:ste_last_col_idx + 1]):
                img_out[i, j] = 1
    return img_out


def close(img=None, kernel=kernel):
    img = img.copy()
    img = dilate(img, kernel)
    img = erode(img, kernel)
    return img


def open(img=None, kernel=kernel):
    img = img.copy()
    img = erode(img, kernel)
    img = dilate(img, kernel)
    return img