import numpy as np
default_structure = np.ones((3, 3))


def idx_check(index):
    if index < 0:
        return 0
    else:
        return index


def dilate(img=None, kernel=default_structure):
    img = np.asarray(img)
    kernel = np.asarray(kernel)
    ste_shp = kernel.shape
    img_out = np.zeros((img.shape[0], img.shape[1]))
    ste_origin = ((kernel.shape[0] - 1) // 2, (kernel.shape[1] - 1) // 2)
    for i in range(len(img)):
        for j in range(len(img[0])):
            overlap = img[idx_check(i - ste_origin[0]):i +
                          (ste_shp[0] - ste_origin[0]),
                          idx_check(j - ste_origin[1]):j +
                          (ste_shp[1] - ste_origin[1])]
            shp = overlap.shape

            ste_first_row_idx = int(
                np.fabs(i - ste_origin[0])) if i - ste_origin[0] < 0 else 0
            ste_first_col_idx = int(
                np.fabs(j - ste_origin[1])) if j - ste_origin[1] < 0 else 0

            ste_last_row_idx = ste_shp[0] - 1 - (
                i + (ste_shp[0] - ste_origin[0]) -
                img.shape[0]) if i + (ste_shp[0] - ste_origin[0]
                                      ) > img.shape[0] else ste_shp[0] - 1
            ste_last_col_idx = ste_shp[1] - 1 - (
                j + (ste_shp[1] - ste_origin[1]) -
                img.shape[1]) if j + (ste_shp[1] - ste_origin[1]
                                      ) > img.shape[1] else ste_shp[1] - 1

            if shp[0] != 0 and shp[1] != 0 and np.logical_and(
                    kernel[ste_first_row_idx:ste_last_row_idx + 1,
                           ste_first_col_idx:ste_last_col_idx + 1],
                    overlap).any():
                img_out[i, j] = 1
    return img_out


def erode(img=None, kernel=default_structure):
    img = np.asarray(img)
    kernel = np.asarray(kernel)
    ste_shp = kernel.shape
    img_out = np.zeros((img.shape[0], img.shape[1]))
    ste_origin = (int(np.ceil(
        (kernel.shape[0] - 1) / 2.0)), int(np.ceil(
            (kernel.shape[1] - 1) / 2.0)))
    for i in range(len(img)):
        for j in range(len(img[0])):
            overlap = img[idx_check(i - ste_origin[0]):i +
                          (ste_shp[0] - ste_origin[0]),
                          idx_check(j - ste_origin[1]):j +
                          (ste_shp[1] - ste_origin[1])]
            shp = overlap.shape
            ste_first_row_idx = int(
                np.fabs(i - ste_origin[0])) if i - ste_origin[0] < 0 else 0
            ste_first_col_idx = int(
                np.fabs(j - ste_origin[1])) if j - ste_origin[1] < 0 else 0

            ste_last_row_idx = ste_shp[0] - 1 - (
                i + (ste_shp[0] - ste_origin[0]) -
                img.shape[0]) if i + (ste_shp[0] - ste_origin[0]
                                      ) > img.shape[0] else ste_shp[0] - 1
            ste_last_col_idx = ste_shp[1] - 1 - (
                j + (ste_shp[1] - ste_origin[1]) -
                img.shape[1]) if j + (ste_shp[1] - ste_origin[1]
                                      ) > img.shape[1] else ste_shp[1] - 1

            if shp[0] != 0 and shp[1] != 0 and np.array_equal(
                    np.logical_and(
                        overlap,
                        kernel[ste_first_row_idx:ste_last_row_idx + 1,
                               ste_first_col_idx:ste_last_col_idx + 1]),
                    kernel[ste_first_row_idx:ste_last_row_idx + 1,
                           ste_first_col_idx:ste_last_col_idx + 1]):
                img_out[i, j] = 1
    return img_out
