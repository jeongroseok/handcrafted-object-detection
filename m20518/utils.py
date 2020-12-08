import numpy as np


def label(img):
    img = np.pad(img, 1)
    img_out = np.zeros(img.shape)
    label = 1
    for y in range(1, img.shape[0]):
        for x in range(1, img.shape[1] - 1):
            if img[y, x] == 0:
                continue
            nb = np.append(img_out[y - 1, x - 1:x + 2], img_out[y, x - 1])
            nb = np.unique(nb)
            nb = np.sort(nb)
            if np.sum(nb) == 0:
                img_out[y, x] = label
                label = label + 1
            else:
                nb_labels = nb[np.nonzero(nb)]
                min_label = nb_labels[0]
                img_out[y, x] = min_label
                if len(nb_labels) <= 1:
                    continue
                for nb_label in nb_labels[1:]:
                    img_out[img_out == nb_label] = min_label
    img_out = img_out[1:-1, 1:-1]
    return img_out


def calc_area(img, label_idx):
    return len(img[img == label_idx])


def calc_bbox(img, label_idx):
    tmp = np.zeros(img.shape)
    tmp[img == label_idx] = 1
    idx_x = np.nonzero(tmp.any(axis=0))[0]
    idx_y = np.nonzero(tmp.any(axis=1))[0]
    start = (idx_x[0], idx_y[0])
    end = (idx_x[-1], idx_y[-1])
    return start, end