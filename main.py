import cv2
import numpy as np
import pandas as pd

VIDEO_PATH = "data/test.mp4"
KELLY_COLORS = [
    'F2F3F4', '222222', 'F3C300', '875692', 'F38400', 'A1CAF1', 'BE0032',
    'C2B280', '848482', '008856', 'E68FAC', '0067A5', 'F99379', '604E97',
    'F6A600', 'B3446C', 'DCD300', '882D17', '8DB600', '654522', 'E25822',
    '2B3D26'
]


def hex2rgb(hex):
    return tuple([int(hex[i:i + 2], 16) for i in (0, 2, 4)])


if __name__ == "__main__":
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
        frame_value = cv2.GaussianBlur(frame_value, (5, 5), 0)
        _, frame_value = cv2.threshold(frame_value, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frame_value = cv2.morphologyEx(frame_value, cv2.MORPH_CLOSE, kernel)

        ret, labels = cv2.connectedComponents(frame_value)
        frame_label = np.zeros(frame_value.shape + (3, ), dtype=np.uint8)
        for idx in range(ret):
            if not idx:
                continue
            frame_label[labels == idx] = hex2rgb(KELLY_COLORS[idx])

        contours, _ = cv2.findContours(frame_value, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        for idx, contour in enumerate(contours):
            cv2.drawContours(frame_label, [contour], 0, (255, 0, 0), 3)
            M = cv2.moments(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            cv2.drawContours(frame_label, [np.int0(box)], 0, (0, 0, 255), 3)

        cv2.imshow('pre', frame)
        cv2.imshow('post', frame_label)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def hex2rgb(hex):
    return [int(KELLY_COLORS[idx][i:i + 2], 16) for i in (0, 2, 4)]
