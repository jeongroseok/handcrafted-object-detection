import cv2
import numpy as np
import pandas as pd
import os

KELLY_COLORS = [
    'F2F3F4', '222222', 'F3C300', '875692', 'F38400', 'A1CAF1', 'BE0032',
    'C2B280', '848482', '008856', 'E68FAC', '0067A5', 'F99379', '604E97',
    'F6A600', 'B3446C', 'DCD300', '882D17', '8DB600', '654522', 'E25822',
    '2B3D26'
]

DATA_PATH = "data.csv"


def hex2rgb(hex):
    return tuple([int(hex[i:i + 2], 16) for i in (0, 2, 4)])


if __name__ == "__main__":
    df = pd.DataFrame(columns=['object', 'area', 'perimeter'])
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, index_col="id")

    object = input()
    print(object)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
        frame_value = cv2.GaussianBlur(frame_value, (5, 5), 0)
        frame_value = cv2.morphologyEx(frame_value, cv2.MORPH_CLOSE, kernel)
        _, frame_th = cv2.threshold(frame_value, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frame_th = cv2.morphologyEx(frame_th, cv2.MORPH_CLOSE, kernel)

        ret, labels = cv2.connectedComponents(frame_th)
        frame_label = np.zeros(frame_th.shape + (3, ), dtype=np.uint8)
        for idx in range(ret):
            if not idx:
                continue
            if idx >= len(KELLY_COLORS):
                break
            frame_label[labels == idx] = hex2rgb(KELLY_COLORS[idx])

        contours, _ = cv2.findContours(frame_th, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        key = cv2.waitKey(1)
        for idx, contour in enumerate(contours):
            cv2.drawContours(frame_label, [contour], 0, (255, 0, 0), 3)
            M = cv2.moments(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            cv2.imshow(
                'box',
                frame_label[int(rect[0][1]):int(rect[0][1]) + int(rect[1][1]),
                            int(rect[0][0]):int(rect[0][0]) + int(rect[1][0])])
            cv2.drawContours(frame_label, [np.int0(box)], 0, (0, 0, 255), 3)
            try:
                center = (int(M['m10'] // M['m00']), int(M['m01'] // M['m00']))
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                df['dist'] = df.apply(lambda _df:
                                      ((_df['area'] - area)**2 +
                                       (_df['perimeter'] - perimeter)**2),
                                      axis=1)
                df.sort_values(by=['dist'], inplace=True)
                text = f"{df['object'][:15].mode()[0]}:{int(area)}, {int(perimeter)}"
                cv2.putText(frame_label, text, center, 0, 1, (0, 255, 0))

                if key & 0xFF == ord(' '):
                    df = df.append(
                        {
                            "object": object,
                            "area": area,
                            "perimeter": perimeter
                        },
                        ignore_index=True)
            except:
                pass

        cv2.imshow('pre', frame)
        cv2.imshow('th', frame_th)
        cv2.imshow('post', frame_label)

        if key & 0xFF == ord('q'):
            break

    df.drop(['dist'], axis=1, inplace=True)
    df.to_csv(DATA_PATH, index_label="id")
    cap.release()
    cv2.destroyAllWindows()