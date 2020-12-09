import cv2
import numpy as np
import pandas as pd
import os
import m20518

KELLY_COLORS = [
    'F2F3F4', '222222', 'F3C300', '875692', 'F38400', 'A1CAF1', 'BE0032',
    'C2B280', '848482', '008856', 'E68FAC', '0067A5', 'F99379', '604E97',
    'F6A600', 'B3446C', 'DCD300', '882D17', '8DB600', '654522', 'E25822',
    '2B3D26'
]

DATA_PATH = "data.csv"


def scale_tuple(t, scale=2):
    return (t[0] * scale, t[1] * scale)


def hex2rgb(hex):
    return tuple([int(hex[i:i + 2], 16) for i in (0, 2, 4)])


if __name__ == "__main__":
    count = 20
    # print('object: ')
    # object = input()
    # print('check: ', object)

    df = pd.DataFrame(
        columns=['object', 'area', 'perimeter', 'width', 'height'])
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, index_col="id")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        frame_resize = cv2.resize(frame, (320, 240))
        if not ret:
            break

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        # OpenCV
        frame_value = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2HSV)[..., 2]
        frame_value = cv2.GaussianBlur(frame_value, (5, 5), 0)
        frame_value = cv2.morphologyEx(frame_value, cv2.MORPH_CLOSE, kernel)
        _, frame_th = cv2.threshold(frame_value, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Implemented
        # frame_th = m20518.close(frame_th, kernel)  # impl
        frame_th = cv2.morphologyEx(frame_th, cv2.MORPH_CLOSE, kernel)  # cv
        frame_labeled = m20518.label(frame_th)
        # frame_outline = (frame_th - m20518.erode(frame_th)).astype(
        #     np.bool)  # impl
        frame_outline = (frame_th - cv2.morphologyEx(
            frame_th, cv2.MORPH_ERODE, m20518.kernel)).astype(np.bool)  # cv
        frame_labeled_outline = frame_labeled * frame_outline

        frame_colored = np.zeros(frame_th.shape + (3, ), dtype=np.uint8)
        frame_temp = np.zeros(frame_th.shape, dtype=np.uint8)
        grid = np.mgrid[:frame_th.shape[0], :frame_th.shape[1]]
        for idx, lbl in enumerate(np.unique(frame_labeled)):
            if not lbl:  # 라벨값이 0이면 무시
                continue

            # Apply kelly color
            if idx >= len(KELLY_COLORS):  # 켈리 색상이 부족하면 무시
                break
            _i = frame_labeled == lbl
            frame_colored[_i] = hex2rgb(KELLY_COLORS[idx])

            # Moments
            frame_temp[_i] = 1
            m00 = np.sum(_i)
            m01 = np.sum(grid[0] * frame_temp)
            m10 = np.sum(grid[1] * frame_temp)
            frame_temp[_i] = 0

            # Center
            center = 0
            if m00 and m01 and m10:
                center = (m10 // m00, m01 // m00)
            else:
                continue

            # Area, Perimeter
            area = m20518.calc_area(frame_labeled, lbl)
            perimeter = m20518.calc_area(frame_labeled_outline, lbl)

            # Bounding Box
            bbox = m20518.calc_bbox(frame_labeled, lbl)
            cv2.rectangle(frame, scale_tuple(bbox[0]), scale_tuple(bbox[1]),
                          (255, 0, 0), 1)

            # Width, Height
            width = bbox[1][0] - bbox[0][0]
            height = bbox[1][1] - bbox[0][1]

            # Formfactor, Aspect ratio
            formfactor = (4 * np.pi * area) / (perimeter**2)
            aspect_ratio = width / height

            # Distance between saved features and current features
            text = f"{int(area)}, {int(perimeter)}"
            if not df.empty:
                df['dist'] = df.apply(lambda _df:
                                      (((_df['area'] - area) / 1000)**2 +
                                       (_df['formfactor'] - formfactor)**2),
                                      axis=1)
                df.sort_values(by=['dist'], inplace=True)
                text = f"{df['object'][:5].mode()[0]}"
            cv2.putText(frame, text, scale_tuple(center), 0, 1, (0, 0, 255))

            # # Save features
            # if key & 0xFF == ord(' '):
            #     df = df.append(
            #         {
            #             "object": object,
            #             "area": area,
            #             "perimeter": perimeter,
            #             "width": width,
            #             "height": height,
            #             "formfactor": formfactor,
            #             "aspect_ratio": aspect_ratio
            #         },
            #         ignore_index=True)
            #     count -= 1
            # print(count)

        cv2.imshow('frame', frame)
        cv2.imshow('frame_colored', frame_colored)

    # df.to_csv(DATA_PATH, index_label="id")
    cap.release()
    cv2.destroyAllWindows()