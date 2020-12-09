import cv2
import numpy as np
import pandas as pd
import os
import m20518

DATA_PATH = "data/features.csv"
SAMPLE_VIDEO_PATH = "data/sample.mp4"


def ensure_dataframe(path):
    # 파일이 있으면 read, 없으면 df 새로 생성
    df = pd.DataFrame(
        columns=['object', 'area', 'perimeter', 'width', 'height'])
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="id")
    return df


def scale_tuple(t, scale=2):
    return (t[0] * scale, t[1] * scale)


if __name__ == "__main__":
    df = ensure_dataframe(DATA_PATH)

    # video capture
    cap = cv2.VideoCapture(SAMPLE_VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 모폴로지 연산에 공용으로 사용될 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    frame_cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 건너뛰기
        frame_cnt += 1
        if frame_cnt % 48 != 0:
            continue
        else:
            frame_cnt = 0

        frame_resize = cv2.resize(frame, (320, 240))  # 속도 문제로 리사이즈

        # HSV 중 V채널로 이진화
        frame_value = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2HSV)[..., 2]
        frame_value = cv2.GaussianBlur(frame_value, (5, 5), 0)
        frame_value = cv2.morphologyEx(frame_value, cv2.MORPH_CLOSE, kernel)
        _, frame_th = cv2.threshold(frame_value, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frame_th = m20518.close(frame_th, kernel)

        # 라벨링
        frame_labeled = m20518.label(frame_th)
        frame_outline = (frame_th - cv2.morphologyEx(
            frame_th, cv2.MORPH_ERODE, m20518.kernel)).astype(np.bool)
        frame_labeled_outline = frame_labeled * frame_outline

        frame_temp = np.zeros(frame_th.shape, dtype=np.uint8)
        grid = np.mgrid[:frame_th.shape[0], :frame_th.shape[1]]
        for idx, lbl in enumerate(np.unique(frame_labeled)):
            if not lbl:  # 라벨값이 0이면 무시
                continue

            _i = frame_labeled == lbl

            # Moments
            frame_temp[_i] = 1
            m00 = np.sum(_i)
            m01 = np.sum(grid[0] * frame_temp)
            m10 = np.sum(grid[1] * frame_temp)
            frame_temp[_i] = 0

            # Center
            center = None
            if m00:
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

            # Formfactor
            formfactor = (4 * np.pi * area) / (perimeter**2)

            # dataframe이 비어있으면 area, perimeter 표기
            text = f"{int(area)}, {int(perimeter)}"
            if not df.empty:
                # area와 formfactor를 기준으로 거리 계산
                df['dist'] = df.apply(lambda _df:
                                      (((_df['area'] - area) / 1000)**2 +
                                       (_df['formfactor'] - formfactor)**2),
                                      axis=1)
                # 거리 기준으로 데이터프레임 오름차순 정렬
                df.sort_values(by=['dist'], inplace=True)
                # 거리가 짦은 5개 객체명들의 최빈값
                object = df['object'][:5].mode()[0]
                text = f"{object}"

            # render
            cv2.rectangle(frame, scale_tuple(bbox[0]), scale_tuple(bbox[1]),
                          (255, 0, 255), 2)
            cv2.putText(frame, text, scale_tuple(center), 2, 1, (0, 255, 0))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.imshow('frame', frame)
        cv2.imshow('frame_th', frame_th)

    cap.release()
    cv2.destroyAllWindows()