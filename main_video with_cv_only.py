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


if __name__ == "__main__":
    df = ensure_dataframe(DATA_PATH)

    # video capture
    cap = cv2.VideoCapture(SAMPLE_VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 모폴로지 연산에 공용으로 사용될 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        # HSV 중 V채널로 이진화
        frame_value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
        frame_value = cv2.GaussianBlur(frame_value, (5, 5), 0)
        frame_value = cv2.morphologyEx(frame_value, cv2.MORPH_CLOSE, kernel)
        _, frame_th = cv2.threshold(frame_value, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frame_th = cv2.morphologyEx(frame_th, cv2.MORPH_CLOSE, kernel)

        # contours 찾기
        contours, _ = cv2.findContours(frame_th, 0, cv2.CHAIN_APPROX_NONE)
        for idx, contour in enumerate(contours):
            # 컨투어와 모멘트로 bbox, centroid, area, perimeter, formfactor 계산
            M = cv2.moments(contour)

            # 0이면 건너뛰기
            if not M["m00"]:
                continue

            center = (int(M["m10"] // M["m00"]), int(M["m01"] // M["m00"]))
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            formfactor = 0
            if perimeter:
                formfactor = (4 * np.pi * area) / (perimeter**2)

            # dataframe이 비어있으면 area, perimeter 표기
            text = f"{int(area)}, {int(perimeter)}"
            if df.empty:
                continue

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
            cv2.drawContours(frame, [np.int0(box)], 0, (255, 0, 255), 2)
            cv2.putText(frame, text, center, 2, 1, (0, 255, 0))

        cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()