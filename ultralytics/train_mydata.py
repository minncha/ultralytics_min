import os
import re
import shutil

import matplotlib

from ultralytics import YOLO

matplotlib.use('Agg')


def save_pt(dataset, list):
    path = '../runs/detect/'
    destination_path = f'./pt/runs/{dataset}/'  # 복사할 폴더 지정
    new_filename = f'{dataset}_{list[4:]}'  # 새로운 파일 이름
    os.makedirs(destination_path, exist_ok=True)

    # 폴더 목록 가져오기
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    train_folders = [f for f in folders if re.match(r'train\d+', f)]

    # 숫자 부분을 추출하여 가장 큰 값 찾기
    if train_folders:
        latest_train = max(train_folders, key=lambda x: int(re.search(r'\d+', x).group()))
        latest_train_path = os.path.join(path, latest_train)
        best_pt_path = os.path.join(latest_train_path, 'weights/best.pt')
        new_file_path = os.path.join(destination_path, new_filename)
        shutil.copy(best_pt_path, new_file_path)
    else:
        print("train 폴더가 존재하지 않습니다.")


def train_YOLO(dataset, model_list):
    prefix = './pt/yolo'
    dataset_path = r'D:/test_code/dataset'
    for list in model_list:
        pt = os.path.join(prefix, list)
        model = YOLO(pt)  # load a pretrained model (recommended for training)
        model.train(data=f"{dataset_path}/{dataset}/{dataset}.yaml", epochs=5, imgsz=640)
        save_pt(dataset, list)


if __name__ == '__main__':
    dataset_list = [
        'min_200',
    ]
    model_list1 = [
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt",
        "yolo11x.pt",
        "yolov10n.pt",
        "yolov10s.pt",
        "yolov10m.pt",
        "yolov10l.pt",
        "yolov10x.pt",
        "yolov9t.pt",
        "yolov9s.pt",
        "yolov9m.pt",
        "yolov9l.pt",
        "yolov9x.pt",
        "yolov8n.pt",
        "yolov8s.pt"
        "yolov8m.pt"
        "yolov8l.pt"
        "yolov8x.pt"
        "yolov6n.yaml",
        "yolov6s.yaml",
        "yolov6m.yaml",
        "yolov6l.yaml",
        "yolov6x.yaml",
        "yolov5n.pt",
        "yolov5s.pt",
        "yolov5m.pt",
        "yolov5l.pt",
        "yolov5x.pt",
    ]

    for d in dataset_list:
        train_YOLO(d, model_list1)
