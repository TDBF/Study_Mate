import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# 데이터 경로 설정
source_folder = 'source_folder'  # 원천 데이터 폴더 경로
label_folder = 'label_folder'    # 라벨링 데이터 폴더 경로

# 이미지 크기 및 클래스 수 정의 (수정 필요)
WIDTH = 28
HEIGHT = 28
NUM_CLASSES = 10  # 예시: 10개의 클래스

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(source_folder, label_file_path):
    images = []
    labels = []

    with open(label_file_path, 'r') as file:
        label_data = json.load(file)
        image_file_name = label_data['image']['filename']
        image_path = os.path.join(source_folder, image_file_name)
        source_image = cv2.imread(image_path)

        for label_info in label_data['annotation']['labels']:
            label_name = label_info['name']
            for segment in label_data['annotation']['segments']:
                x_min, y_min = map(int, segment['box'][0])
                x_max, y_max = map(int, segment['box'][2])
                roi = source_image[y_min:y_max, x_min:x_max]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (WIDTH, HEIGHT))
                normalized = resized / 255.0
                images.append(normalized)
                labels.append(label_name)

    return np.array(images), np.array(labels)


# 이미지와 라벨을 로드하고 전처리
X, y = load_and_preprocess_data(source_folder, label_folder)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
epochs = 10
batch_size = 32
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
