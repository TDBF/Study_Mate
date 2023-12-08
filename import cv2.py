import cv2

# 이미지 파일 경로
image_path = 'path/to/your/image.jpg'

# 이미지 읽기
image = cv2.imread(image_path)

# 이미지가 제대로 읽혔는지 확인
if image is None:
    print(f"Error: Unable to read the image from {image_path}")
else:
    # 이미지를 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 여기에서 이미지 처리 및 다음 단계 수행
