import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

# 1. 모델 및 전처리 함수 초기화
model = mobilenet_v2(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# 2. 이미지 로드 및 전처리
image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)  # 모델은 배치 차원을 기대하므로 차원을 추가합니다.

# 3. 모델 추론
with torch.no_grad():
    output = model(input_batch)

# 4. 결과 해석
_, predicted_idx = torch.max(output, 1)
predicted_label = predicted_idx.item()

# 여기에서는 ImageNet의 클래스 인덱스를 사용하므로 클래스 레이블을 가져오는 코드가 필요합니다.
# ImageNet의 클래스 레이블을 사용하려면 추가적인 작업이 필요합니다.

print(f"Predicted Label: {predicted_label}")
