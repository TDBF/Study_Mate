import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread('ex5.png')

# 이미지를 그레이 스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이미지를 이진화
_, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# 이미지에서 텍스트를 추출
text = pytesseract.image_to_string(binary_image)

print('Extracted Text:', text)
