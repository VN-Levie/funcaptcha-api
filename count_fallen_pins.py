# count_fallen_pins.py
from PIL import Image
import cv2
import numpy as np

def count_fallen_pins(image: Image.Image) -> int:
    """Nhận vào ảnh bowling 200x200 → trả về số pin bị ngã."""
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Làm mịn và threshold để dễ detect blob
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY_INV)

    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc bỏ noise nhỏ (nhỏ hơn 200px)
    pins = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]

    return len(pins)
