import cv2
import numpy as np

img = cv2.imread('assorted_goods.png')
if img is None:
    raise FileNotFoundError("Check image path!")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

color_classes = {
    "Cone": {
        "lower": np.array([15, 140, 120], dtype=np.uint8),
        "upper": np.array([35, 255, 255], dtype=np.uint8),
        "box_color": (0, 255, 255)
    },
    "Cube": {
        "lower": np.array([110, 70, 50], dtype=np.uint8),
        "upper": np.array([160, 255, 255], dtype=np.uint8),
        "box_color": (255, 0, 255)
    },
    "Ring": {
        "lower": np.array([0, 120, 100], dtype=np.uint8),
        "upper": np.array([14, 255, 255], dtype=np.uint8),
        "box_color": (0, 165, 255)
    }
}

output = img.copy()

kernel_small = np.ones((3, 3), np.uint8)
kernel_large = np.ones((7, 7), np.uint8)

for label, info in color_classes.items():
    mask = cv2.inRange(hsv, info["lower"], info["upper"])
    
    if label == "Cube":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_large)
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 250:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), info["box_color"], 2)
            cv2.putText(
                output, 
                label, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                info["box_color"], 
                2
            )

cv2.imshow("Detected Objects", output)
cv2.waitKey(0)
cv2.destroyAllWindows()