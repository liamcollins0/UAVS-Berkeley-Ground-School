import cv2
import numpy as np

img = cv2.imread('polkadots.png')
if img is None:
    raise FileNotFoundError("Check image path!")

blurred = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

pixels = hsv.reshape(-1, 3)
unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
dominant_hsv = unique_colors[np.argmax(counts)]

h_tol = 12
s_tol = 50
v_tol = 50

lower_bg = np.array([
    max(0, int(dominant_hsv[0]) - h_tol),
    max(0, int(dominant_hsv[1]) - s_tol),
    max(0, int(dominant_hsv[2]) - v_tol)
], dtype=np.uint8)

upper_bg = np.array([
    min(180, int(dominant_hsv[0]) + h_tol),
    min(255, int(dominant_hsv[1]) + s_tol),
    min(255, int(dominant_hsv[2]) + v_tol)
], dtype=np.uint8)

bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)

dots_mask = cv2.bitwise_not(bg_mask)

kernel = np.ones((5, 5), np.uint8)
dots_mask = cv2.morphologyEx(dots_mask, cv2.MORPH_OPEN, kernel)
dots_mask = cv2.morphologyEx(dots_mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(dots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()
dot_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if perimeter > 0 and 20 < area < 30000:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        if circularity > 0.45:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(output, center, radius, (0, 0, 255), 2)
            dot_count += 1

print(f"Detected {dot_count} polka dots!")

cv2.imshow("Dots Mask", dots_mask)
cv2.imshow("Detected Polka Dots", output)
cv2.waitKey(0)
cv2.destroyAllWindows()