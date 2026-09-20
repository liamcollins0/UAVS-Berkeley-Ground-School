import cv2
import os
import numpy as np

def process_color(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    red_result = cv2.bitwise_and(img, img, mask=mask_red)

    y_coords, x_coords = np.where(mask_red > 0)

    if len(x_coords) > 0 and len(y_coords) > 0:
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        print(f"Red Object Center: (X: {center_x}, Y: {center_y})")

        cv2.circle(red_result, (center_x, center_y), 5, (0, 255, 0), -1)

    cv2.imshow("Original Image", img)
    cv2.imshow("Red Object Isolated", red_result)

    print("Press any key to close the images.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    image_path = os.path.join(script_dir, "Haramous_ss.png")

    process_color(image_path)

    