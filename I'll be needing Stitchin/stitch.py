import cv2
import numpy as np
import os

def extract_frames(video_path, sample_interval=30, scale_factor=0.3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            if scale_factor != 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            frames.append(frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {len(frames)} frames.")
    return frames

def compute_homographies(frames):
    """
    Computes cumulative Homography matrices relative to the FIRST frame.
    """
    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # First frame matrix is Identity matrix (no transform)
    H_cumulative = [np.eye(3, dtype=np.float32)]

    for i in range(len(frames) - 1):
        img1 = frames[i]
        img2 = frames[i + 1]

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            print(f"Frame {i} to {i+1}: Not enough keypoints detected.")
            break

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 4:
            print(f"Frame {i} to {i+1}: Not enough matches found.")
            break

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:60]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:60]]).reshape(-1, 1, 2)

        # Compute transform from frame i+1 to frame i
        H_step, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H_step is None:
            print(f"Frame {i} to {i+1}: Homography calculation failed.")
            break

        # Chain transformation back to the base coordinate frame
        H_curr = np.dot(H_cumulative[-1], H_step)
        H_cumulative.append(H_curr)

    return H_cumulative

def stitch_frames_global(frames, H_matrices):
    if not frames or not H_matrices:
        return None

    # Find the bounding box of the entire stitched canvas
    all_corners = []
    for img, H in zip(frames, H_matrices):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped_corners)

    all_corners = np.vstack(all_corners)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Shift translation matrix so negative coordinates map to positive canvas indices
    H_translation = np.array([[1, 0, -x_min],
                              [0, 1, -y_min],
                              [0, 0, 1]], dtype=np.float32)

    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Warp each frame to its proper place on the joint canvas
    for img, H in zip(frames, H_matrices):
        H_final = np.dot(H_translation, H)
        warped = cv2.warpPerspective(img, H_final, (canvas_width, canvas_height))
        
        # Overlay non-zero pixels
        mask = (warped > 0)
        panorama[mask] = warped[mask]

    return panorama

def run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "Minecraft_stitch_test.mp4")

    if not os.path.exists(video_path):
        print(f"Video file not found at {video_path}")
        return

    # Extract frames
    frames = extract_frames(video_path, sample_interval=20, scale_factor=0.3)
    if len(frames) < 2:
        return

    # Calculate cumulative motion across frames
    H_matrices = compute_homographies(frames)

    # Render global canvas
    print("Stitching all frames onto dynamic canvas...")
    result = stitch_frames_global(frames[:len(H_matrices)], H_matrices)

    if result is not None:
        out_path = os.path.join(script_dir, "stitched_output.jpg")
        cv2.imwrite(out_path, result)
        print(f"Done! Saved stitched panorama to {out_path}")
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()