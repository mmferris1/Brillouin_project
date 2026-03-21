import cv2
from pupil_detection_ellipse_points import PupilDetection

def run_on_image(image_path, radius_guess=40):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return

    detector = PupilDetection()
    drawing, center, ellipse, radius, gray, grayblur, threshold, closed = detector.DetectPupil(img, radius_guess)

    print(f"[RESULT] {image_path} → Center: {center}, Radius: {radius:.2f}")

    # Show all processing stages
    cv2.imshow("1 - Grayscale", gray)
    cv2.imshow("2 - Blurred & Inpainted", grayblur)
    cv2.imshow("3 - Thresholded Binary", threshold)
    cv2.imshow("4 - Closed Mask", closed)
    cv2.imshow("5 - Final Annotated Result", drawing)

    key = cv2.waitKey(0)
    if key == 27:  # ESC key
        print("[INFO] Exiting early...")
        exit()

if __name__ == "__main__":
    filename = f"/Users/margotferris/Dropbox/Margot_Brillouin/eye_images/right/pair_0000_right.png"
    run_on_image(filename, radius_guess=40)

    cv2.destroyAllWindows()
