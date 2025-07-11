import cv2
#from src.eye_tracking.pupil_detection import PupilDetection
from src.eye_tracking.pupil_detection_laser_focus import PupilDetection


def run_on_image(image_path, radius_guess=100):
    # Load image from file using cv2
    image = cv2.imread(image_path)

    print(f"Image shape: {image.shape}")

    # Create PupilDetection instantiation
    PD = PupilDetection()

    # Run detection
    result_image, center, ellispe, pupil_radius = PD.DetectPupil(image, radius_guess)

    if ellispe is None:
        print(f"No pupil detected in {image_path}")
    else:
        print(f"Pupil center: {center}")
        print(f"Ellipse size: {ellispe[1]}")
        print(f"Pupil radius: {pupil_radius}")

    # print(f"Pupil radius : {pupil_radius}")

    # Show result
    cv2.imshow("Pupil Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Loop through all images in folder and call run_on_image function
for i in range(1, 11):  # Replace N with the total number of images you have
    filename = f"/Users/margaretferris/Desktop/dummyeye/left/left_dummyeye_{i}.png"
    run_on_image(filename, radius_guess=40)

#filename = f"/Users/margaretferris/Desktop/take7/left.bmp"
#run_on_image(filename, radius_guess=40)