import cv2
import numpy as np
from src.eye_tracking.devices.allied_vision_camera import AlliedVisionCamera

#from allied_vision_camera import AlliedVisionCamera

# Init both cameras
cam0 = AlliedVisionCamera(index=0)
cam1 = AlliedVisionCamera(index=1)

# Set acquisition + trigger mode
for cam in [cam0, cam1]:
    cam.set_acquisition_mode("Continuous")
    cam.set_trigger_mode(mode="On", source="Software")

# Get and queue one frame per camera
frame0 = cam0.camera.get_frame()
frame1 = cam1.camera.get_frame()

cam0.camera.queue_frame(frame0)
cam1.camera.queue_frame(frame1)

# Trigger software capture (quickly, back-to-back)
cam0.camera.run_feature_command("TriggerSoftware")
cam1.camera.run_feature_command("TriggerSoftware")

# Wait for frame to be captured
frame0.wait_for_capture(1000)
frame1.wait_for_capture(1000)

# Convert to numpy
img0 = frame0.as_numpy_ndarray()
img1 = frame1.as_numpy_ndarray()

print("Captured:", img0.shape, img1.shape)

# Shutdown
cam0.close()
cam1.close()
