import time

from eye_tracking.devices.connor.dual_allied_vision_cameras import DualAlliedVisionCameras

cams = DualAlliedVisionCameras()
cams.start_stream()
time.sleep(1)  # allow frames to flow
f0, f1 = cams.get_latest_frames()
img0 = f0.as_numpy_ndarray()
img1 = f1.as_numpy_ndarray()
print("Cam0:", img0.shape, "Cam1:", img1.shape)
cams.close()
