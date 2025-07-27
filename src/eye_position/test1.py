import math

# === Parameters ===
radius = 1
center_x = 10
center_y = 10

angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]

for angle_deg in angles_deg:
    angle_rad = math.radians(angle_deg)
    x = center_x + radius * math.cos(angle_rad)
    y = center_y + radius * math.sin(angle_rad)
    print(f"Angle {angle_deg}° → x = {x:.6f}, y = {y:.6f}")
