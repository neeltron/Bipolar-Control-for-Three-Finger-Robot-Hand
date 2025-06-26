import os
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
import mujoco

import mujoco.viewer
import pygame
import sys

# ----------------------------------------
# Joystick zone mapping logic
def map_to_zone(x, y, prefix):
    if -0.2 <= x <= 0.2 and -0.2 <= y <= 0.2:
        return f"{prefix}0"
    elif -1 <= x <= -0.7 and 0.7 <= y <= 1:
        return f"{prefix}1"
    elif -0.3 <= x <= 0.3 and 0.7 <= y <= 1:
        return f"{prefix}2"
    elif 0.7 <= x <= 1 and 0.7 <= y <= 1:
        return f"{prefix}3"
    elif 0.7 <= x <= 1 and -0.3 <= y <= 0.3:
        return f"{prefix}4"
    elif 0.7 <= x <= 1 and -1 <= y <= -0.7:
        return f"{prefix}5"
    elif -0.3 <= x <= 0.3 and -1 <= y <= -0.7:
        return f"{prefix}6"
    elif -1 <= x <= -0.7 and -1 <= y <= -0.7:
        return f"{prefix}7"
    elif -1 <= x <= -0.7 and -0.3 <= y <= 0.3:
        return f"{prefix}8"
    else:
        return None  # undefined

# ----------------------------------------
# Zone to actuator control mapping
zone_to_ctrl_values = {
    "I0": (0.00,  0.78),
    "I1": (-0.34, -0.26),
    "I2": (0.00, -0.26),
    "I3": (0.34, -0.26),
    "I4": (0.34,  0.78),
    "I5": (0.34,  1.50),
    "I6": (0.00,  1.50),
    "I7": (-0.34, 1.50),
    "I8": (-0.34, 0.78),
    "J0": (0.60,  0.00),
    "J1": (1.20, -0.67),
    "J2": (0.60, -0.67),
    "J3": (0.00, -0.67),
    "J4": (0.00,  0.00),
    "J5": (0.00,  0.67),
    "J6": (0.60,  0.67),
    "J7": (1.20, 0.67),
    "J8": (1.20, 0.00),
    # I9 is only handled separately for rh_A_FFJ0
}

# ----------------------------------------
# Initialize pygame for joystick
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("No joystick connected.")
    sys.exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Joystick connected: {joystick.get_name()}")

# ----------------------------------------
# Load MuJoCo model and data
model = mujoco.MjModel.from_xml_path(
    "shadow_hand/scene_right.xml"
)
data = mujoco.MjData(model)

# Get actuator indices
ffj4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_A_FFJ4")
ffj3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_A_FFJ3")
ffj0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_A_FFJ0")
thj4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_A_THJ4")
thj2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_A_THJ2")
thj1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_A_THJ1")

# Last known stick zone (I0–I8)
last_zoneI = "I0"
last_zoneJ = "J0"

# ----------------------------------------
# Launch viewer and run control loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        pygame.event.pump()

        # Check if button 10 is pressed (I9 trigger)
        button_10 = joystick.get_button(10)
        button_11 = joystick.get_button(11)

        # Set rh_A_FFJ0 based on button 10
        if button_11:
            data.ctrl[ffj0_id] = 3.1
        else:
            data.ctrl[ffj0_id] = 0.0
        

        # Set rh_A_FFJ0 based on button 10
        if button_10:
            data.ctrl[thj1_id] = 1.56
        else:
            data.ctrl[thj1_id] = 0

        # Handle left stick for I0–I8 zones (only for FFJ3/FFJ4)
        Ix = joystick.get_axis(2)
        Iy = joystick.get_axis(3)
        Jx = joystick.get_axis(0)
        Jy = joystick.get_axis(1)

        zoneI = map_to_zone(Ix, Iy, "I")
        zoneJ = map_to_zone(Jx, Jy, "J")
        if zoneI in zone_to_ctrl_values:
            last_zoneI = zoneI
        if zoneJ in zone_to_ctrl_values:
            last_zoneJ = zoneJ

        ffj4_val, ffj3_val = zone_to_ctrl_values[last_zoneI]
        thj4_val, thj2_val = zone_to_ctrl_values[last_zoneJ]
        data.ctrl[ffj4_id] = ffj4_val
        data.ctrl[ffj3_id] = ffj3_val
        data.ctrl[thj4_id] = thj4_val
        data.ctrl[thj2_id] = thj2_val

        # Step simulation
        mujoco.mj_step(model, data)
        viewer.sync()
