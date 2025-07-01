import mujoco
import mujoco.viewer
import time

# Path to your Shadow Hand XML model
XML_PATH = r"E:\Research\MujocoProjects\RLwithHand\shadow_hand\scene_right.xml"
#XML_PATH = r"E:\Research\MujocoProjects\RLwithHand\wonik_allegro\scene_right.xml"
#XML_PATH = r"E:\Research\MujocoProjects\RLwithHand\wonik_allegro\tri_finger_hand.xml"

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path(XML_PATH)

# Create simulation data
data = mujoco.MjData(model)

# Launch interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer launched. Press ESC to quit.")
    # Keep running until viewer is closed
    while viewer.is_running():
        #data.ctrl[model.actuator(name='rh_A_FFJ3').id] = 2.0
        #print(model.actuator_ctrlrange[model.actuator('rh_A_FFJ3').id])
        step_start = time.time()

        # Advance the simulation by one step (for visualization)
        mujoco.mj_step(model, data)

        # Sync with real time
        viewer.sync()

        # (Optional) Slow down simulation if running too fast
        time.sleep(max(0, 0.01 - (time.time() - step_start)))
