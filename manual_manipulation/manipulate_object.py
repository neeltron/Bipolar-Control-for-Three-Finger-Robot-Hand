import mujoco
import mujoco.viewer
import glfw
import numpy as np
from scipy.spatial.transform import Rotation as R

# Load model and data
#XML_PATH = r"E:\Research\MujocoProjects\RLwithHand\shadow_hand\scene_right.xml"
XML_PATH = r"E:\Research\MujocoProjects\RLwithHand\wonik_allegro\scene_right.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Freejoint qpos index for the object
object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
object_qpos_adr = model.jnt_qposadr[object_joint_id]

# Get body ID of the stick
stick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick_rotated")

# Movement and rotation step sizes
step_size = 0.01  # meters
angle_step = np.deg2rad(5)  # 5 degrees in radians

# Gravity toggle flag
deactivate_gravity = False

# Rotation axes
X_AXIS = np.array([1, 0, 0])
Y_AXIS = np.array([0, 1, 0])
Z_AXIS = np.array([0, 0, 1])

# Rotation helper
def rotate_freejoint(axis, angle_rad):
    quat = data.qpos[object_qpos_adr + 3:object_qpos_adr + 7]  # [qw, qx, qy, qz]
    r_current = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [x, y, z, w]
    r_delta = R.from_rotvec(angle_rad * axis)
    r_new = r_delta * r_current
    quat_new = r_new.as_quat()  # [x, y, z, w]
    data.qpos[object_qpos_adr + 3:object_qpos_adr + 7] = [quat_new[3], quat_new[0], quat_new[1], quat_new[2]]

# Key handler
def key_callback(keycode):
    global deactivate_gravity

    if keycode == glfw.KEY_LEFT:
        data.qpos[object_qpos_adr + 0] -= step_size
    elif keycode == glfw.KEY_RIGHT:
        data.qpos[object_qpos_adr + 0] += step_size
    elif keycode == glfw.KEY_UP:
        data.qpos[object_qpos_adr + 1] += step_size
    elif keycode == glfw.KEY_DOWN:
        data.qpos[object_qpos_adr + 1] -= step_size
    elif keycode == glfw.KEY_PAGE_UP:
        data.qpos[object_qpos_adr + 2] += step_size
    elif keycode == glfw.KEY_PAGE_DOWN:
        data.qpos[object_qpos_adr + 2] -= step_size

    elif keycode == glfw.KEY_A:
        rotate_freejoint(X_AXIS, +angle_step)
    elif keycode == glfw.KEY_D:
        rotate_freejoint(X_AXIS, -angle_step)
    elif keycode == glfw.KEY_W:
        rotate_freejoint(Y_AXIS, +angle_step)
    elif keycode == glfw.KEY_S:
        rotate_freejoint(Y_AXIS, -angle_step)
    elif keycode == glfw.KEY_Q:
        rotate_freejoint(Z_AXIS, +angle_step)
    elif keycode == glfw.KEY_E:
        rotate_freejoint(Z_AXIS, -angle_step)

    elif keycode == glfw.KEY_N:
        deactivate_gravity = not deactivate_gravity
        if deactivate_gravity:
            print("[INFO] Stick physics DISABLED")
        else:
            print("[INFO] Stick physics ENABLED")

    print(f"[DEBUG] Pos: {data.qpos[object_qpos_adr:object_qpos_adr+3]}")
    print(f"[DEBUG] Quat: {data.qpos[object_qpos_adr+3:object_qpos_adr+7]}")



with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():

        if deactivate_gravity:
            # --- Freeze physics for stick by saving and restoring its state ---
            pos = np.copy(data.qpos[object_qpos_adr : object_qpos_adr + 7])  # 3 pos + 4 quat
            vel = np.copy(data.qvel[object_qpos_adr : object_qpos_adr + 6])  # 3 lin + 3 ang vel

            mujoco.mj_step(model, data)

            data.qpos[object_qpos_adr : object_qpos_adr + 7] = pos
            data.qvel[object_qpos_adr : object_qpos_adr + 6] = 0
        else:
            # --- Normal physics step ---
            mujoco.mj_step(model, data)

        viewer.sync()


