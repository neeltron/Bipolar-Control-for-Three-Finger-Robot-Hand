import mujoco
import mujoco.viewer
import numpy as np
import time
import glfw

# === Load model and data ===
XML_PATH = r"C:\CARL_Summer_Research\Bipolar-Control-for-Three-Finger-Robot-Hand-main (1)\Bipolar-Control-for-Three-Finger-Robot-Hand-main\Flexibletip_shadowhand\scene_right_m.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
global current_ctrl, should_update_ctrl, indexcrx, thumbcrx, middlecrx

# === Set stick pose === 
#lrge cube
qpos_stick = np.array([0.3366342, -0.02074563,  0.05997783, 0.99318724,  0.00353254,  0.06888206, -0.09392493])

#large cube X axis
#qpos_stick = np.array([0.36366342, -0.00474563,  0.05997783, 0.99759396, -0.00248437,  0.06892782, -0.00700555])
#qpos_stick = np.array([0.35366342, -0.02474563,  0.05997783, 0.99759396, -0.00248437,  0.06892782, -0.00700555])

#stapler
#qpos_stick = np.array([0.34366342, -0.02474563, 0.01997783, 0.99440839, -0.00848237,  0.06844901,  0.07996715])
# mouse pose
#qpos_stick = np.array([0.35, -0.02,  0.055,  0.99352163, -0.04745924, -0.0468759, 0.09200562]) s
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
qpos_addr = model.jnt_qposadr[joint_id]
data.qpos[qpos_addr : qpos_addr + 7] = qpos_stick


# === Set desired finger joint angles ===
actuator_names = [
    "rh_A_THJ5", "rh_A_THJ4", "rh_A_THJ1", 
    "rh_A_FFJ4", "rh_A_FFJ3", "rh_A_FFJ0",
    "rh_A_RFJ4", "rh_A_RFJ3", "rh_A_RFJ0", "rh_A_THJ2"
]

fixed_acuators = [
    "rh_A_LFJ4", "rh_A_LFJ3", "rh_A_LFJ0"
]

fixed_pos = np.array([0, 2, 1.57])
#fixed_pos = np.array([0, 0, 1])

#grasp_pose_hand = np.array([0.514, 1.33, -0.361, 0, 1.74, 0, 0, 1.74, 0]) # perfect angles
#grasp_pose_hand = np.array([0.314, 1.53, -0.361, 0, 1.5, 0, 0, 1.5, 0, 0]) 
grasp_pose_hand = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
current_ctrl = grasp_pose_hand.copy()

# === Map tripolar values to actuator control values ===
def map_tripolar_to_ctrl(tripolar_values):
    ctrl = np.zeros(len(actuator_names), dtype=np.float32)
    for i, v in enumerate(tripolar_values):
        lo, hi = model.actuator_ctrlrange[model.actuator(actuator_names[i]).id]
        mid = 0.5 * (lo + hi)
        if v == -1:
            ctrl[i] = lo
        elif v == 0:
            ctrl[i] = mid
        else:
            ctrl[i] = hi
    return ctrl


def apply_control(ctrl_values):
    for i, name in enumerate(actuator_names):
        actuator_id = model.actuator(name).id
        data.ctrl[actuator_id] = ctrl_values[i]

def step_for(n):
    for _ in range(n):
        #apply_control(current_ctrl)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

def read_current_pos():
    for i, actuator_name in enumerate(actuator_names):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        joint_id = model.actuator_trnid[actuator_id][0]  # Get joint ID
        qpos_adr = model.jnt_qposadr[joint_id]           # Get qpos address
        current_ctrl[i] = data.qpos[qpos_adr]            # Update from simulation
    return current_ctrl

# === Set initial joint angles AND actuator targets ===

# === Set initial joint angles AND actuator targets ===
for i, actuator in enumerate(actuator_names):
    joint_id = model.actuator(actuator).trnid[0]
    qpos_index = model.jnt_qposadr[joint_id]
    data.qpos[qpos_index] = grasp_pose_hand[i]           # for initial display
    #data.ctrl[model.actuator(actuator).id] = grasp_pose_hand[i]  # for actuator to maintain

# initial fixed actuator poses 
for i, actuator in enumerate(fixed_acuators):
    joint_id = model.actuator(actuator).trnid[0]
    qpos_index = model.jnt_qposadr[joint_id]
    #data.qpos[qpos_index] = fixed_pos[i]           # for initial display
    data.ctrl[model.actuator(actuator).id] = fixed_pos[i]  # for actuator to maintain




# === Commit changes ===
mujoco.mj_forward(model, data)

# Modify the key_callback to set the flag

should_update_ctrl = True
gravity_flag = False
del_step = 0.05
indexcrx = False
thumbcrx = False
middlecrx = False
f3 = False
f4 = False
f5 = False
f6 = False 
f7 = False
f8 = False
f9 = False
f10 = False
f11 = False
f12 = False
f13 = False


def key_callback(keycode):
    global current_ctrl, should_update_ctrl, gravity_flag, indexcrx, thumbcrx, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13
    if keycode == glfw.KEY_C:
        gravity_flag = True
        model.opt.gravity[:] = np.array([0, 0, -9.8])
        # skills 

    elif keycode == glfw.KEY_V:
        print("Caging action")
        indexcrx = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_B:
        print("Rotating action Y")
        thumbcrx = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_N:
        print("Stabilizing action")
        f3 = True
        should_update_ctrl = False
    
    elif keycode == glfw.KEY_M:
        print("Manipulation action")
        f4 = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_A:
        print("Further manipulation")
        f5 = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_L:
        print("Correction")
        f6 = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_R:
        print("Further correction")
        f7 = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_Y:
        print("Final thumb realignment")
        f8 = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_H:
        print("Rotating action X")
        f9 = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_O:
        print("Correcting via thumb and index")
        f10 = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_P:
        print("Correcting via ring")
        f11 = True
        should_update_ctrl = False


    elif keycode == glfw.KEY_Q:
        print("Correcting via ring")
        f12 = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_E:
        print("Correcting via INDEX")
        f13 = True
        should_update_ctrl = False


    elif keycode == glfw.KEY_Z:
        print("Correcting via thumb_side")
        p14 = True
        should_update_ctrl = False


    elif keycode == glfw.KEY_U:
        print("Correcting via index_side")
        p15 = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_K:
        print("Correcting via ring_side")
        p16 = True
        should_update_ctrl = False


    elif keycode == glfw.KEY_4:
        print("Rotating via X axis 2")
        p17 = True
        should_update_ctrl = False


# skills

# Build your primitive list P exactly as before (p1..p11 -> to_primitive(...))

def to_primitive(seq_rows, n):
    """
    seq_rows: list of np.array rows with values in {-1,0,+1}, length n each.
    Returns a (L, n) np.int8 array with validation.
    """
    arr = np.stack([np.asarray(r, dtype=np.int8) for r in seq_rows], axis=0)
    if arr.shape[1] != n:
        raise ValueError(f"Primitive row width {arr.shape[1]} != actuator count n={n}")
    # Validate ternary values
    if not np.isin(arr, (-1, 0, 1)).all():
        bad = arr[~np.isin(arr, (-1, 0, 1))]
        raise ValueError(f"Primitive contains non-ternary values: {np.unique(bad)}")
    return arr

n_act = 10
    # ring sweep in
p1 = [
    np.array([0,  -1,  -1,  -1, -1, 0,  0, -1, 0, 1]),
    np.array([0,  -1,  -1,  -1, -1, 0,  1, 0, -1, 1]),
    np.array([0,  -1,  -1,  -1, -1, 0,  1, 1, -1, 1]),
    np.array([0,  -1,  -1,  -1, -1, 0,  1, 1, 1, 1]),
    np.array([0,  -1,  -1,  -1, -1, 0,  -1, 1, 1, 1]),
    np.array([0,  -1,  -1,  -1, -1, 0,  -1, 1, -1, 1]),
    np.array([0,  -1,  -1,  -1, -1, 0,  -1, 0, -1, 1]),
    np.array([0,  -1,  -1,  -1, -1, 0,  -1, -1, 0, 1])
]
# inddex sweep  in
p2 = [
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, 0,  -1,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, 1,  -1,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, 1,  1,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, 1, 1,  1,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, 0, 1,  -1,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, 0,  -1,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1])

    ]
# index full push 
p3 = [
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, 0,  -1,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, 1,  -1,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, 0,  -1,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1])
]
# thumb-index srong pivot 
p4 = [
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  -1]),
    np.array([ -1, 0, 0, -1, -1,  0,  -1, -1,  0,  -1]),
    np.array([ 1, 1, 1, -1, -1,  0,  -1, -1,  0,  1]),
    np.array([ 1, 0, 1, -1, 0,  1,  -1, -1,  0,  1]),
    np.array([ 1, 0, 1, 1, 0,  1,  -1, -1,  0,  1]),
    np.array([ 0, 0, 1, 1, 0,  0,  -1, -1,  0,  1]),
    np.array([ -1, 0, 1, -1, -1,  0,  -1, -1,  0,  1]),


]
#thumb pull ## modified _____________
p5 = [
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),
    np.array([ -1, 0, -1, -1, -1,  0,  -1, -1,  0,  -1]),
    np.array([ 0, 1, -1, -1, -1,  0,  -1, -1,  0,  -1]),
    np.array([ 1, 1, 1, -1, -1,  0,  -1, -1,  0,  0]),
    np.array([ 1, 0, 0, -1, 0,  -1,  -1, -1,  0,  1]),
    np.array([ 1, -1, 0, -1, 1,  -1,  -1, -1,  0,  0]),
    np.array([ 0, 0, -1, -1, 0,  0,  -1, -1,  0,  -1]),
    np.array([ -1, 0, -1, -1, -1,  0,  -1, -1,  0,  0]),
    np.array([ 0, -1, 1, -1, -1,  0,  -1, -1,  0,  1])

]

# ring push and rotate ### modified: thumb push added 
p6 = [
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),
    np.array([ 1, 0, -1, -1, -1,  0,  1, 0,  1,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, 0,  1,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),


]
# ring full push 
p7 = [
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  1, 0,  -1,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  1, 1,  -1,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, 1,  -1,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, 0,  -1,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),

]

# thumb push 

p8 = [
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),
    np.array([ 1, 0, -1, -1, -1,  0,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, 0,  0,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),
]
# index push and rotate 

p9 = [
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, 1, 0,  1,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, 0,  1,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),

]


p10 = [
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, 0, 0,  -1,  0, 0,  -1,  1]),
    np.array([ 0, -1, -1, 0, 1,  -1,  0, 1,  -1,  1]),
    np.array([ 0, -1, -1, -1, 1,  -1,  -1, 1,  -1,  1]),
    np.array([ 0, -1, -1, -1, 0,  -1,  -1, 0,  -1,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1])

]


p11 = [
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),
    np.array([ 0, -1, -1, 0, 0,  -1,  0, 0,  -1,  1]),
    np.array([ 0, -1, -1, 0, 1,  -1,  0, 1,  -1,  1]),
    np.array([ 0, -1, -1, 0, 1,  1,  0, 1,  1,  1]),
    np.array([ 0, -1, -1, 0, 0,  1, 0, 0,  1,  1]),
    np.array([ 0, -1, -1, 0, -1,  1,  0, -1,  1,  1]),
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0,  1]),


]


# pn = [
#     np.array([ 0, -1, -1, 0, -1, -1, 0, -1, -1, 0]),          ### only caging action, CAGING PRIMITIVE
#     np.array([ 0, -1, -1, 0, -1, -1, 0, 0, 0, 0]),
#     np.array([ 0, -1, -1, 0, -1, -1, 0, 0, 0, 0]),
#     np.array([ 0, -1, -1, -1, -1, 0, 0, 0, 0, 0]),
#     np.array([ 0, -1, -1, -1, -1, 0, 0, 0, 0, 0]),
#     np.array([ 0, -1, 0, -1, -1, 0, 0, 0, 0, 0]),
# ]

p12 = [
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0, 1]), # fixed
    np.array([ 0, -1, -1, -1, -1, 0, 0, -1, 0, 1]),       ### rotating action with caging motion, CAGING + ROTATION PRIMITIVE 
    np.array([ 0, -1, -1, -1, -1, 0, 0, 0, 0, 1]),
    np.array([ 0, -1, -1, -1, -1, 0, 0, 0, 1, 1]),
    np.array([ 0, -1, -1, -1, -1, 0, 0, 0, 1, 1]),
    np.array([ 1, 0, 0, -1, -1, 0, 0, 0, 1, 1]),
    np.array([ 0, 1, 0, -1, 0, 0, 0, 0, 0, 1]),
    np.array([ -1, 1, -1, -1, 1, 0, 0, 0, 0, 1]),
    np.array([ -1, 0, -1, -1, 0, 0, -1, -1, 0, 1]),
    np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 0, 1])
]

# p14 = [
#     np.array([ 0, 0, 0, -1, -1, 0, 0, -1, 0, 1]),       ####releasing object onto the palm - RELEASE PRIMITIVE
#     np.array([ 0, 0, 0, -1, -1, -1, 0, -1, 0, 1]),
#     np.array([ 0, 0, -1, -1, -1, -1, 0, -1, 0, 1]),
#     np.array([ 0, 0, -1, -1, -1, -1, 0, -1, 0, 1]),
#     np.array([ 0, 0, -1, -1, -1, -1, 0, -1, 1, 1]),
# ]

# p15 = [
#     np.array([ 0, 0, -1, -1, -1, -1, 0, -1, 1, 1]),      ####Ring finger slightly reorients the object while other two cages - REORIENTATION PRIMITIVE
#     np.array([ 0, 0, -1, -1, -1, 0, 0, -1, 1, 1]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, -1, 1, 1]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 1, 1]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 1, 1]),####
#     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 1, 1]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 1]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 1]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 0]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 0]),
# ]

# p5 = [
#     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 1, 1]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 1]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 1]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 0]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 0]),
# ]

# p16 = [
#     np.array([ 0, 0, -1, -1, -1, 0, 0, -1, 0, 0]),       ####Thumb tries to pull back object from one corner - CORRECTIVE PRIMITIVE
#     np.array([ 0, 1, 0, -1, -1, 0, 0, -1, 0, 1]),
#     np.array([ 0, 0, 0, -1, -1, 0, 0, -1, 0, 1]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, -1, 0, -1]),
#     np.array([ -1, 0, 0, -1, -1, 0, 0, -1, 0, 0]),
#     np.array([ -1, -1, 0, -1, -1, 0, 0, -1, 0, 0]),
# ]

# p7 = [
#     np.array([ 0, 0, 0, -1, -1, 0, 0, -1, 0, 1]),
#     np.array([ 0, 0, -1, -1, -1, 0, 0, -1, 0, 0]),
#     np.array([ 0, -1, -1, -1, -1, 0, 0, -1, 0, 0]),
#     np.array([ 0, -1, -1, -1, -1, 0, 0, -1, 0, 1]),
# ]

# p17 = [
#     np.array([ 0, -1, -1, -1, -1, 0, 0, 0, 0, 1]),         ####When object is at the corner, thumb and index pushes it back in  - CORRECTIVE PRIMITIVE
#     np.array([ 0, -1, 0, -1, -1, 1, 0, 0, 0, 0]),
#     np.array([ 0, -1, 0, -1, -1, 0, 0, 0, 0, 0]),
#     np.array([ 0, -1, 0, 0, -1, 0, 0, 0, 0, 0]),
#     np.array([ 0, -1, 0, 0, -1, 1, 0, 0, 0, 0]),
#     np.array([ 0, -1, 0, -1, -1, 0, 0, 0, 0, 0]),
# ]

p13 = [
    np.array([ 0, -1, -1, -1, -1,  0,  -1, -1,  0, 1]),   ## modified     #####When object is tilted on the palm, might perform Y axis rotation - ROTATION PRIMITIVE
    np.array([0, -1, -1, 0, -1, 1, 0, -1, 1, 1]),
    np.array([0, -1, 0, 0, 0, 1, 0, -1, 1, 1]),
    np.array([0, -1, 0, 0, 0, 1, 0, 0, 1, 1]),
    np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 1]),
    np.array([0, -1, 0, 0, 1, -1, 0, 0, -1, 1]),
    np.array([0, -1, 0, 0, 0, -1, 0, -1, 0, 1]),
    np.array([0, -1, 0, -1, -1, 0, -1, -1, 0, 1]),
]

# p19 = [
#     np.array([-1, -1, 0, 0, 1, -1, 0, 0, 0, 0]),       ####When object is at the far end of a corner, thumb pulls it back in  - CORRECTIVE PRIMITIVE
#     np.array([-1, 0, 0, 0, 1, -1, 0, -1, 0, 0]),
#     np.array([-1, 1, 0, 0, 0, -1, 0, -1, 0, 0]),
#     np.array([0, 1, 0, 0, -1, -1, 0, -1, 0, 0]),
#     np.array([1, 1, -1, -1, -1, 0, 0, -1, 0, 0]),
#     np.array([0, 1, -1, -1, -1, 0, 0, -1, 0, 0]),
#     np.array([-1, 1, -1, -1, -1, 0, 0, -1, 0, 0]),
#     np.array([-1, 0, -1, -1, 0, 0, 0, -1, 0, 1]),
#     np.array([-1, -1,-1, -1, 0, 0, 0, -1, 0, 1]),
#     np.array([0, -1,-1, -1, -1, 0, 0, -1, 0, 1]),
# ]

# p20 = [
#     np.array([ 0, -1, -1, -1, 0, 0, 0, -1, 0, 1]),       ####Reorients the object from the ring side - REORIENTATION PRIMITIVE
#     np.array([ 0, -1, -1, -1, 0, 0, 0, -1, 1, 1]),
#     np.array([ 0, -1, -1, -1, 0, 0, 0, 0, 1, 1]),
#     np.array([ 0, -1, -1, -1, 0, 0, 0, 0, 1, 1]),
#     #np.array([ 0, -1, -1, -1, 0, 0, 0, 0, 1, 1]),
#     #np.array([ 0, -1, -1, -1, 0, 0, 0, -1, 1, 1]),
# ]

# p21 = [
#     np.array([-1, -1, 0, 0, -1, -1, 0, -1, 0, 0]),        ####When object is at the far back end of the palm, thumb push the object back in from behind - CORRECTIVE PRIMITIVE
#     np.array([-1, 0, 0, -1, -1, -1, 0, -1, 0, -1]),
#     np.array([-1, 1, -1, -1, -1, -1, 0, -1, 0, -1]),
#     np.array([0, 1, -1, -1, -1, 0, 0, -1, 0, -1]),
#     np.array([0, 1, -1, -1, -1, 0, 0, -1, 0, 1]),
#     np.array([-1, 1, -1, -1, -1, 0, 0, -1, 0, 1]),
#     np.array([-1, 0, -1, -1, -1, 0, 0, -1, 0, 1]),
#     np.array([-1, -1, -1, -1, -1, 0, 0, -1, 0, 1]),
#     np.array([0, -1, -1, -1, -1, 0, 0, -1, 0, 0]),
# ]

# p13 = [
#     np.array([-1, 0, 0, 0, 1, -1, 0, 0, 0, 0]),
#     np.array([-1, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
#     np.array([-1, 0, 0, 0, 1, -1, 0, 0, 0, 0]),
# ]

# p22 = [
#     np.array([0, -1, -1, 0, -1, 0, 0, -1, 0, 0]),        ####When object is laterally disbalanced and about to fall off from the edge of the palm, thumb pushes it back in - CORRECTIVE PRIMITIVE
#     np.array([0, -1, -1, -1, -1, 0, -1, -1, 0, 1]),
#     np.array([0, -1, -1, -1, -1, 0, -1, -1, 0, 0]),
#     np.array([0, -1,-1, -1, -1, 0, -1, -1, 0, 1]),
#     np.array([0, -1, 0, -1, -1, 0, -1, -1, 0, 1]),
# ]

# p23 = [
#     np.array([ 0, -1, -1, 0, -1, -1, -1, -1, 0, 0]),      ####When object at the corner of index finger, index finger pushes the object in - CORRECTIVE PRIMITIVE
#     np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 0, 0]),
#     np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 0, 0]),
#     np.array([ 0, -1, -1, -1, -1, 1, -1, -1, 0, 0]),
#     np.array([ 0, -1, -1, -1, -1, 1, -1, -1, 0, 0]),
# ]

# p24 = [
#     np.array([ 0, -1, -1, -1, -1, 0, 0, -1, -1, 0]),      ####When object at the corner of ring finger, ring finger pushes the object in - CORRECTIVE PRIMITIVE
#     np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 0, 0]),
#     np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 0, 0]),
#     np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 1, 0]),
#     np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 1, 0]),
# ]

# p25 = [
#     np.array([ 0, -1, -1, 0, -1, -1, 0, -1, -1, 0]),     ###rotates object front to back Y axis (according to mujoco) - ROTATORY PRIMITIVE
#     np.array([ 0, -1, -1, 0, 0, -1, 0, 0, -1, 0]),
#     np.array([ 0, -1, -1, 0, 0, 0, 0, 0, 0, 0]),
#     np.array([ 0, -1, -1, 0, 0, 0, 0, 0, 0, 0]),
#     np.array([ 0, -1, -1, 0, 0, 0, 0, 0, 0, -1]),
#     np.array([ 0, 0, -1, 0, 0, 0, 0, 0, 0, -1]),
#     np.array([ 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]),
# ]

#Convert to validated (L, n) arrays
P = [
    to_primitive(p1, n_act),
    to_primitive(p2, n_act),
    to_primitive(p3, n_act),
    to_primitive(p4, n_act),
    to_primitive(p5, n_act),
    to_primitive(p6, n_act),
    to_primitive(p7, n_act),
    to_primitive(p8, n_act),
    to_primitive(p9, n_act),
    to_primitive(p10, n_act),
    to_primitive(p11, n_act),
    to_primitive(p12, n_act),
    to_primitive(p13, n_act)
]




# === Enable gravity ===
model.opt.gravity[:] = np.array([0, 0, 0])


def action_primitive_1_caging():
    for row in p1:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(50)

def action_primitive_2_rotateY():
    for row in p2:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(50)

def action_primitive_3_stabilize():
    for row in p3:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(50)

def action_primitive_4_manipulate():
    for row in p4:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(50)

def action_primitive_5_manipulate():
    for row in p5:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(50)

def action_primitive_6_correct():
    for row in p6:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(50)

def action_primitive_7_correct():
    for row in p7:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(50)

def action_primitive_8_correct():
    for row in p8:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(50)

def action_primitive_9_rotateX():
    for row in p9:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(50)

def action_primitive_10_correct():
    for row in p10:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(50)

def action_primitive_11_correct():
    for row in p11:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(50)

def action_primitive_12_correct():
    for row in p12:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(100)

def action_primitive_13_correct():
    for row in p13:
        ctrl = map_tripolar_to_ctrl(row)
        apply_control(ctrl)
        step_for(100)



# === Run viewer with physics enabled ===
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    #print("Press 'A' for initial position, 'B' for target position.")
    
    while viewer.is_running():
        # if not any([indexcrx, thumbcrx, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]):
        #     mujoco.mj_step(model, data)
        #     viewer.sync()
        #     continue
        if should_update_ctrl:
            apply_control(current_ctrl)
            should_update_ctrl = False  # only set once per key press
        else:
            if indexcrx:
                action_primitive_1_caging()
                should_update_ctrl = False
                indexcrx = False
            elif thumbcrx:
                action_primitive_2_rotateY()
                should_update_ctrl = False
                thumbcrx = False
            elif f3:
                action_primitive_3_stabilize()
                should_update_ctrl = False
                f3 = False

            elif f4:
                action_primitive_4_manipulate()
                should_update_ctrl = False
                f4 = False

            elif f5:
                action_primitive_5_manipulate()
                should_update_ctrl = False
                f5 = False

            elif f6:
                action_primitive_6_correct()
                should_update_ctrl = False
                f6 = False
    
            elif f7:
                action_primitive_7_correct()
                should_update_ctrl = False
                f7 = False

            elif f8:
                action_primitive_8_correct()
                should_update_ctrl = False
                f8 = False

            elif f9:
                action_primitive_9_rotateX()
                should_update_ctrl = False
                f9 = False

            elif f10:
                action_primitive_10_correct()
                should_update_ctrl = False
                f10 = False


            elif f11:
                action_primitive_11_correct()
                should_update_ctrl = False
                f11 = False


            elif f12:
                action_primitive_12_correct()
                should_update_ctrl = False
                f12 = False


            elif f13:
                action_primitive_13_correct()
                should_update_ctrl = False
                f13 = False




        mujoco.mj_step(model, data)
        viewer.sync()
        #time.sleep(0.01)
