import mujoco
import mujoco.viewer
import numpy as np
import time
import glfw

# === Load model and data ===
XML_PATH = r"E:\Research\MujocoProjects\RLwithHand\shadow_hand\scene_right_m.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
global current_ctrl, should_update_ctrl, indexcrx, thumbcrx, middlecrx, t_i_pull

# === Set stick pose ===
trnl_qpos_stick = np.array([0.30226065, 0.03, 0.03, 0.26610262, -0.65136273,  0.27518253,  0.65512638])
rot_qpos_stick = np.array([3.22260647e-01, -0.02,  0.095, 0.49655277, -0.4964729, 0.50350123, 0.50342479])
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
qpos_addr = model.jnt_qposadr[joint_id]
data.qpos[qpos_addr : qpos_addr + 7] = rot_qpos_stick


# === Set desired finger joint angles ===
actuator_names = [
    "rh_A_THJ5", "rh_A_THJ4", "rh_A_THJ1",
    "rh_A_FFJ4", "rh_A_FFJ3", "rh_A_FFJ0",
    "rh_A_MFJ4", "rh_A_MFJ3", "rh_A_MFJ0", "rh_A_THJ2"
]
#grasp_pose_hand = np.array([0.514, 1.33, -0.361, 0, 1.74, 0, 0, 1.74, 0]) # parfect angles
nrtl_pose = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 
grasp_pose_hand = np.array([0.314, 1.53, -0.361, 0, 1.5, 0, 0, 1.5, 0, 0]) 
current_ctrl = nrtl_pose


def apply_control(ctrl_values):
    for i, name in enumerate(actuator_names):
        actuator_id = model.actuator(name).id
        data.ctrl[actuator_id] = ctrl_values[i]

def read_current_pos():
    for i, actuator_name in enumerate(actuator_names):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        joint_id = model.actuator_trnid[actuator_id][0]  # Get joint ID
        qpos_adr = model.jnt_qposadr[joint_id]           # Get qpos address
        current_ctrl[i] = data.qpos[qpos_adr]            # Update from simulation

# === Set initial joint angles AND actuator targets ===
for i, actuator in enumerate(actuator_names):
    joint_id = model.actuator(actuator).trnid[0]
    qpos_index = model.jnt_qposadr[joint_id]
    data.qpos[qpos_index] = current_ctrl[i]           # for initial display
    data.ctrl[model.actuator(actuator).id] = current_ctrl[i]  # for actuator to maintain

# === Commit changes ===
mujoco.mj_forward(model, data)

# Modify the key_callback to set the flag

should_update_ctrl = True
gravity_flag = False
del_step = 0.05
indexcrx = False
thumbcrx = False
middlecrx = False
t_i_pull = False


def key_callback(keycode):
    global current_ctrl, should_update_ctrl, indexcrx, thumbcrx, middlecrx, t_i_pull
    if keycode == glfw.KEY_U:
        print("T-I-M-grasp")
        read_current_pos()
        current_ctrl[0] += del_step  # rh_A_THJ5
        current_ctrl[1] -= del_step  # rh_A_THJ4
        current_ctrl[4] += del_step  # rh_A_FFJ3
        current_ctrl[7] += del_step  # rh_A_MFJ3
        should_update_ctrl = True
    elif keycode == glfw.KEY_H:
        print("T-I-M-release")
        read_current_pos()
        current_ctrl[0] -= del_step  # rh_A_THJ5
        current_ctrl[1] += del_step  # rh_A_THJ4
        current_ctrl[4] -= del_step  # rh_A_FFJ3
        current_ctrl[7] -= del_step  # rh_A_MFJ3
        should_update_ctrl = True
    elif keycode == glfw.KEY_O:
        print("T-I grasp")
        read_current_pos()
        current_ctrl[0] += del_step  # rh_A_THJ5
        current_ctrl[1] -= 0.6*del_step  # rh_A_THJ4
        #current_ctrl[4] += del_step  # rh_A_FFJ3
        should_update_ctrl = True
    elif keycode == glfw.KEY_P:
        print("T-M grasp")
        read_current_pos()
        current_ctrl[0] += 0.6*del_step  # rh_A_THJ5
        current_ctrl[1] -= del_step  # rh_A_THJ4
        #current_ctrl[7] += del_step  # rh_A_FFJ3
        should_update_ctrl = True
    elif keycode == glfw.KEY_Q:
        print("T flex")
        read_current_pos()
        current_ctrl[0] += del_step  # rh_A_THJ5
        current_ctrl[1] -= del_step  # rh_A_THJ4
        should_update_ctrl = True
    elif keycode == glfw.KEY_E:
        print("T exten")
        read_current_pos()
        current_ctrl[0] -= del_step  # rh_A_THJ5
        current_ctrl[1] += del_step  # rh_A_THJ4
        should_update_ctrl = True

    # set inital pose
    elif keycode == glfw.KEY_Z:
        print("Initial Pose")
        current_ctrl = grasp_pose_hand
        # current_ctrl[0] = 0.0105   # rh_A_THJ4
        # current_ctrl[1] = 0.19  # rh_A_THJ3
        # current_ctrl[2] = 1.33 
        # current_ctrl[9] = 0.698
        
        # current_ctrl[3] = 0.34   # rh_A_FFJ4
        # current_ctrl[4] = 0.62   # rh_A_FFJ3
        # current_ctrl[5] = 3.1    # rh_A_FFJ0
        # current_ctrl[6] = 0.34   # rh_A_FFJ0
        # current_ctrl[7] = 1.79   # rh_A_FFJ0
        # current_ctrl[8] = 3.1
        should_update_ctrl = True

    elif keycode == glfw.KEY_C:
        gravity_flag = True
        model.opt.gravity[:] = np.array([0, 0, -9.8])
        # skills 

    elif keycode == glfw.KEY_V:
        print("Index Crossover")
        indexcrx = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_B:
        print("Thumb Crossover")
        thumbcrx = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_N:
        print("Middle F Crossover")
        middlecrx = True
        should_update_ctrl = False
    elif keycode == glfw.KEY_M:
        print("T-I F pull")
        t_i_pull = True
        should_update_ctrl = False


# skills

def indexCrossOver():
    print("nested loop")
    # === Step 1: Pose A ===
    current_ctrl[3] = 0.0    # rh_A_FFJ4
    current_ctrl[4] = 0.3    # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    apply_control(current_ctrl)
    for _ in range(100):
        apply_control(current_ctrl)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    # === Step 2: Pose B ===
    current_ctrl[3] = 0.36   # rh_A_FFJ4
    current_ctrl[4] = 0.3   # rh_A_FFJ3
    current_ctrl[5] = 3.1    # rh_A_FFJ0
    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    #step 3
    current_ctrl[3] = 0.36   # rh_A_FFJ4
    current_ctrl[4] = 2   # rh_A_FFJ3
    current_ctrl[5] = 3.1    # rh_A_FFJ0
    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
    
    # step 4
    current_ctrl[3] = 0.36   # rh_A_FFJ4
    current_ctrl[4] = 2   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
    #step 5
    current_ctrl[3] = 0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.74   # rh_A_FFJ3 ><> meeting MFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
    #step 6
    current_ctrl[0] = -0.9   # rh_A_THJ4
    current_ctrl[1] = 1.47   # rh_A_THJ3
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.74   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = 0.36   # rh_A_FFJ0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
    #step 7
    current_ctrl[0] = -0.9   # rh_A_THJ4
    current_ctrl[1] = 1.47   # rh_A_THJ3
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.6   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = 0.36   # rh_A_FFJ0
    current_ctrl[7] = 1.7   # rh_A_FFJ0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
    


def thumbCrossOver():
    #step 1 t crx
    current_ctrl[0] = -0.9   # rh_A_THJ4
    current_ctrl[1] = 1.47   # rh_A_THJ3
    current_ctrl[2] = 1.57 
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.6   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = 0.36   # rh_A_FFJ0
    current_ctrl[7] = 1.7   # rh_A_FFJ0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    #step 2 t crx
    current_ctrl[0] = -1.05   # rh_A_THJ4
    current_ctrl[1] = 0.4   # rh_A_THJ3
    current_ctrl[2] = 1.57 
    current_ctrl[9] = 0.698

    current_ctrl[3] = 0   # rh_A_FFJ4
    current_ctrl[4] = 1.4   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = 0   # rh_A_FFJ0
    current_ctrl[7] = 1.4   # rh_A_FFJ0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    #step 3 t crx
    current_ctrl[0] = 1.05   # rh_A_THJ4
    current_ctrl[1] = 1.84   # rh_A_THJ3
    current_ctrl[2] = 0.5 
    current_ctrl[9] = 0.4
    
    current_ctrl[3] = +0.27   # rh_A_FFJ4
    current_ctrl[4] = 1.4   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = -0.27   # rh_A_FFJ0
    current_ctrl[7] = 1.4   # rh_A_FFJ0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    #step 3_post t crx
    current_ctrl[0] = 1.05   # rh_A_THJ4
    current_ctrl[1] = 1.84   # rh_A_THJ3
    current_ctrl[2] = 0.5 
    current_ctrl[9] = 0.4
    
    current_ctrl[3] = -0.2  # rh_A_FFJ4
    current_ctrl[4] = 1.6   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = +0.2   # rh_A_FFJ0
    current_ctrl[7] = 1.6   # rh_A_FFJ0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    #step 4 t crx
    current_ctrl[0] = 1.05   # rh_A_THJ4
    current_ctrl[1] = 1.84   # rh_A_THJ3
    current_ctrl[2] = 0.5 
    current_ctrl[9] = 0.4
    
    current_ctrl[3] = -0.3   # rh_A_FFJ4
    current_ctrl[4] = 1.71   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = +0.3   # rh_A_FFJ0
    current_ctrl[7] = 1.71   # rh_A_FFJ0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)


    #step 5 t crx
    current_ctrl[0] = 1.05   # rh_A_THJ4
    current_ctrl[1] = 1.65   # rh_A_THJ3
    current_ctrl[2] = -0.362 
    current_ctrl[9] = 0
    
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.71   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = 0.36   # rh_A_FFJ0
    current_ctrl[7] = 1.71   # rh_A_FFJ0


    #step 5 t crx
    current_ctrl[0] = 1.05   # rh_A_THJ4
    current_ctrl[1] = 1.58   # rh_A_THJ3
    current_ctrl[2] = -0.362 
    current_ctrl[9] = 0
    
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.71   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = 0.36   # rh_A_FFJ0
    current_ctrl[7] = 1.71   # rh_A_FFJ0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
    #step 6 t crx
    current_ctrl[0] = 0.806   # rh_A_THJ4
    current_ctrl[1] = 1.58   # rh_A_THJ3
    current_ctrl[2] = -0.362 
    current_ctrl[9] = 0
    
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.71   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = 0.36   # rh_A_FFJ0
    current_ctrl[7] = 1.71   # rh_A_FFJ0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    #step 7 t crx
    current_ctrl[0] = 0.597   # rh_A_THJ4
    current_ctrl[1] = 1.58   # rh_A_THJ3
    current_ctrl[2] = -0.362 
    current_ctrl[9] = 0
    
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.71   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = 0.36   # rh_A_FFJ0
    current_ctrl[7] = 1.71   # rh_A_FFJ0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    #step 8 t crx
    current_ctrl[0] = 0.597   # rh_A_THJ4
    current_ctrl[1] = 1.47   # rh_A_THJ3
    current_ctrl[2] = -0.362 
    current_ctrl[9] = 0
    
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.71   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = 0   # rh_A_FFJ0
    current_ctrl[7] = 1.71   # rh_A_FFJ0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    #step 9 t crx
    current_ctrl[0] = 0.550   # rh_A_THJ4
    current_ctrl[1] = 1.31  # rh_A_THJ3
    current_ctrl[2] = -0.362 
    current_ctrl[9] = 0
    
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.71   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = -0.3   # rh_A_FFJ0
    current_ctrl[7] = 1.71   # rh_A_FFJ0
    current_ctrl[8] = 1.71

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    #step 10 t crx
    current_ctrl[0] = 0.590   # rh_A_THJ4
    current_ctrl[1] = 1.27  # rh_A_THJ3
    current_ctrl[2] = -0.362 
    current_ctrl[9] = 0
    
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.71   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = -0.3   # rh_A_FFJ0
    current_ctrl[7] = 0   # rh_A_FFJ0
    current_ctrl[8] = 3.14

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

def middleCrossOver():
        #step 1 m crx
    current_ctrl[0] = 0.590   # rh_A_THJ4
    current_ctrl[1] = 1.27  # rh_A_THJ3
    current_ctrl[2] = -0.362 
    current_ctrl[9] = 0
    
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.71   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = 0.3   # rh_A_FFJ0
    current_ctrl[7] = 0.6   # rh_A_FFJ0
    current_ctrl[8] = 0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

        #step 2 m crx
    current_ctrl[0] = 0.590   # rh_A_THJ4
    current_ctrl[1] = 1.27  # rh_A_THJ3
    current_ctrl[2] = -0.362 
    current_ctrl[9] = 0
    
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.71   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = 0.3   # rh_A_FFJ0
    current_ctrl[7] = 1.4   # rh_A_FFJ0
    current_ctrl[8] = 0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

        #step 2 m crx
    current_ctrl[0] = 0.53   # rh_A_THJ4
    current_ctrl[1] = 1.3  # rh_A_THJ3
    current_ctrl[2] = -0.362 
    current_ctrl[9] = 0
    
    current_ctrl[3] = -0.36   # rh_A_FFJ4
    current_ctrl[4] = 1.71   # rh_A_FFJ3
    current_ctrl[5] = 0    # rh_A_FFJ0
    current_ctrl[6] = 0.3   # rh_A_FFJ0
    current_ctrl[7] = 1.71   # rh_A_FFJ0
    current_ctrl[8] = 0

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)




# ==== Thumb-Index pull skill for translation 

def t_i_pullTrnsl():

    # step 01
    current_ctrl[0] = 0.105   # rh_A_THJ4
    current_ctrl[1] = 0.48  # rh_A_THJ3
    current_ctrl[2] = 1.57 
    current_ctrl[9] = -0.69
    
    current_ctrl[3] = 0.34   # rh_A_FFJ4
    current_ctrl[4] = 0.72   # rh_A_FFJ3
    current_ctrl[5] = 2.7    # rh_A_FFJ0
    current_ctrl[6] = 0.34   # rh_A_FFJ0
    current_ctrl[7] = 1.79   # rh_A_FFJ0
    current_ctrl[8] = 3.1

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        print("here")
        time.sleep(0.01)

    # step 02
    current_ctrl[0] = 0.0105   # rh_A_THJ4
    current_ctrl[1] = 0  # rh_A_THJ3
    current_ctrl[2] = 1.57 
    current_ctrl[9] = -0.69
    
    current_ctrl[3] = 0.34   # rh_A_FFJ4
    current_ctrl[4] = 0.72   # rh_A_FFJ3
    current_ctrl[5] = 2.7    # rh_A_FFJ0
    current_ctrl[6] = 0.34   # rh_A_FFJ0
    current_ctrl[7] = 1.79   # rh_A_FFJ0
    current_ctrl[8] = 3.1

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        print("here")
        time.sleep(0.01)

    # step 03
    current_ctrl[0] = 0.0105   # rh_A_THJ4
    current_ctrl[1] = 0  # rh_A_THJ3
    current_ctrl[2] = 1.57 
    current_ctrl[9] = 0
    
    current_ctrl[3] = 0.34   # rh_A_FFJ4
    current_ctrl[4] = 0.72   # rh_A_FFJ3
    current_ctrl[5] = 2.7     # rh_A_FFJ0
    current_ctrl[6] = 0.34   # rh_A_FFJ0
    current_ctrl[7] = 1.79   # rh_A_FFJ0
    current_ctrl[8] = 2.9

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        print("here")
        time.sleep(0.01)

    # step 03
    current_ctrl[0] = 0.0105   # rh_A_THJ4
    current_ctrl[1] = 0.1  # rh_A_THJ3
    current_ctrl[2] = 1.57 
    current_ctrl[9] = 0.5
    
    current_ctrl[3] = 0.34   # rh_A_FFJ4
    current_ctrl[4] = 0.72   # rh_A_FFJ3
    current_ctrl[5] = 2.7     # rh_A_FFJ0
    current_ctrl[6] = 0.34   # rh_A_FFJ0
    current_ctrl[7] = 1.79   # rh_A_FFJ0
    current_ctrl[8] = 2.9

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        print("here")
        time.sleep(0.01)


    # step 1
    current_ctrl[0] = -0.0105   # rh_A_THJ4
    current_ctrl[1] = 0.2  # rh_A_THJ3 0.19
    current_ctrl[2] = 1.57 
    current_ctrl[9] = 1
    
    current_ctrl[3] = 0.34   # rh_A_FFJ4
    current_ctrl[4] = 0.72   # rh_A_FFJ3
    current_ctrl[5] = 2.7     # rh_A_FFJ0
    current_ctrl[6] = 0.34   # rh_A_FFJ0
    current_ctrl[7] = 1.79   # rh_A_FFJ0
    current_ctrl[8] = 2.7

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        print("here")
        time.sleep(0.01)

    # step 2
    current_ctrl[0] = -0.105   # rh_A_THJ4
    current_ctrl[1] = 0.2  # rh_A_THJ3
    current_ctrl[2] = -0.3 
    current_ctrl[9] = 1
    
    current_ctrl[3] = -0.34   # rh_A_FFJ4
    current_ctrl[4] = 0.72   # rh_A_FFJ3
    current_ctrl[5] = 2.7     # rh_A_FFJ0
    current_ctrl[6] = 0.34   # rh_A_FFJ0
    current_ctrl[7] = 1.79   # rh_A_FFJ0
    current_ctrl[8] = 2.7

    apply_control(current_ctrl)
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
        print("here")
        time.sleep(0.01)

def t_i_pullTrnsl1():
        # step 1

        current_ctrl[0] = 0.0105   # rh_A_THJ4
        current_ctrl[1] = 0.19  # rh_A_THJ3
        current_ctrl[2] = 1.33 
        current_ctrl[9] = 0.698
        
        current_ctrl[3] = 0.34   # rh_A_FFJ4
        current_ctrl[4] = 0.62   # rh_A_FFJ3
        current_ctrl[5] = 3.1    # rh_A_FFJ0
        current_ctrl[6] = 0.34   # rh_A_FFJ0
        current_ctrl[7] = 1.79   # rh_A_FFJ0
        current_ctrl[8] = 3.1
        apply_control(current_ctrl)
        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()
            #print("here")
            time.sleep(0.01)
        # step 2

        current_ctrl[0] = -0.205   # rh_A_THJ4
        current_ctrl[1] = 0.19  # rh_A_THJ3
        current_ctrl[2] = 0 
        current_ctrl[9] = 0.5
        
        current_ctrl[3] = -0.34   # rh_A_FFJ4
        current_ctrl[4] = 0.62   # rh_A_FFJ3
        current_ctrl[5] = 2.7    # rh_A_FFJ0
        current_ctrl[6] = 0.34   # rh_A_FFJ0
        current_ctrl[7] = 1.5   # rh_A_FFJ0
        current_ctrl[8] = 2.7
        apply_control(current_ctrl)
        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()
            print("here")
            time.sleep(0.01)

        # step 3

        current_ctrl[0] = -0.205   # rh_A_THJ4
        current_ctrl[1] = 0.19  # rh_A_THJ3
        current_ctrl[2] = -0.33 
        current_ctrl[9] = 0.69
        
        current_ctrl[3] = -0.34   # rh_A_FFJ4
        current_ctrl[4] = 0.62   # rh_A_FFJ3
        current_ctrl[5] = 2.7    # rh_A_FFJ0
        current_ctrl[6] = 0.34   # rh_A_FFJ0
        current_ctrl[7] = 1.5   # rh_A_FFJ0
        current_ctrl[8] = 2.7
        apply_control(current_ctrl)
        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()
            print("here")
            time.sleep(0.01)

        # step 3  release 

        # current_ctrl[0] = -0.205   # rh_A_THJ4
        # current_ctrl[1] = 0  # rh_A_THJ3
        # current_ctrl[2] = -0.33 
        # current_ctrl[9] = -0.45
        
        # current_ctrl[3] = -0.34   # rh_A_FFJ4
        # current_ctrl[4] = 0.62   # rh_A_FFJ3
        # current_ctrl[5] = 2.7    # rh_A_FFJ0
        # current_ctrl[6] = 0.34   # rh_A_FFJ0
        # current_ctrl[7] = 1.5   # rh_A_FFJ0
        # current_ctrl[8] = 2.7
        # apply_control(current_ctrl)
        # for _ in range(100):
        #     mujoco.mj_step(model, data)
        #     viewer.sync()
        #     print("here")
        #     time.sleep(0.01)

        # # step 4

        # current_ctrl[0] = -0.205   # rh_A_THJ4
        # current_ctrl[1] = 0  # rh_A_THJ3
        # current_ctrl[2] = 1.33 
        # current_ctrl[9] = -0.45
        
        # current_ctrl[3] = -0.34   # rh_A_FFJ4
        # current_ctrl[4] = 0.62   # rh_A_FFJ3
        # current_ctrl[5] = 2.7    # rh_A_FFJ0
        # current_ctrl[6] = 0.34   # rh_A_FFJ0
        # current_ctrl[7] = 1.5   # rh_A_FFJ0
        # current_ctrl[8] = 2.7
        # apply_control(current_ctrl)
        # for _ in range(100):
        #     mujoco.mj_step(model, data)
        #     viewer.sync()
        #     print("here")
        #     time.sleep(0.01)

        # # step 5

        # current_ctrl[0] = -0.205   # rh_A_THJ4
        # current_ctrl[1] = 0  # rh_A_THJ3
        # current_ctrl[2] = 1.33 
        # current_ctrl[9] = 0
        
        # current_ctrl[3] = -0.34   # rh_A_FFJ4
        # current_ctrl[4] = 0.62   # rh_A_FFJ3
        # current_ctrl[5] = 2.7    # rh_A_FFJ0
        # current_ctrl[6] = 0.34   # rh_A_FFJ0
        # current_ctrl[7] = 1.5   # rh_A_FFJ0
        # current_ctrl[8] = 2.7
        # apply_control(current_ctrl)
        # for _ in range(100):
        #     mujoco.mj_step(model, data)
        #     viewer.sync()
        #     print("here")
        #     time.sleep(0.01)
    

# === Enable gravity ===
model.opt.gravity[:] = np.array([0, 0, 0])

# === Run viewer with physics enabled ===
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    #print("Press 'A' for initial position, 'B' for target position.")
    
    while viewer.is_running():
        if should_update_ctrl:
            apply_control(current_ctrl)
            should_update_ctrl = False  # only set once per key press
        else:
            if indexcrx:
                indexCrossOver()
                should_update_ctrl = False
                indexcrx = False
            elif thumbcrx:
                thumbCrossOver()
                should_update_ctrl = False
                thumbcrx = False
            elif middlecrx:
                middleCrossOver()
                should_update_ctrl = False
                middlecrx = False
            elif t_i_pull:
                t_i_pullTrnsl1()
                should_update_ctrl = False
                t_i_pull = False

        mujoco.mj_step(model, data)
        viewer.sync()
        #time.sleep(0.01)
