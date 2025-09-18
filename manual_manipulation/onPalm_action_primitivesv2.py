import mujoco
import mujoco.viewer
import numpy as np
import time
import glfw

# === Load model and data ===
XML_PATH = r"E:\Research\MujocoProjects\RLwithHand\shadow_hand\scene_right_onPalm.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
global current_ctrl, should_update_ctrl, indexcrx, thumbcrx, middlecrx

# === Set stick pose ===
qpos_stick = np.array([0.37, -0.02,  0.055, 0.49655277, -0.4964729, 0.50350123, 0.50342479])
# mouse pose
#qpos_stick = np.array([0.35, -0.02,  0.035,  0.99352163, -0.04745924, -0.0468759, 0.09200562]) 
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

fixed_pos = np.array([0, 1.29, 2.61])
#fixed_pos = np.array([0, 0, 0])

#grasp_pose_hand = np.array([0.514, 1.33, -0.361, 0, 1.74, 0, 0, 1.74, 0]) # parfect angles
#grasp_pose_hand = np.array([0.314, 1.53, -0.361, 0, 1.5, 0, 0, 1.5, 0, 0]) 
grasp_pose_hand = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
current_ctrl = grasp_pose_hand


def apply_control(ctrl_values):
    for i, name in enumerate(actuator_names):
        actuator_id = model.actuator(name).id
        data.ctrl[actuator_id] = ctrl_values[i]

def apply_interpolated_control(target_cntrl,num_steps):
    current_pos = read_current_pos().copy()
    print(current_pos)
    traj= np.linspace(current_pos, target_cntrl, num_steps,dtype=np.float32)
    #print(traj)
    for step_vec in (traj):
        for i, name in enumerate(actuator_names):
            actuator_id = model.actuator(name).id
            data.ctrl[actuator_id] = step_vec[i]
        #mujoco.mj_step(model, data)
        #viewer.sync()
        step_for(20)


def step_for(n):
    for _ in range(n):
        #apply_control(current_ctrl)
        mujoco.mj_step(model, data)
        viewer.sync()
        #time.sleep(0.01)

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
    data.ctrl[model.actuator(actuator).id] = grasp_pose_hand[i]  # for actuator to maintain

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
p3 = False
p4 = False
p5 = False
p6 = False 
p7 = False
p8 = False 
p9 = False 
p10 = False

def key_callback(keycode):
    global current_ctrl, should_update_ctrl, indexcrx, thumbcrx, p3, p4, p5, p6, p7, p8, p9, p10
    if keycode == glfw.KEY_U:
        print("T-I-M-grasp")
        read_current_pos()
        current_ctrl[0] += del_step  # rh_A_THJ5
        current_ctrl[1] -= del_step  # rh_A_THJ4
        current_ctrl[4] += del_step  # rh_A_FFJ3
        current_ctrl[7] += del_step  # rh_A_MFJ3
        should_update_ctrl = True
    elif keycode == glfw.KEY_F:
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
    elif keycode == glfw.KEY_Z:
        print("Initial Pose")
        current_ctrl = grasp_pose_hand
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
        print("Thumb Crossover")
        p3 = True
        should_update_ctrl = False
    
    elif keycode == glfw.KEY_M:
        print("Thumb Crossover")
        p4 = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_A:
        print("Thumb Crossover")
        p5 = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_S:
        print("Thumb Crossover")
        p6 = True
        should_update_ctrl = False
    elif keycode == glfw.KEY_G:
        print("Thumb Crossover")
        p7 = True
        should_update_ctrl = False
    elif keycode == glfw.KEY_H:
        print("Thumb Crossover")
        p8 = True
        should_update_ctrl = False
    elif keycode == glfw.KEY_J:
        print("Thumb Crossover")
        p9 = True
        should_update_ctrl = False

    elif keycode == glfw.KEY_K:
        print("Thumb Crossover")
        p10 = True
        should_update_ctrl = False

# skills

def test_move():
    nnn = 100
    a = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.6])
    b = np.array([-1.05, 2, 1.57, 0, 0, 0, 0, 0, 0, -0.3]) 
    apply_interpolated_control(b, nnn)
    apply_interpolated_control(a, nnn)



def thumb_push_to_palm():
    nnn = 5

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.6]) # all below 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([-1.05, 2, 1.57, 0, 0, 0, 0, 0, 0, -0.3]) 
    apply_control(current_ctrl)
    step_for(100)
    # apply_interpolated_control(current_ctrl, nnn)

    current_ctrl = np.array([0.45, 0.94, 1.57, 0, 0, 0, 0, 0, 0, 0.6]) 
    apply_control(current_ctrl)
    step_for(100)
    #apply_interpolated_control(current_ctrl, nnn)

    current_ctrl = np.array([0.45, 0.94, 1.57, -0.349, 0, 0, 0, 0, 0, 1.5]) 
    apply_control(current_ctrl)
    step_for(100)
    #apply_interpolated_control(current_ctrl, nnn)

    current_ctrl = np.array([0.45, 0.94, 1.57, -0.349, 0, 2.5, 0, 0, 0, 1.5]) 
    apply_control(current_ctrl)
    step_for(100)
    #apply_interpolated_control(current_ctrl, nnn)

    current_ctrl = np.array([0, 0, 1.57, -0.349, 0, 2.5, 0, 0, 0, 1.5]) #
    apply_control(current_ctrl)
    step_for(100)
    #apply_interpolated_control(current_ctrl, nnn)


    current_ctrl = np.array([0, 0, 1.57, -0.349, 0, 2.5, 0, 0, 1.5, 1.5]) #
    apply_control(current_ctrl)
    step_for(100)
    #apply_interpolated_control(current_ctrl, nnn)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0.0, 1.5, 0, 0, 1.5, 1]) #
    apply_control(current_ctrl)
    step_for(100)
    #apply_interpolated_control(current_ctrl, nnn)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0.0, 1.5, 0, 0, 1.5, 0.6]) #
    apply_control(current_ctrl)
    step_for(100)
    #apply_interpolated_control(current_ctrl, nnn)


    current_ctrl = np.array([0, 0, -0.262, -0.349, 0.0, 1.5, 0, 0, 1.5, 0.6]) #
    apply_control(current_ctrl)
    step_for(100)
    #apply_interpolated_control(current_ctrl, nnn)

    # current_ctrl = np.array([1.05, 1, -0.262, -0.349, 0, 1.5, 0, 0, 1.5, 1.5]) #
    # apply_control(current_ctrl)
    # step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, 0, 0, 1.5, 0.6]) #
    apply_control(current_ctrl)
    step_for(100)
    #apply_interpolated_control(current_ctrl, nnn)


def thumb_p3():


    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.6]) # all below 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, -0.3]) 
    apply_control(current_ctrl)
    step_for(100)

    # current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, 0, 0, 2.5, 0.698]) 
    # apply_control(current_ctrl)
    # step_for(100)

    current_ctrl = np.array([-1.05, 1.16, 1.5, -0.349, 0, 1.5, -0.349, 0, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([1.05, 2, 1.5, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0.534, 1, 1.5, -0.349, 0.7, 1.5, -0.349, 0, 1.5, 0.7]) 
    apply_control(current_ctrl)
    step_for(100)
    current_ctrl = np.array([1.05, 2, 1.5, 0.349, 0.7, 2.5, -0.349, 0, 1.5, 0.7]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([-1.05, 1, 1.5, 0.349, 0.7, 2.5, -0.349, 0, 1.5, 0.7]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([-1.05, 1, 0, 0, 0, 2.5, -0.349, 0, 1.5, 0.7]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.7]) 
    apply_control(current_ctrl)
    step_for(100)



def thumb_p4():


    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.6]) # all below 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, -0.3]) 
    apply_control(current_ctrl)
    step_for(100)

    # current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, 0, 0, 2.5, 0.698]) 
    # apply_control(current_ctrl)
    # step_for(100)

    current_ctrl = np.array([-1.05, 1.16, 1.5, -0.349, 0, 1.5, -0.349, 0, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([1.05, 2, 1.5, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0.534, 1, 1.5, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.7]) 
    apply_control(current_ctrl)
    step_for(100)
    # current_ctrl = np.array([0.534, 1, 1.5, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.7]) 
    # apply_control(current_ctrl)
    # step_for(100)

    current_ctrl = np.array([0.534, 1, 1.5, -0.349, 0.7, 1.5, -0.349, 0, 1.5, 0.7]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([1.05, 2, 1.5, -0.349, 0.7, 1.5, -0.349, 0, 1.5, 0.7]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([-1.05, 1, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.7]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.7]) 
    apply_control(current_ctrl)
    step_for(100)

def ring_push():

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, 0, 0, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    #current_ctrl = np.array([0.5, 0.77, -0.262, -0.349, 0, 2.5, -0.349, 1, 2.5, 0.698]) 
    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 2.5, -0.349, 0.7, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 2.5, -0.349, 0, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.25, -0.349, 0, 1.5, 0.698]) # new release 
    apply_control(current_ctrl)
    step_for(100)

def ring_push1():

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, 0, 0, 1.5, 0.698]) # new
    apply_control(current_ctrl)
    step_for(100)

    #current_ctrl = np.array([0.5, 0.77, -0.262, -0.349, 0, 2.5, -0.349, 1, 2.5, 0.698]) 
    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 2.5, -0.349, 0.7, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 2.5, -0.349, 1.6, 0, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0.7, 0, 0.698]) # new
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.698]) # new
    apply_control(current_ctrl)
    step_for(100)

def thumb_push():
    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.698]) #
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([1.05, 1, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.698]) #
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0.7, 1.5, -0.349, 0.7, 1.5, 0.698]) # new release 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)


def index_push():

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 2.5, -0.349, 0, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, 0.349, 0.7, 2.5, -0.349, 0, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, 0.349, 0.7, 2.5, -0.349, 0.7, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, 0.349, 0.7, 1.5, -0.349, 0.7, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, 0, 0, 1.5, 0, 0, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, 0, 0, 1.5, 0.698]) # new release 
    apply_control(current_ctrl)
    step_for(100)



def digits_pull_to_palm():

    
    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0.7, 0, -0.349, 0.7, 0, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, 0, 1.4, 0, 0, 1.4, 0, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)


    current_ctrl = np.array([0, 0, -0.262, 0, 1.4, 2.5, 0, 1.4, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, 0, 0.7, 2.5, 0, 0.7, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, 0, 0, 1.5, 0, 0, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)


def digits_pull_to_palm_half():

    
    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0.7, 0, -0.349, 0.7, 0, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)



    current_ctrl = np.array([0, 0, -0.262, 0, 0.7, 1.5, 0, 0.7, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, 0, 0, 1.5, 0, 0, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)


# -----------------------------------------------------------------------------------
# could be important primitive for rolling 
def thumb_pull_to_fingers():
    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, 0, 0, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 0, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 2.5, -0.349, 0, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([.251, 0.77, -0.262, -0.349, 0.6, 2.5, -0.349, 0.6, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(300)

    current_ctrl = np.array([.251, 0.77, -0.262, -0.349, 0, 2.5, -0.349, 0, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 2.5, -0.349, 0, 2.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)


    current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, -0.349, 0, 1.5, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)


    # current_ctrl = np.array([0, 0, -0.262, -0.349, 0, 1.5, 0.349, 0, 1.5, 0.698]) 
    # apply_control(current_ctrl)
    # step_for(100)








# could be important primitive for rolling 
def thumb_pull():
    current_ctrl = np.array([-1.05, 0.0, -0.262, 0, 0, 0, 0, 0, 0, -0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([-1.05, 2.0, -0.262, -0.349, 1.6, 0, 0, 0, 0, -0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([1.05, 2.0, -0.262, -0.349, 1.6, 0, 0, 0, 0, -0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([1.05, 2.0, 1.57, -0.349, 1.6, 0, 0, 0, 0, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)

    current_ctrl = np.array([1.05, 0, 1.57, -0.349, 1.6, 0, 0, 0, 0, 0.698]) 
    apply_control(current_ctrl)
    step_for(100)



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
            if indexcrx: #V
                thumb_push_to_palm()
                should_update_ctrl = False
                indexcrx = False
            elif thumbcrx:#B
                ring_push()
                should_update_ctrl = False
                thumbcrx = False
            elif p3:#N
                index_push()
                should_update_ctrl = False
                p3 = False

            elif p4:#M
                thumb_push()
                should_update_ctrl = False
                p4 = False

            elif p5:#A
                thumb_p3()
                should_update_ctrl = False
                p5 = False

            elif p6:#S
                thumb_p4()
                should_update_ctrl = False
                p6 = False
    
            elif p7:#G
                ring_push1()
                should_update_ctrl = False
                p7 = False

            elif p8:#G
                digits_pull_to_palm()
                should_update_ctrl = False
                p8 = False


            elif p9:#G
                digits_pull_to_palm_half()
                should_update_ctrl = False
                p9 = False


            elif p10:#G
                test_move()
                should_update_ctrl = False
                p10 = False


        mujoco.mj_step(model, data)
        viewer.sync()
        #time.sleep(0.01)
