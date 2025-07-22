# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 11:27:39 2025

@author: neeltron
"""

import sys, serial, pygame, mujoco, numpy as np
from mujoco import viewer
from collections import deque

model = mujoco.MjModel.from_xml_path("shadow_hand/scene_right.xml")
data  = mujoco.MjData(model)

model.opt.gravity[:] = (0.0, 0.0, -9.81)

hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_forearm")
queue = deque([hand_body_id])
while queue:
    bid = queue.popleft()
    model.body_gravcomp[bid] = 1.0
    queue.extend([i for i in range(model.nbody) if model.body_parentid[i] == bid])

# js1 = middle, js2 = thumb, js3 = index

thumb_ids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("rh_A_THJ4", "rh_A_THJ2", "rh_A_THJ1")]
index_ids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("rh_A_FFJ4", "rh_A_FFJ3", "rh_A_FFJ0")]
middle_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("rh_A_MFJ4", "rh_A_MFJ3", "rh_A_MFJ0")]

root_jid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rh_root_free")
ROOT_QPOS  = model.jnt_qposadr[root_jid]
ROOT_QVEL  = model.jnt_dofadr[root_jid]

initial_pos  = data.qpos[ROOT_QPOS:ROOT_QPOS+3].copy()
initial_quat = data.qpos[ROOT_QPOS+3:ROOT_QPOS+7].copy()

target_z   = float(initial_pos[2])
last_pos   = initial_pos.copy()
rotation_mode = False
control_mode = 1

port = serial.Serial("COM9", 9600, timeout=0.005)
pygame.init()
pygame.display.set_mode((320, 200))
pygame.display.set_caption("Hand Control – click here")
pygame.key.set_repeat(1, 10)

MOVE_STEP_XY = 0.002
MOVE_STEP_Z  = 0.002
ROT_STEP     = np.deg2rad(0.5)

def map_axis(val, mj_id, invert=False):
    lo, hi = model.actuator_ctrlrange[mj_id]
    ctrl = lo + (hi - lo) * ((1023-val)/1023 if invert else val/1023)
    return max(lo, min(hi, ctrl))

def map_bipolar_discrete(val, mj_id):
    lo, hi = model.actuator_ctrlrange[mj_id]
    mid = 512
    step = (hi - lo) / 2
    if val > mid + 200:
        return hi
    elif val < mid - 200:
        return lo
    else:
        return lo + step

def map_bipolar_discrete_flipped(val, mj_id):
    lo, hi = model.actuator_ctrlrange[mj_id]
    mid = 512
    step = (hi - lo) / 2
    if val > mid + 200:
        return lo  # Reversed
    elif val < mid - 200:
        return hi  # Reversed
    else:
        return lo + step

def small_quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= (np.linalg.norm(axis) + 1e-9)
    half = angle * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), *(s*axis)])

def mul_quat(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        pkt = port.readline().decode('utf-8', errors='ignore').strip()
        if pkt:
            try:
                x1,y1,b1, x2,y2,b2, x3,y3,b3 = map(int, pkt.split(','))

                if control_mode == 1:
                    data.ctrl[middle_ids[0]] = map_axis(y1, middle_ids[0])
                    data.ctrl[middle_ids[1]] = map_axis(x1, middle_ids[1])
                    data.ctrl[thumb_ids[0]]  = map_axis(y3, thumb_ids[0])
                    data.ctrl[thumb_ids[1]]  = map_axis(x3, thumb_ids[1])
                    data.ctrl[index_ids[0]]  = map_axis(y2, index_ids[0])
                    data.ctrl[index_ids[1]]  = map_axis(x2, index_ids[1])
                    data.ctrl[thumb_ids[2]]  = map_axis(x3, thumb_ids[2])
                    data.ctrl[index_ids[2]]  = map_axis(x2, index_ids[2])
                    data.ctrl[middle_ids[2]] = map_axis(x1, middle_ids[2])

                elif control_mode == 2:
                    # Finger tip joint (joint 0) bending on joystick button press
                    for btn, aid in zip([b1, b3, b2], [middle_ids[2], thumb_ids[2], index_ids[2]]):
                        lo, hi = model.actuator_ctrlrange[aid]
                        data.ctrl[aid] = lo if btn else hi

                    # Bipolar control for proximal & lateral joints
                    data.ctrl[middle_ids[0]] = map_bipolar_discrete_flipped(y1, middle_ids[0])
                    data.ctrl[middle_ids[1]] = map_bipolar_discrete(x1, middle_ids[1])
                    data.ctrl[thumb_ids[0]]  = map_bipolar_discrete(y3, thumb_ids[0])
                    data.ctrl[thumb_ids[1]]  = map_bipolar_discrete(x3, thumb_ids[1])
                    data.ctrl[index_ids[0]]  = map_bipolar_discrete_flipped(y2, index_ids[0])
                    data.ctrl[index_ids[1]]  = map_bipolar_discrete(x2, index_ids[1])

            except ValueError:
                print("Bad packet:", pkt)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    rotation_mode = True
                    last_pos = data.qpos[ROOT_QPOS:ROOT_QPOS+3].copy()
                    print("Rotation")
                if e.key == pygame.K_t:
                    rotation_mode = False
                    target_z = float(data.qpos[ROOT_QPOS+2])
                    last_pos = data.qpos[ROOT_QPOS:ROOT_QPOS+3].copy()
                    print("Translation")
                if e.key == pygame.K_1:
                    control_mode = 1
                    print("Normal")
                if e.key == pygame.K_2:
                    control_mode = 2
                    print("Bipolar Discrete")

        keys = pygame.key.get_pressed()
        data.qvel[ROOT_QVEL:ROOT_QVEL+6] = 0

        if rotation_mode:
            quat = data.qpos[ROOT_QPOS+3:ROOT_QPOS+7].copy()
            if keys[pygame.K_w]: quat = mul_quat(quat, small_quat([1,0,0],  ROT_STEP))
            if keys[pygame.K_s]: quat = mul_quat(quat, small_quat([1,0,0], -ROT_STEP))
            if keys[pygame.K_a]: quat = mul_quat(quat, small_quat([0,0,1],  ROT_STEP))
            if keys[pygame.K_d]: quat = mul_quat(quat, small_quat([0,0,1], -ROT_STEP))
            if keys[pygame.K_q]: quat = mul_quat(quat, small_quat([0,1,0],  ROT_STEP))
            if keys[pygame.K_e]: quat = mul_quat(quat, small_quat([0,1,0], -ROT_STEP))
            data.qpos[ROOT_QPOS+3:ROOT_QPOS+7] = quat / (np.linalg.norm(quat)+1e-9)
            data.qpos[ROOT_QPOS:ROOT_QPOS+3] = last_pos
        else:
            if keys[pygame.K_w]: data.qpos[ROOT_QPOS+1] += MOVE_STEP_XY
            if keys[pygame.K_s]: data.qpos[ROOT_QPOS+1] -= MOVE_STEP_XY
            if keys[pygame.K_a]: data.qpos[ROOT_QPOS]   -= MOVE_STEP_XY
            if keys[pygame.K_d]: data.qpos[ROOT_QPOS]   += MOVE_STEP_XY
            if keys[pygame.K_q]: target_z += MOVE_STEP_Z
            if keys[pygame.K_e]: target_z -= MOVE_STEP_Z
            data.qpos[ROOT_QPOS+2] = target_z
            last_pos = data.qpos[ROOT_QPOS:ROOT_QPOS+3].copy()

        mujoco.mj_step(model, data)
        data.qvel[ROOT_QVEL:ROOT_QVEL+3] = 0
        if rotation_mode:
            data.qpos[ROOT_QPOS:ROOT_QPOS+3] = last_pos

        v.sync()
