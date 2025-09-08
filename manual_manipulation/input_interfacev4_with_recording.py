# -*- coding: utf-8 -*-
"""
Shadow Hand tele-op with:
- zone mapping for thumb/index/ring
- atomic interpolated actions (ignore inputs while executing)
- TWO CSV logs:
    1) joint_positions_YYYYMMDD_HHMMSS.csv   -> time + 10 joint positions
    2) io_and_mapping_YYYYMMDD_HHMMSS.csv    -> time + raw/scaled input + zones + accepted flag + 10D control vector
"""

import sys
import time
import csv
from datetime import datetime

import numpy as np
import serial
import serial.tools.list_ports

import mujoco
import mujoco.viewer


# ------------------------------------------------------------
# Zone mapping (unchanged from your version)
# ------------------------------------------------------------
def map_to_zone(x, y, prefix):
    if 256 <= x <= 768 and 256 <= y <= 768:
        return f"{prefix}0"
    elif 768 <= x and 768 <= y:
        return f"{prefix}1"
    elif 768 <= x and 256 <= y <= 768:
        return f"{prefix}2"
    elif 768 <= x and y <= 256:
        return f"{prefix}3"
    elif 256 <= x <= 768 and y <= 256:
        return f"{prefix}4"
    elif x <= 256 and y <= 256:
        return f"{prefix}5"
    elif x <= 256 and 256 <= y <= 768:
        return f"{prefix}6"
    elif x <= 256 and 768 <= y:
        return f"{prefix}7"
    elif 256 <= x <= 768 and 768 <= y:
        return f"{prefix}8"
    else:
        return None


# ------------------------------------------------------------
# Zone → control values
# (kept as you posted; adjust as needed)
# ------------------------------------------------------------
zone_to_ctrl_values = {
    # Index finger (FFJ4, FFJ3)
    "I0": (0.00, 0.78),
    "I1": (0.34, 1.6),
    "I2": (0.00, 1.6),
    "I3": (-0.34, 1.6),
    "I8": (-0.34, 0.78),
    "I7": (-0.34, 0.00),
    "I6": (0.00, 0.00),
    "I5": (0.34, 0.00),
    "I4": (0.34, 0.78),

    # Ring finger (RFJ4, RFJ3)
    "R0": (0.00, 0.78),
    "R1": (-0.34, 1.6),
    "R2": (0.00, 1.6),
    "R3": (0.34, 1.6),
    "R8": (0.34, 0.78),
    "R7": (0.34, 0.00),
    "R6": (0.00, 0.00),
    "R5": (-0.34, 0.00),
    "R4": (-0.34, 0.78),

    # Thumb (THJ5, THJ4, THJ2, THJ1)
    "T0": (0.00, 0.00, 0.7, -0.262),
    "T7": (1.05, 2.0, 0.0, -0.262),
    "T6": (0.00, 2.0, 0.0, -0.262),
    "T5": (-1.05, 2.0, 0.7, -0.262),
    "T4": (0.00, 0.00, 0.7, -0.262),
    "T3": (1.05, 0.0, 0.7, 1.5),
    "T2": (1.05, 1.0, 0.7, 1.5),
    "T1": (1.05, 2.0, 0.7, 1.5),
    "T8": (1.05, 1.0, 0.7, 0.0),
}


# ------------------------------------------------------------
# Serial helpers
# ------------------------------------------------------------
def list_com_ports():
    return [p.device for p in serial.tools.list_ports.comports()]

def scale_to_1024(raw, raw_max=773.0):
    # Tune raw_max for your sensor’s range
    return float(raw) / raw_max * 1024.0

def try_parse_line(line):
    """
    Returns tuple (ok, arr) where arr is a float numpy array if ok else None.
    Expects at least 8 values for R(x,y), I(x,y), T(x,y) triples.
    """
    try:
        parts = [s.strip() for s in line.split(",")]
        if len(parts) < 8:
            return False, None
        arr = np.array(parts, dtype=float)
        return True, arr
    except Exception:
        return False, None

def flush_serial(ser):
    try:
        if ser is not None:
            ser.reset_input_buffer()
    except Exception:
        pass


# ------------------------------------------------------------
# Model / actuators
# ------------------------------------------------------------
XML = r"E:\Research\MujocoProjects\RLwithHand\shadow_hand\scene_right_onPalm.xml"
model = mujoco.MjModel.from_xml_path(XML)
data  = mujoco.MjData(model)

# # Optional: set object free joint pose (your values)
# qpos_stick = np.array([3.22260647e-01, -0.02,  0.095, 0.49655277, -0.4964729, 0.50350123, 0.50342479])
# joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
# qpos_addr = model.jnt_qposadr[joint_id]
# data.qpos[qpos_addr : qpos_addr + 7] = qpos_stick
# mujoco.mj_forward(model, data)  # compute derived quantities after direct state set


# object id placement 
qpos_stick = np.array([3.22260647e-01, -0.02,  0.095, 0.49655277, -0.4964729, 0.50350123, 0.50342479])
OBJ_JOINT_NAME = "object_freejoint"

obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJ_JOINT_NAME)
if obj_jid < 0:
    raise RuntimeError(f"Joint '{OBJ_JOINT_NAME}' not found")

OBJ_QPOS_ADR = model.jnt_qposadr[obj_jid]  # start index in data.qpos (size 7 for free joint)
OBJ_QVEL_ADR = model.jnt_dofadr[obj_jid]   # start index in data.qvel (size 6 for free joint)


RESET_HEIGHT_Z = 0.025  # meters (z threshold)

def get_object_xyz():
    """Return (x, y, z) position of the free joint frame."""
    return data.qpos[OBJ_QPOS_ADR:OBJ_QPOS_ADR+3].copy()

def reset_object_pose():
    """Reset object pose and zero its velocity, then recompute derived quantities."""
    # qpos_stick must be the 7D [x,y,z, qw,qx,qy,qz] you already defined
    data.qpos[OBJ_QPOS_ADR:OBJ_QPOS_ADR+7] = qpos_stick
    data.qvel[OBJ_QVEL_ADR:OBJ_QVEL_ADR+6] = 0.0
    mujoco.mj_forward(model, data)  # important after direct state write



# Actuator names (10 channels you control)
CTRL_NAMES = [
    "rh_A_THJ5", "rh_A_THJ4", "rh_A_THJ2", "rh_A_THJ1",
    "rh_A_FFJ4", "rh_A_FFJ3", "rh_A_FFJ0",
    "rh_A_RFJ4", "rh_A_RFJ3", "rh_A_RFJ0",
]
CTRL_IDX = np.array([
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm) for nm in CTRL_NAMES
], dtype=int)

# For reading the 10 joints driven by those actuators
ACTUATOR_TO_JOINT = np.array([model.actuator_trnid[i, 0] for i in CTRL_IDX], dtype=int)
JOINT_QPOS_ADR    = np.array([model.jnt_qposadr[jid] for jid in ACTUATOR_TO_JOINT], dtype=int)

# fixed actuator position 

fixed_actuators = [
    "rh_A_LFJ4", "rh_A_LFJ3", "rh_A_LFJ0"
]
fixed_pos = np.array([0, 1.29, 2.61], dtype=np.float32)

FIXED_IDX = np.array([
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm) for nm in fixed_actuators
], dtype=int)

data.ctrl[FIXED_IDX] = fixed_pos



# ------------------------------------------------------------
# Read current 10-joint positions
# ------------------------------------------------------------
def read_current_pos():
    # For hinge joints, a single qpos index is fine
    return data.qpos[JOINT_QPOS_ADR].astype(float).copy()


# ------------------------------------------------------------
# Interpolated control (atomic action)
# ------------------------------------------------------------
def apply_interpolated_control(viewer, initial_cntrl, target_cntrl, num_steps, mj_steps_per_point=6):
    traj = np.linspace(initial_cntrl, target_cntrl, num_steps, dtype=np.float32)
    for step_vec in traj:
        data.ctrl[CTRL_IDX] = step_vec
        for _ in range(mj_steps_per_point):
            mujoco.mj_step(model, data)
            viewer.sync()


# ------------------------------------------------------------
# CSV logging setup
# ------------------------------------------------------------
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

CSV_DIR = r"E:\Research\MujocoProjects\RLwithHand\input_interface"

JOINT_CSV_PATH = fr"{CSV_DIR}\joint_positions_{_ts}.csv"
IO_CSV_PATH    = fr"{CSV_DIR}\io_and_mapping_{_ts}.csv"


def now_ms():
    return int(time.monotonic() * 1000)

def open_csvs():
    joint_csv = open(JOINT_CSV_PATH, "w", newline="")
    io_csv    = open(IO_CSV_PATH, "w", newline="")
    jw = csv.writer(joint_csv)
    iw = csv.writer(io_csv)

    jw.writerow(["t_ms", *CTRL_NAMES])

    io_header = (
        ["t_ms"] +
        [f"raw{i}" for i in range(8)] +
        ["rx_s","ry_s","ix_s","iy_s","tx_s","ty_s"] +
        ["zoneI","zoneT","zoneR","ffj0_val","rfj0_val","action_busy","accepted"] +
        [f"ctrl_{nm}" for nm in CTRL_NAMES]
    )
    iw.writerow(io_header)
    return joint_csv, io_csv, jw, iw

def log_joint_positions(joint_writer):
    joint_writer.writerow([now_ms(), *read_current_pos().tolist()])

def log_io_row(io_writer, raw_vals, rx_s, ry_s, ix_s, iy_s, tx_s, ty_s,
               zoneI, zoneT, zoneR, ffj0_val, rfj0_val, action_busy, accepted, control_vec):
    row = (
        [now_ms()] +
        [float(raw_vals[i]) if i < len(raw_vals) else "" for i in range(8)] +
        [rx_s, ry_s, ix_s, iy_s, tx_s, ty_s] +
        [zoneI, zoneT, zoneR, ffj0_val, rfj0_val, int(bool(action_busy)), int(bool(accepted))] +
        control_vec.tolist()
    )
    io_writer.writerow(row)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # Serial open (first available port; adjust if needed)
    ports = list_com_ports()
    ser = None
    if not ports:
        print("No COM ports found.")
    else:
        print("Available COM ports:", ports)
        try:
            ser = serial.Serial(ports[0], 9600, timeout=0)  # non-blocking
            print(f"Listening on {ports[0]} @ 9600 baud")
        except Exception as e:
            print("Failed to open serial:", e)
            ser = None

    # Initial zones and toggles
    last_zoneI = "I0"
    last_zoneT = "T0"
    last_zoneR = "R0"
    prev_I, prev_T, prev_R = last_zoneI, last_zoneT, last_zoneR

    ffj0_val = 0.0
    rfj0_val = 0.0
    prev_ffj0_val = ffj0_val
    prev_rfj0_val = rfj0_val

    action_busy = False
    prev_control = np.zeros(len(CTRL_NAMES), dtype=np.float32)
    last_control_signals = prev_control.copy()

    # CSVs
    joint_csv, io_csv, joint_writer, io_writer = open_csvs()
    print("Logging to:\n  ", JOINT_CSV_PATH, "\n  ", IO_CSV_PATH)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                line = ""  # avoid stale usage

                # ---- serial read (only when idle) ----
                if ser is not None:
                    try:
                        if not action_busy:
                            line = ser.readline().decode(errors="ignore").strip()
                        if line:
                            ok, arr = try_parse_line(line)
                            if ok:
                                # Layout assumption:
                                #   R: arr[0], arr[1]
                                #   (arr[2] toggles rfj0)
                                #   I: arr[3], arr[4]
                                #   (arr[5] toggles ffj0)
                                #   T: arr[6], arr[7]
                                rx, ry = scale_to_1024(arr[0]), scale_to_1024(arr[1])
                                ix, iy = scale_to_1024(arr[3]), scale_to_1024(arr[4])
                                tx, ty = scale_to_1024(arr[6]), scale_to_1024(arr[7])

                                id_I = map_to_zone(ix, iy, "I")
                                id_T = map_to_zone(tx, ty, "T")
                                id_R = map_to_zone(rx, ry, "R")

                                if id_I in zone_to_ctrl_values:
                                    last_zoneI = id_I
                                if id_T in zone_to_ctrl_values:
                                    last_zoneT = id_T
                                if id_R in zone_to_ctrl_values:
                                    last_zoneR = id_R

                                # Base joint toggles (keep previous if value not 0/1)
                                rfj0_val = 2.5 if arr[2] == 0 else (0.7 if arr[2] == 1 else rfj0_val)
                                ffj0_val = 2.5 if arr[5] == 0 else (0.7 if arr[5] == 1 else ffj0_val)

                                # Prospective control from current zones (even if busy)
                                if (
                                    (last_zoneI in zone_to_ctrl_values) and
                                    (last_zoneT in zone_to_ctrl_values) and
                                    (last_zoneR in zone_to_ctrl_values)
                                ):
                                    ffj4_val, ffj3_val = zone_to_ctrl_values[last_zoneI]
                                    rfj4_val, rfj3_val = zone_to_ctrl_values[last_zoneR]
                                    thj5_val, thj4_val, thj2_val, thj1_val = zone_to_ctrl_values[last_zoneT]
                                    prospective_control = np.array([
                                        thj5_val, thj4_val, thj2_val, thj1_val,
                                        ffj4_val, ffj3_val, ffj0_val,
                                        rfj4_val, rfj3_val, rfj0_val
                                    ], dtype=np.float32)
                                else:
                                    prospective_control = last_control_signals.copy()

                                # Will we accept (start) a new action this cycle?
                                changed = (
                                    (prev_I != last_zoneI) or
                                    (prev_R != last_zoneR) or
                                    (prev_T != last_zoneT) or
                                    (prev_rfj0_val != rfj0_val) or
                                    (prev_ffj0_val != ffj0_val)
                                )
                                will_be_accepted = changed and (not action_busy)

                                # Log I/O row
                                log_io_row(
                                    io_writer=io_writer,
                                    raw_vals=arr,
                                    rx_s=rx, ry_s=ry, ix_s=ix, iy_s=iy, tx_s=tx, ty_s=ty,
                                    zoneI=last_zoneI, zoneT=last_zoneT, zoneR=last_zoneR,
                                    ffj0_val=ffj0_val, rfj0_val=rfj0_val,
                                    action_busy=action_busy, accepted=will_be_accepted,
                                    control_vec=prospective_control
                                )
                    except Exception:
                        # swallow transient serial issues
                        pass

                # ---- gate: start a new action only if changed and idle ----
                changed = (
                    (prev_I != last_zoneI) or
                    (prev_R != last_zoneR) or
                    (prev_T != last_zoneT) or
                    (prev_rfj0_val != rfj0_val) or
                    (prev_ffj0_val != ffj0_val)
                )
                if changed and (not action_busy):
                    # compute control signals from current latched zones
                    ffj4_val, ffj3_val = zone_to_ctrl_values[last_zoneI]
                    rfj4_val, rfj3_val = zone_to_ctrl_values[last_zoneR]
                    thj5_val, thj4_val, thj2_val, thj1_val = zone_to_ctrl_values[last_zoneT]

                    control_signals = np.array([
                        thj5_val, thj4_val, thj2_val, thj1_val,
                        ffj4_val, ffj3_val, ffj0_val,
                        rfj4_val, rfj3_val, rfj0_val
                    ], dtype=np.float32)

                    action_busy = True
                    flush_serial(ser)  # drop queued bytes so this action stays clean

                    # Execute atomic trajectory
                    apply_interpolated_control(
                        viewer=viewer,
                        initial_cntrl=prev_control,
                        target_cntrl=control_signals,
                        num_steps=70,
                        mj_steps_per_point=6
                    )

                    prev_control = control_signals
                    last_control_signals = control_signals
                    action_busy = False
                    # print("Applied new control:", control_signals)

                # ---- run sim one step and log joint positions every tick ----
                mujoco.mj_step(model, data)
                viewer.sync()
                log_joint_positions(joint_writer)

                x, y, z = get_object_xyz()
                if z < RESET_HEIGHT_Z:
                    reset_object_pose()

                # update previous latches
                prev_I = last_zoneI
                prev_T = last_zoneT
                prev_R = last_zoneR
                prev_ffj0_val = ffj0_val
                prev_rfj0_val = rfj0_val

    finally:
        # Close serial and CSVs
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        try:
            io_csv.flush(); io_csv.close()
            joint_csv.flush(); joint_csv.close()
        except Exception:
            pass
        print(f"Saved joint logs to: {JOINT_CSV_PATH}")
        print(f"Saved I/O + mapping logs to: {IO_CSV_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user; exiting.")
        sys.exit(0)
