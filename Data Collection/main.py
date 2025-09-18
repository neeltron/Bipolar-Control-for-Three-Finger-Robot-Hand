# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 15:00:43 2025

@author: neel
"""

# -*- coding: utf-8 -*-
"""
Shadow Hand tele-op with:
- zone mapping for thumb/index/ring
- atomic interpolated actions (ignore inputs while executing)
- tanh(k)-based S-curve easing for atomic trajectories (adjust k per participant)
- STARTUP TUTORIAL: plays MP4 slides in a side window; SPACE to advance.
  Cube is hidden during tutorial and appears after the last slide.
- TWO CSV logs:
    1) joint_positions_YYYYMMDD_HHMMSS.csv
    2) io_and_mapping_YYYYMMDD_HHMMSS.csv
"""

import os
import sys
import time
import csv
from datetime import datetime
from glob import glob

import numpy as np
import serial
import serial.tools.list_ports

import mujoco
import mujoco.viewer

# ---------- Tutorial window deps ----------
import pygame

# Prefer OpenCV; otherwise fall back to ImageIO (v3 or v2)
_BACKEND = None
try:
    import cv2  # works with numpy>=2 if opencv-python>=4.10
    _ = cv2.UMat
    _BACKEND = "opencv"
except Exception:
    cv2 = None

# Try imageio v3 first
try:
    import imageio.v3 as iio3
    _BACKEND = _BACKEND or "imageio_v3"
except Exception:
    iio3 = None

# Then imageio v2
try:
    import imageio as iio2
    if _BACKEND is None:
        _BACKEND = "imageio_v2"
except Exception:
    iio2 = None

if _BACKEND is None:
    print("[WARN] No video backend available. Install one of:\n"
          "  pip install -U opencv-python\n"
          "  pip install imageio imageio-ffmpeg")

# ------------------------------------------------------------
# Tutorial / slides configuration
# ------------------------------------------------------------
VIDEO_DIR    = "Tutorial_Videos"   # put your MP4 files here
SLIDE_WIN_W  = 880                 # slide window width (px)
SLIDE_WIN_H  = 660                 # slide window height (px)
ADVANCE_KEYS = {pygame.K_SPACE}    # keys that advance slides

def natural_sort_key(s: str):
    import re, os
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(s))]

class TutorialPlayer:
    """
    MP4 player using OpenCV if available, else ImageIO (v3 or v2).
    SPACE advances to next video; when a video ends, last frame is held until SPACE.
    """
    def __init__(self, video_dir, size):
        self.backend = _BACKEND
        self.size = size
        self.screen = None
        self.paths = []
        self.idx = 0
        self.enabled = self.backend is not None

        if not self.enabled:
            return

        from glob import glob
        import os
        for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.m4v", "*.M4V"):
            self.paths.extend(glob(os.path.join(video_dir, ext)))
        self.paths = sorted(self.paths, key=natural_sort_key)

        if not self.paths:
            self.enabled = False
            print(f"[INFO] No tutorial videos found in '{video_dir}'. Skipping tutorial.")
            return

        # Init pygame window
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "40,40")
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Tutorial (SPACE to advance)")
        self.screen.fill((8, 8, 8))
        pygame.display.flip()

        # Backend state
        self.cap = None        # OpenCV
        self.reader = None     # imageio v2 / v3 object
        self.frame_iter = None
        self.fps = 30.0
        self.frame_interval = 1.0 / self.fps
        self.next_time = time.monotonic()
        self.frame_rgb = None
        self.ended = False

        self._open_current()
        print(f"[Tutorial] Backend: {self.backend}")
        print(f"[Tutorial] Loaded {len(self.paths)} video(s). Press SPACE to advance.")

    def _open_current(self):
        path = self.paths[self.idx]

        # Close previous
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        if self.reader is not None:
            try:
                close = getattr(self.reader, "close", None)
                if callable(close): close()
            except Exception:
                pass
            self.reader = None
        self.frame_iter = None

        # Open new
        if self.backend == "opencv":
            try:
                self.cap = cv2.VideoCapture(path)
                if not self.cap.isOpened():
                    raise RuntimeError("cv2.VideoCapture failed")
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.fps = fps if fps and fps > 0 else 30.0
            except Exception as e:
                print(f"[WARN] OpenCV failed on '{os.path.basename(path)}' ({e}). Falling back to ImageIO.")
                # Prefer v3, then v2
                if iio3 is not None:
                    self.backend = "imageio_v3"
                elif iio2 is not None:
                    self.backend = "imageio_v2"
                else:
                    self.ended = True
                    return

        if self.backend == "imageio_v3":
            # imageio v3 path (requires pyav / imageio-ffmpeg)
            try:
                # imopen may not exist in some builds; imopen + pyav preferred
                if hasattr(iio3, "imopen"):
                    self.reader = iio3.imopen(path, "r", plugin="pyav")
                    self.frame_iter = iio3.imiter(self.reader)
                    meta = getattr(self.reader, "metadata", None)
                    fps = (meta or {}).get("fps", None)
                else:
                    # Fallback: direct imiter from file path
                    self.frame_iter = iio3.imiter(path)
                    fps = None
            except Exception:
                # Last resort: try v2
                if iio2 is not None:
                    self.backend = "imageio_v2"
                else:
                    raise

            if self.backend == "imageio_v3" and self.frame_iter is not None:
                self.fps = fps if fps and fps > 0 else 30.0

        if self.backend == "imageio_v2":
            # imageio v2 path
            if iio2 is None:
                raise RuntimeError("imageio v2 not available")
            self.reader = iio2.get_reader(path)   # needs imageio-ffmpeg
            try:
                meta = self.reader.get_meta_data()
                fps = meta.get("fps", None)
            except Exception:
                fps = None
            self.fps = fps if fps and fps > 0 else 30.0
            self.frame_iter = iter(self.reader)

        self.frame_interval = 1.0 / self.fps
        self.next_time = time.monotonic()
        self.ended = False
        self.frame_rgb = None
        print(f"[Tutorial] Now playing: {os.path.basename(path)} @ {self.fps:.2f} fps")

    def _blit_rgb(self, frame_rgb):
        """Letterbox to window and blit using pygame."""
        h, w = frame_rgb.shape[:2]
        target_w, target_h = self.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        surf = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), 'RGB')
        if (new_w, new_h) != (w, h):
            surf = pygame.transform.smoothscale(surf, (new_w, new_h))

        self.screen.fill((8, 8, 8))
        self.screen.blit(surf, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        pygame.display.flip()
        self.frame_rgb = frame_rgb

    def _read_frame(self):
        if self.backend == "opencv":
            ok, frame_bgr = self.cap.read()
            if not ok:
                self.ended = True
                return
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        elif self.backend == "imageio_v3":
            try:
                frame = next(self.frame_iter)
            except StopIteration:
                self.ended = True
                return
            frame_rgb = np.asarray(frame)
            if frame_rgb.ndim == 2:
                frame_rgb = np.stack([frame_rgb]*3, axis=-1)
            if frame_rgb.dtype != np.uint8:
                frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

        else:  # imageio_v2
            try:
                frame = next(self.frame_iter)
            except StopIteration:
                self.ended = True
                return
            frame_rgb = np.asarray(frame)
            if frame_rgb.ndim == 2:
                frame_rgb = np.stack([frame_rgb]*3, axis=-1)
            if frame_rgb.dtype != np.uint8:
                frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

        self._blit_rgb(frame_rgb)

    def poll(self):
        """Call every loop. Returns True while tutorial is running."""
        if not self.enabled:
            return False

        advance = False
        quit_requested = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                quit_requested = True
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                advance = True

        if quit_requested:
            self.close()
            return False

        if advance:
            if self.idx + 1 < len(self.paths):
                self.idx += 1
                self._open_current()
            else:
                self.close()
                return False

        now = time.monotonic()
        if not self.ended and now >= self.next_time:
            self._read_frame()
            self.next_time = now + (1.0 / max(1e-6, self.fps))

        if self.ended and self.frame_rgb is not None:
            # keep last frame drawn
            self._blit_rgb(self.frame_rgb)

        return True

    def close(self):
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        if self.reader is not None:
            try:
                close = getattr(self.reader, "close", None)
                if callable(close): close()
            except Exception:
                pass
            self.reader = None
        try:
            pygame.quit()
        except Exception:
            pass
        self.enabled = False



# ------------------------------------------------------------
# Zone mapping
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
    return float(raw) / raw_max * 1024.0

def try_parse_line(line):
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
XML = r"scene_right_onPalm.xml"
model = mujoco.MjModel.from_xml_path(XML)
data  = mujoco.MjData(model)

# object placement
qpos_stick = np.array([3.22260647e-01, -0.02,  0.095, 0.49655277, -0.4964729, 0.50350123, 0.50342479])
OBJ_JOINT_NAME = "object_freejoint"

obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJ_JOINT_NAME)
if obj_jid < 0:
    raise RuntimeError(f"Joint '{OBJ_JOINT_NAME}' not found")

OBJ_QPOS_ADR = model.jnt_qposadr[obj_jid]
OBJ_QVEL_ADR = model.jnt_dofadr[obj_jid]

# convenience body/geom lookup for showing/hiding the object
OBJ_BODY_ID   = model.jnt_bodyid[obj_jid]
OBJ_GEOM_IDS  = np.where(model.geom_bodyid == OBJ_BODY_ID)[0]
_ORIG_RGBA    = model.geom_rgba.copy()
_ORIG_CONTYPE = model.geom_contype.copy()
_ORIG_CONAFF  = model.geom_conaffinity.copy()

RESET_HEIGHT_Z = 0.025  # m

def get_object_xyz():
    return data.qpos[OBJ_QPOS_ADR:OBJ_QPOS_ADR+3].copy()

def reset_object_pose():
    data.qpos[OBJ_QPOS_ADR:OBJ_QPOS_ADR+7] = qpos_stick
    data.qvel[OBJ_QVEL_ADR:OBJ_QVEL_ADR+6] = 0.0
    mujoco.mj_forward(model, data)

def hide_object_geoms():
    for gid in OBJ_GEOM_IDS:
        model.geom_rgba[gid, 3] = 0.0
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
    mujoco.mj_forward(model, data)

def show_object_geoms():
    for gid in OBJ_GEOM_IDS:
        model.geom_rgba[gid] = _ORIG_RGBA[gid]
        model.geom_contype[gid] = _ORIG_CONTYPE[gid]
        model.geom_conaffinity[gid] = _ORIG_CONAFF[gid]
    mujoco.mj_forward(model, data)


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

# fixed actuator positions for little finger
fixed_actuators = ["rh_A_LFJ4", "rh_A_LFJ3", "rh_A_LFJ0"]
fixed_pos = np.array([0, 1.29, 2.61], dtype=np.float32)
FIXED_IDX = np.array([
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm) for nm in fixed_actuators
], dtype=int)
data.ctrl[FIXED_IDX] = fixed_pos


# ------------------------------------------------------------
# Read current 10-joint positions
# ------------------------------------------------------------
def read_current_pos():
    return data.qpos[JOINT_QPOS_ADR].astype(float).copy()


# ------------------------------------------------------------
# S-curve easing (tanh with adjustable k)
# ------------------------------------------------------------
EASE_KIND = "tanh"
EASE_K    = 3.0     # adjustable k (1–2 novice, 3–4 intermediate, 5–8 advanced)
EASE_GAIN = 8.0 

def s_curve_ease(t: np.ndarray, kind: str = EASE_KIND, k: float = EASE_K, gain: float = EASE_GAIN) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    if kind == "tanh":
        if k <= 0:
            return t
        x = 2.0 * t - 1.0
        y = np.tanh(k * x) / np.tanh(k)
        return 0.5 * (y + 1.0)
    if kind == "quintic":
        return t**3 * (10 - 15*t + 6*t*t)
    if kind == "cubic":
        return t*t*(3 - 2*t)
    if kind == "logistic":
        x = gain * (t - 0.5)
        sig = 1.0 / (1.0 + np.exp(-x))
        s0 = 1.0 / (1.0 + np.exp( gain * 0.5))
        s1 = 1.0 / (1.0 + np.exp(-gain * 0.5))
        return (sig - s0) / (s1 - s0)
    return t  # fallback linear


# ------------------------------------------------------------
# Interpolated control (atomic action) using S-curve easing
# ------------------------------------------------------------
def apply_interpolated_control(viewer, initial_cntrl, target_cntrl, num_steps, mj_steps_per_point=6):
    t = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    te = s_curve_ease(t)  # uses EASE_KIND / EASE_K
    delta = (target_cntrl - initial_cntrl).astype(np.float32)

    for alpha in te:
        step_vec = initial_cntrl + alpha * delta
        data.ctrl[CTRL_IDX] = step_vec
        for _ in range(mj_steps_per_point):
            mujoco.mj_step(model, data)
            viewer.sync()


# ------------------------------------------------------------
# CSV logging setup
# ------------------------------------------------------------
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

JOINT_CSV_PATH = fr"Joint_Data\joint_positions_{_ts}.csv"
IO_CSV_PATH    = fr"Mapped_Data\io_and_mapping_{_ts}.csv"

def now_ms():
    return int(time.monotonic() * 1000)

def open_csvs():
    os.makedirs(os.path.dirname(JOINT_CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(IO_CSV_PATH), exist_ok=True)

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
            chosen = ports[0]
            ser = serial.Serial(chosen, 9600, timeout=0)  # non-blocking
            print(f"Listening on {chosen} @ 9600 baud")
        except Exception as e:
            print("Failed to open serial:", e)
            ser = None

    # ---- Tutorial setup ----
    # Hide the object (cube) for the tutorial phase
    hide_object_geoms()

    # Prepare MP4 tutorial player
    tplayer = TutorialPlayer(video_dir=VIDEO_DIR, size=(SLIDE_WIN_W, SLIDE_WIN_H))
    tutorial_active = bool(tplayer.enabled)

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
                # ---- Tutorial loop (non-blocking) ----
                if tutorial_active:
                    tutorial_active = tplayer.poll()
                    if not tutorial_active:
                        # finished/closed -> reveal object (cube)
                        show_object_geoms()
                        print("Tutorial complete. Cube is now visible.")

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

                    # Execute atomic trajectory (S-curve eased via tanh(k))
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

                # ---- run sim one step and log joint positions every tick ----
                mujoco.mj_step(model, data)
                viewer.sync()
                log_joint_positions(joint_writer)

                _, _, z = get_object_xyz()
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

        # Close tutorial window if still open
        try:
            if 'tplayer' in locals() and tplayer.enabled:
                tplayer.close()
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
