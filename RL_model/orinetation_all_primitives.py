import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import glfw
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import time

class PrimitiveOnlyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, model_path, primitives, max_steps=1000, render=False, frame_skip=20):
        super().__init__()
        # ----- MuJoCo core -----
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_enabled = render
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.current_step = 0

        if self.render_enabled:
            glfw.init()
            self.window = glfw.create_window(800, 600, "MuJoCo Viewer", None, None)
            glfw.make_context_current(self.window)
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = self.model.stat.center
            self.cam.distance = 0.5
            self.cam.azimuth = 90
            self.cam.elevation = -65
            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        else:
            self.window = None

        # ----- Actuators (same as your base env) -----
        self.actuator_names = [
            "rh_A_THJ5", "rh_A_THJ4", "rh_A_THJ1",
            "rh_A_FFJ4", "rh_A_FFJ3", "rh_A_FFJ0",
            "rh_A_RFJ4", "rh_A_RFJ3", "rh_A_RFJ0", "rh_A_THJ2"
        ]
        self.df_digits = ["rh_A_LFJ4", "rh_A_LFJ3", "rh_A_LFJ0"]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                             for n in self.actuator_names]
        self.df_actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                                for n in self.df_digits]
        assert all(aid != -1 for aid in self.actuator_ids)
        self.grasp_pose_hand = np.zeros(len(self.actuator_ids), dtype=np.float32)
        self.df_digit_poses = np.array([0.0, 2.0, 2.42], dtype=np.float32)

        # ----- Primitives -----
        # primitives must be a list of (L, n) arrays with values in {-1,0,+1}
        self.P = [np.asarray(p, dtype=np.int8) for p in primitives]
        self.K = len(self.P)
        assert self.K >= 1
        self.action_space = spaces.Discrete(self.K)  # choose primitive ID

        # ----- Observation space (same shapes you used) -----
        self.n_finger_joints = len(self.actuator_names)
        self.obs_dim = self.n_finger_joints + 7
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
            'prev_action': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_finger_joints,), dtype=np.float32),
            'Obj_shape':   spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_finger_joints,), dtype=np.float32),
        })
        self.prev_action = np.zeros(self.n_finger_joints, dtype=np.float32)
        #self.rot_qpos_stick = np.array([0.755, -0.01, 0.065, 0.496, -0.496, 0.503, 0.503], dtype=np.float32)
        self.rot_qpos_stick = np.array([0.355, -0.01, 0.06, 0.496, -0.496, 0.503, 0.503], dtype=np.float32)

        self.prev_face = -1
        self.prev_face_or = -1
        self.face_penalty = 0

    # -------- helpers you already have (object_state, quaternion_to_euler_z, _quat_* etc.) --------
    # ... paste your helper methods (_get_obs, object_state, compute_reward, render, etc.) unchanged ...
    # The only change will be in step(): we unroll a chosen primitive.

# --------- Helpers ---------

    def object_state(self, obj_pos, obs_quat):

        def quat_to_euler(q):
            w, x, y, z = q

            # roll (x-axis rotation)
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll = np.arctan2(t0, t1)

            # pitch (y-axis rotation)
            t2 = +2.0 * (w * y - z * x)
            t2 = np.clip(t2, -1.0, 1.0)
            pitch = np.arcsin(t2)

            # yaw (z-axis rotation)
            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw = np.arctan2(t3, t4)

            return np.array([roll, pitch, yaw])
 
        euler_or = quat_to_euler(obs_quat)

        faceup = [
            np.array([1.57, 1.55]), 
            np.array([0, 0]),  
            np.array([-1.57, -1.55]),
            np.array([3.14, 0.08]),
            np.array([1.56, 0.073]),
            np.array([-1.66, -0.06])
        ]

        def find_faceup(euler_or):
            delta = 0.4
            roll, pitch = euler_or[:2]
            for i, ref in enumerate(faceup):
                if (abs(roll - ref[0]) <= delta) and (abs(pitch - ref[1]) <= delta):
                    return i
            return -1
        up_face = find_faceup(euler_or)

        def find_face_or(yaw):
            face_or = np.array([0, 0.78, 1.57, 2.18, 3.1416, -2.18, -1.57, -0.78])
            delta = 0.3
            for i, ref in enumerate(face_or):
                if abs(yaw-ref) <= delta:
                    return i
            return -1
        
        face_or = find_face_or(euler_or[2])

        def point_to_block(x, y):
            # Rectangle boundaries
            x_min, x_max = 0.302, 0.372
            y_min, y_max = -0.03, 0.02
            
            # Grid size
            nx, ny = 3, 3
            dx = (x_max - x_min) / nx
            dy = (y_max - y_min) / ny
            
            # Check if inside rectangle
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                return -1  # outside
            
            # Column index (0..2)
            col = int((x - x_min) // dx)
            if col == nx: col = nx - 1  # edge correction
            
            # Row index (0..2) — note y axis goes bottom→top
            row = int((y - y_min) // dy)
            if row == ny: row = ny - 1  # edge correction
            
            # Block ID (row-major order: top row = 0..2)
            block_id = (ny - 1 - row) * nx + col
            return block_id

        pos_affordance = point_to_block(obj_pos[0],obj_pos[1])

        return pos_affordance, up_face, face_or


    def quaternion_to_euler_z(self, quat):
        w, x, y, z = quat
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)  # yaw

    def _get_obs(self):
        joint_obs = []
        for name in self.actuator_names:
            joint_name = name.replace("rh_A_", "rh_")
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_index = self.model.jnt_qposadr[joint_id]
            joint_obs.append(self.data.qpos[qpos_index])

        # Stick pose
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        qpos_addr = self.model.jnt_qposadr[joint_id]
        stick_pose = self.data.qpos[qpos_addr: qpos_addr + 7]

        # Yaw
        quat = stick_pose[3:7]
        z_rot = self.quaternion_to_euler_z(quat)

        # Stick angular velocity (xyz)
        qvel_addr = self.model.jnt_dofadr[joint_id]
        #stick_ang_vel = self.data.qvel[qvel_addr + 3: qvel_addr + 6]
        joint_obs = np.array(joint_obs, dtype=np.float32)  # ensure ndarray
        object_shape = self.prev_action - joint_obs
        #print("object_shape", object_shape)
        obs = np.concatenate([joint_obs, stick_pose]).astype(np.float32)
        return {
            'observation': obs,
            'prev_action': self.prev_action,
            'Obj_shape': object_shape
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        # hand default pose
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = self.grasp_pose_hand[i]
        for _ in range(50): mujoco.mj_step(self.model, self.data)
        for i, aid in enumerate(self.df_actuator_ids):
            self.data.ctrl[aid] = self.df_digit_poses[i]
        for _ in range(50): mujoco.mj_step(self.model, self.data)
        # randomize object orientation (same as your code) ...
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        qpos_addr = self.model.jnt_qposadr[jid]


        def random_quaternion():
            u1, u2, u3 = np.random.rand(3)
            x = np.sqrt(1 - u1) * np.sin(2*np.pi*u2)
            y = np.sqrt(1 - u1) * np.cos(2*np.pi*u2)
            z = np.sqrt(u1)     * np.sin(2*np.pi*u3)
            w = np.sqrt(u1)     * np.cos(2*np.pi*u3)
            # return as [w, x, y, z]
            return np.array([w, x, y, z], dtype=np.float32)
        


        pos = self.rot_qpos_stick[:3]
        quat = random_quaternion()
        self.data.qpos[qpos_addr:qpos_addr+7] = np.concatenate([pos, quat])
        #self.data.qpos[qpos_addr: qpos_addr + 7] = self.rot_qpos_stick
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, primitive_idx):
        k = int(primitive_idx)
        seq = self.P[k]                   # (L, n) with values in {-1,0,+1}
        total_r, terminated, truncated = 0.0, False, False
        info, last_obs = {}, None

        for row in seq:
            # map {-1,0,1} -> {lo, mid, hi} for each actuator
            ctrl = np.empty(len(self.actuator_ids), dtype=np.float32)
            self.prev_action = np.empty(len(self.actuator_ids), dtype=np.float32)
            #self.prev_action[:] = ctrl
            for i, aid in enumerate(self.actuator_ids):
                lo, hi = self.model.actuator_ctrlrange[aid]
                mid = 0.5 * (lo + hi)
                v = int(row[i])
                if   v == -1: ctrl[i] = lo
                elif v ==  0: ctrl[i] = mid
                else:         ctrl[i] = hi
                self.prev_action[i] = ctrl[i]

            self.data.ctrl[self.actuator_ids] = ctrl
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
                if self.render_enabled: self.render()

            self.current_step += 1
            last_obs = self._get_obs()
            obs_vec = last_obs["observation"]
            obj_state = obs_vec[-7:]
            obj_pos, obj_quat = obj_state[:3], obj_state[3:]
            r, succ = self.compute_reward(obj_pos, obj_quat)
            total_r += float(r)
            dropped = (obj_pos[2] < 0.027)
            terminated = bool(succ) or dropped

            # if bool(succ):
            #     #print("------success---------")
                
            #     time.sleep(5)

            truncated  = self.current_step >= self.max_steps

            info = {
                "is_success": int(bool(succ)),                 # 1 only for clean success
                "event": "success" if succ else ("drop" if dropped else "none"),
                "terminated": terminated,
                "truncated": truncated,
            }

            # if terminated or truncated:
            #     return last_obs, total_r, terminated, truncated, info
            #     break
                

            # after the for-loop:
        #print("total_reward", total_r)
        # un comment sleep to visualize during evaluation only 
        # if bool(succ):
        #     time.sleep(5)
        return last_obs, total_r, terminated, truncated, info

    def render(self):
        if not self.render_enabled: return
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(),
            mujoco.MjvPerturb(), self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        mujoco.mjr_render(mujoco.MjrRect(0, 0, 800, 600), self.scene, self.context)
        glfw.swap_buffers(self.window); glfw.poll_events()

    def close(self):
        if self.window is not None:
            glfw.terminate()
    def _step_for(self, n):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self.render()

    def _quat_conj(self, q):
        w, x, y, z = q
        return np.array([w, -x, -y, -z], dtype=np.float32)

    def _quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.float32)

    def _quat_rotate(self, q, v):
        qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
        return self._quat_mul(self._quat_mul(q, qv), self._quat_conj(q))[1:]

    def compute_reward(self, obj_pos, obj_quat):
        rew = 0
        pos_aff, up_face, face_or = self.object_state(obj_pos, obj_quat)

        drop = (obj_pos[2] < 0.027)
        success = (up_face == 4) and (pos_aff in (0,1,2,3,4,5,7,8)) and face_or == 7

        if self.prev_face == up_face or self.prev_face_or == face_or:
            self.face_penalty = self.face_penalty - 0.08
            #print("Face_penalty", self.face_penalty)

        else:
            self.face_penalty = 0


        pf_r = (
            -0.3  if pos_aff in (0,1,3) else
            -0.2  if pos_aff in (2,5)   else
            -0.35 if pos_aff in (6,7,8) else
            -0.5  if pos_aff == -1      else
            0.0
        )

        


        if drop:
            rew = pf_r - 5
            succ = 0   # drop overrides success
        if success and not drop:
            rew = pf_r + 100
            succ = 1
        else:
            succ = 0

        self.prev_face = up_face 
        self.prev_face_or = face_or


        return float(rew+self.face_penalty), succ



class PrimitiveUsageCallback(BaseCallback):
    def __init__(self, K: int, log_prefix="primitives", verbose=0):
        super().__init__(verbose)
        self.K = K
        self.log_prefix = log_prefix

    def _on_step(self) -> bool:
        # required by BaseCallback; return True to keep training
        return True

    def _on_rollout_end(self) -> bool:
        acts = self.model.rollout_buffer.actions
        if hasattr(acts, "cpu"):
            acts = acts.cpu().numpy()
        acts = acts.reshape(-1).astype(int)
        hist = np.bincount(acts, minlength=self.K)
        total = hist.sum()
        probs = hist / (total + 1e-12)
        ent = -(probs * np.log(probs + 1e-12)).sum()

        for i, c in enumerate(hist):
            self.logger.record(f"{self.log_prefix}/use_{i}", float(c))
        self.logger.record(f"{self.log_prefix}/entropy", float(ent))
        return True



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

p21 = [
    np.array([-1, -1, 0, 0, -1, -1, 0, -1, 0, 0]),        ####When object is at the far back end of the palm, thumb push the object back in from behind - CORRECTIVE PRIMITIVE
    np.array([-1, 0, 0, -1, -1, -1, 0, -1, 0, -1]),
    np.array([-1, 1, -1, -1, -1, -1, 0, -1, 0, -1]),
    np.array([0, 1, -1, -1, -1, 0, 0, -1, 0, -1]),
    np.array([0, 1, -1, -1, -1, 0, 0, -1, 0, 1]),
    np.array([-1, 1, -1, -1, -1, 0, 0, -1, 0, 1]),
    np.array([-1, 0, -1, -1, -1, 0, 0, -1, 0, 1]),
    np.array([-1, -1, -1, -1, -1, 0, 0, -1, 0, 1]),
    np.array([0, -1, -1, -1, -1, 0, 0, -1, 0, 0]),
]

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

# P = [
#     to_primitive(pn, n_act)
# ]



# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# env = DummyVecEnv([lambda: PrimitiveOnlyEnv("scene_right_op.xml", primitives=P, render=False, frame_skip=100)])
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

import os
os.environ.setdefault("MUJOCO_GL", "egl")  # headless in workers

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import torch

XML = os.path.abspath("scene_right_op.xml")  # avoid CWD issues

def make_env():
    # top-level function (picklable) that returns an env when called
    def _init():
        return Monitor(
            PrimitiveOnlyEnv(XML, primitives=P, render=False, frame_skip=100),
            info_keywords=("is_success",)
        )
    return _init

policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
)

if __name__ == "__main__":
    # On Linux you can force 'fork' if your Python defaults to 'spawn'/'forkserver'
    import multiprocessing as mp
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass  # already set

    n_envs = 8
    venv = SubprocVecEnv([make_env() for _ in range(n_envs)])
    #venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    venv = VecNormalize(
    venv,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    norm_obs_keys=["observation", "prev_action", "Obj_shape"],  # or just the ones you want\
    )

    model = PPO(
        "MultiInputPolicy",
        venv,
        n_steps=512,          # => buffer 512 * 8 = 4096
        batch_size=1024,      # divides buffer cleanly
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        device="cuda",
        verbose=1,
    )

    callback = PrimitiveUsageCallback(K=len(P))
    model.learn(total_timesteps=5560000, callback=callback)
    venv.save("vecnormalize_all_or.pkl")
    model.save("ppo_primitives_only_all_or")