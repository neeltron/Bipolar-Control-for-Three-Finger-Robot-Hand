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

# =========================
# Your base env (ternary)
# =========================
class StickSpinEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, model_path, max_steps=1000, render=True, frame_skip=20):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.current_step = 0

        # --- Rendering setup (optional) ---
        self.render_enabled = render
        self.frame_skip = frame_skip
        if self.render_enabled:
            glfw.init()
            self.window = glfw.create_window(800, 600, "MuJoCo Viewer", None, None)
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)

            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = self.model.stat.center
            self.cam.distance = 0.7
            self.cam.azimuth = 130
            self.cam.elevation = -50

            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        else:
            self.window = None

        self.actuator_names = [
            "rh_A_THJ5", "rh_A_THJ4", "rh_A_THJ1",
            "rh_A_FFJ4", "rh_A_FFJ3", "rh_A_FFJ0",
            "rh_A_RFJ4", "rh_A_RFJ3", "rh_A_RFJ0", "rh_A_THJ2"
        ]
        self.df_digits = ["rh_A_LFJ4", "rh_A_LFJ3", "rh_A_LFJ0"]

        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                             for name in self.actuator_names]
        self.df_actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                                for name in self.df_digits]
        assert all(aid != -1 for aid in self.actuator_ids), "Bad actuator name in self.actuator_names"

        # Initial ctrl targets (tune as needed)
        self.grasp_pose_hand = np.array([0.0]*len(self.actuator_ids), dtype=np.float32)
        self.df_digit_poses = np.array([0.0, 2.0, 2.42], dtype=np.float32)

        # --- ACTION SPACE: ternary per actuator -> {lo, mid, hi} ---
        n = len(self.actuator_ids)
        self.action_space = spaces.MultiDiscrete([3] * n)  # 0,1,2 per actuator

        # --- OBSERVATION SPACE ---
        self.n_finger_joints = len(self.actuator_names)
        self.obs_dim = self.n_finger_joints + 7 # joints + freejoint pose + ang vel
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
            'prev_action': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_finger_joints,), dtype=np.float32),
            'Obj_shape': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_finger_joints,), dtype=np.float32)
        })
        self.prev_action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Initial stick pose [x y z qw qx qy qz]
        #self.rot_qpos_stick = np.array([0.36878775, -0.02474563,  0.05997783, 0.99318724,  0.00353254,  0.06888206, -0.09392493], dtype=np.float32) #0.32
        #self.rot_qpos_stick = np.array([0.36878775, -0.0243089,  0.05475683, -0.699839,   -0.013892,  0.713185, 0.035728], dtype=np.float32)
        self.rot_qpos_stick = np.array([0.35366342, -0.02474563,  0.05997783, 0.99318724,  0.00353254,  0.06888206, -0.09392493], dtype=np.float32)

    # --------- Helpers ---------
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

        joint_obs = np.asarray(joint_obs, dtype=np.float32)   # < adding this to convert from pythonlist to numpy array


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
        object_shape = self.prev_action - joint_obs
        #print("object_shape", object_shape)
        obs = np.concatenate([joint_obs, stick_pose]).astype(np.float32)
        return {
            'observation': obs,
            'prev_action': self.prev_action,
            'Obj_shape': object_shape
        }

    # --------- Gym API ---------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        # Set initial hand pose via actuators and settle
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = self.grasp_pose_hand[i]
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # default finger pose
        for i, aid in enumerate(self.df_actuator_ids):
            self.data.ctrl[aid] = self.df_digit_poses[i]
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Set stick initial pose
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        qpos_addr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qpos_addr: qpos_addr + 7] = self.rot_qpos_stick

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # 0/1/2 -> lo/mid/hi

        ctrl = np.empty(len(self.actuator_ids), dtype=np.float32)
        self.prev_action = np.empty(len(self.actuator_ids), dtype=np.float32)
        for i, aid in enumerate(self.actuator_ids):
            lo, hi = self.model.actuator_ctrlrange[aid]
            mid = 0.5 * (lo + hi)
            choice = int(action[i])  # 0, 1, or 2
            if choice == 0:
                ctrl[i] = lo
                self.prev_action[i] = lo
            elif choice == 1:
                ctrl[i] = mid
                self.prev_action[i] = mid
            else:
                ctrl[i] = hi
                self.prev_action[i] = hi
            

            #ctrl[i] = lo if choice == 0 else (mid if choice == 1 else hi)

        # Apply control
        self.data.ctrl[self.actuator_ids] = ctrl

        # Physics stepping
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self.render_enabled:
                    self.render()

        self.current_step += 1
        obs = self._get_obs()
        #reward = float(self.compute_reward(None, None, {}))
        reward, error_scaler = self.compute_reward(None, None, {})

        # success check
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        qvel_addr = self.model.jnt_dofadr[jid]
        wz = float(self.data.qvel[qvel_addr+5])
        succ = error_scaler<0.25


        # Drop condition
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        qpos_addr = self.model.jnt_qposadr[joint_id]
        z_pos = float(self.data.qpos[qpos_addr + 2])
        x_pos = float(self.data.qpos[qpos_addr])
        truncated = self.current_step >= self.max_steps
        
        if z_pos < 0.03 or x_pos < 0.27:
            #reward -= 10.0
            truncated = True

        info = {"is_success": succ}
        terminated = False
        #print("Reward", reward)
        #print(f"error {error_scaler:.3f}, reward {reward:.3f}, step_count {self.current_step:.3f}")
        return obs, reward, terminated, truncated, info

    # --------- Rendering ---------
    def render(self):
        if not self.render_enabled:
            return
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(),
            mujoco.MjvPerturb(), self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        mujoco.mjr_render(mujoco.MjrRect(0, 0, 800, 600), self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

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

    def compute_reward(self, achieved_goal, desired_goal, info):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        qpos_addr = self.model.jnt_qposadr[jid]
        qvel_addr = self.model.jnt_dofadr[jid]

        quat = self.data.qpos[qpos_addr+3:qpos_addr+7]
        wx, wy, wz = self.data.qvel[qvel_addr+3:qvel_addr+6]

        z_local_world = self._quat_rotate(quat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        cos_tilt = np.clip(z_local_world[2], -1.0, 1.0)
        tilt_pen = 1.0 - cos_tilt

        wobble_pen = wx*wx + wy*wy

        wz_target = 5.0
        spin_term = - (wz - wz_target) ** 2
        #print(f"rotation {wz:.3f}, spin_term {spin_term:.3f}, z_tilt {tilt_pen:.3f}")

        vx, vy, vz_lin = self.data.qvel[qvel_addr:qvel_addr+3]
        lin_pen = vx*vx + vy*vy + vz_lin*vz_lin

        z_pos = float(self.data.qpos[qpos_addr + 2])
        drop_pen = 10.0 if z_pos < 0.03 else 0.0

        w_spin   = 0 #1.0
        w_wobble = 0 #0.05
        w_tilt   = 0 #0.5
        w_lin    = 0.01
        w_drop   = 1.0
        w_or = 1


        def quaternion_distance(q1, q2):
            q1 = q1 / np.linalg.norm(q1)
            q2 = q2 / np.linalg.norm(q2)
            dot_product = np.abs(np.dot(q1, q2))
            dot_product = np.clip(dot_product, -1.0, 1.0)  # For numerical stability
            angle = 2 * np.arccos(dot_product)
            return angle  # in radians


        target = np.array([-0.0356975, -0.70663227, -0.02587772, 0.70620596])
        error_scalar = quaternion_distance(quat, target)
        #print("Error", error_scalar)
        #print(f"rotation {wz:.3f}, spin_term {spin_term:.3f}, z_tilt {tilt_pen:.3f}")
        reward = (
            w_spin * spin_term
            - w_wobble * wobble_pen
            - w_tilt * tilt_pen
            - w_lin * lin_pen
            - w_drop * drop_pen - w_or*error_scalar
        )
        
        return float(reward), float(error_scalar)

class RealTimeRenderCallback(BaseCallback):
    def __init__(self, env, render_freq=1, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.render_freq = render_freq
        self.counter = 0
    def _on_step(self) -> bool:
        self.counter += 1
        if self.counter % self.render_freq == 0:
            self.env.render()
        return True


# ===========================================
# Primitive pool and micro-wrapper
# ===========================================
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

class PrimitiveMacroWrapper(gym.Wrapper):
    """
    Exposes Discrete(K) where K = number of primitives.
    On step(k): unroll that primitive's atomic rows inside, summing rewards,
    returning the LAST observation. Ends early if the episode ends mid-primitive.
    """
    def __init__(self, env: gym.Env, primitives):
        super().__init__(env)
        self.primitives = [np.asarray(p, dtype=np.int8) for p in primitives]
        self.K = len(self.primitives)
        assert self.K >= 1, "Need at least one primitive"
        self.action_space = spaces.Discrete(self.K)
        self.observation_space = env.observation_space

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, primitive_idx):
        k = int(primitive_idx)
        assert 0 <= k < self.K
        seq = self.primitives[k]      # (L, n) in {-1,0,+1}
        total_r = 0.0
        terminated = False
        truncated = False
        info = {}
        last_obs = None

        for row in seq:
            # map flags {-1,0,1,} -> actions {0,1,2}
            atomic = (row + 1).astype(np.int64)  # {-1,0,+1} -> {0,1,2}
            last_obs, r, term, trunc, info = self.env.step(atomic)
            total_r += float(r)
            if term or trunc:
                terminated = term
                truncated = trunc
                break

        info = dict(info or {})
        info["primitive_index"] = k
        info["primitive_len"] = int(seq.shape[0])
        return last_obs, total_r, terminated, truncated, info


# =========================
# Main / Training
# =========================
if __name__ == "__main__":
    # Base env
    env_base = StickSpinEnv(r"C:\CARL_Summer_Research\Bipolar-Control-for-Three-Finger-Robot-Hand-main (1)\Bipolar-Control-for-Three-Finger-Robot-Hand-main\Flexibletip_shadowhand\scene_right_m.xml", render=True, frame_skip=90)
    n_act = len(env_base.actuator_ids)

    # --------------------------
    # Hard-coded primitives
    # --------------------------



    p1 = [
        np.array([ 0, -1, -1, 0, -1, -1, 0, -1, -1, 0]),          ### only caging action, CAGING PRIMITIVE
        np.array([ 0, -1, -1, 0, -1, -1, 0, 0, 0, 0]),
        np.array([ 0, -1, -1, 0, -1, -1, 0, 0, 0, 0]),
        np.array([ 0, -1, -1, -1, -1, 0, 0, 0, 0, 0]),
        np.array([ 0, -1, -1, -1, -1, 0, 0, 0, 0, 0]),
        np.array([ 0, -1, 0, -1, -1, 0, 0, 0, 0, 0]),
    ]

    p2 = [
        np.array([ 0, -1, -1, 0, -1, -1, 0, -1, -1, 0]),       ### rotating action with caging motion, CAGING + ROTATION PRIMITIVE 
        np.array([ 0, -1, -1, 0, -1, -1, 0, 0, 0, 0]),
        np.array([ 0, -1, -1, 0, -1, -1, 0, 0, 0, 0]),
        np.array([ 0, -1, -1, -1, -1, 0, 0, 0, 0, 0]),
        np.array([ 0, -1, -1, -1, -1, 0, 0, 0, 0, 0]),
        np.array([ 0, -1, 0, -1, -1, 0, 0, 0, 0, 0]),
        np.array([ 0, -1, 0, -1, -1, 0, 0, 0, 0, 1]), ##
        np.array([ 0, 0, 0, -1, -1, 0, 0, 0, 0, 1]),
        np.array([ 0, 0, 0, -1, -1, 0, 0, 0, 0, 1]),
        np.array([ 0, 0, 0, -1, -1, 0, 0, -1, 0, 1]),
    ]

    p3 = [
        np.array([ 0, 0, 0, -1, -1, 0, 0, -1, 0, 1]),       ####releasing object onto the palm - RELEASE PRIMITIVE
        np.array([ 0, 0, 0, -1, -1, -1, 0, -1, 0, 1]),
        np.array([ 0, 0, -1, -1, -1, -1, 0, -1, 0, 1]),
        np.array([ 0, 0, -1, -1, -1, -1, 0, -1, 0, 1]),
        np.array([ 0, 0, -1, -1, -1, -1, 0, -1, 1, 1]),
    ]

    p4 = [
        np.array([ 0, 0, -1, -1, -1, -1, 0, -1, 1, 1]),      ####Ring finger slightly reorients the object while other two cages - REORIENTATION PRIMITIVE
        np.array([ 0, 0, -1, -1, -1, 0, 0, -1, 1, 1]),
        np.array([ 0, 0, -1, -1, -1, 0, 0, -1, 1, 1]),
        np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 1, 1]),
        np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 1, 1]),####
        np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 1, 1]),
        np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 1]),
        np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 1]),
        np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 0]),
        np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 0]),
    ]

    # p5 = [
    #     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 1, 1]),
    #     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 1]),
    #     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 1]),
    #     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 0]),
    #     np.array([ 0, 0, -1, -1, -1, 0, 0, 0, 0, 0]),
    # ]

    p6 = [
        np.array([ 0, 0, -1, -1, -1, 0, 0, -1, 0, 0]),       ####Thumb tries to pull back object from one corner - CORRECTIVE PRIMITIVE
        np.array([ 0, 1, 0, -1, -1, 0, 0, -1, 0, 1]),
        np.array([ 0, 0, 0, -1, -1, 0, 0, -1, 0, 1]),
        np.array([ 0, 0, -1, -1, -1, 0, 0, -1, 0, -1]),
        np.array([ -1, 0, 0, -1, -1, 0, 0, -1, 0, 0]),
        np.array([ -1, -1, 0, -1, -1, 0, 0, -1, 0, 0]),
    ]

    # p7 = [
    #     np.array([ 0, 0, 0, -1, -1, 0, 0, -1, 0, 1]),
    #     np.array([ 0, 0, -1, -1, -1, 0, 0, -1, 0, 0]),
    #     np.array([ 0, -1, -1, -1, -1, 0, 0, -1, 0, 0]),
    #     np.array([ 0, -1, -1, -1, -1, 0, 0, -1, 0, 1]),
    # ]

    p8 = [
        np.array([ 0, -1, -1, -1, -1, 0, 0, 0, 0, 1]),         ####When object is at the corner, thumb and index pushes it back in  - CORRECTIVE PRIMITIVE
        np.array([ 0, -1, 0, -1, -1, 1, 0, 0, 0, 0]),
        np.array([ 0, -1, 0, -1, -1, 0, 0, 0, 0, 0]),
        np.array([ 0, -1, 0, 0, -1, 0, 0, 0, 0, 0]),
        np.array([ 0, -1, 0, 0, -1, 1, 0, 0, 0, 0]),
        np.array([ 0, -1, 0, -1, -1, 0, 0, 0, 0, 0]),
    ]

    p9 = [
        np.array([0, -1, -1, 0, -1, -1, 0, -1, -1, 0]),        #####When object is tilted on the palm, might perform Y axis rotation - ROTATION PRIMITIVE
        np.array([0, 0, -1, 0, -1, -1, 0, -1, -1, 0]),
        np.array([-1, 0, -1, 0, -1, -1, 0, -1, -1, 0]),
        np.array([-1, 1, -1, 0, -1, -1, 0, -1, -1, 0]),
        np.array([-1, 1, 0, 0, 0, 0, 0, -1, 0, 0]),
        np.array([-1, 1, -1, 0, 0, 1, 0, -1, 1, 0]),
        np.array([-1, 1, 0, 0, 1, 0, 0, 0, 1, 0]),
        np.array([-1, 1, 0, 0, 1, -1, 0, 0, 0, 0]),
        np.array([-1, 1, 0, 0, 1, -1, 0, 0, 0, 0]),
        np.array([-1, 0, 0, 0, 1, -1, 0, 0, 0, 0]),
    ]

    p10 = [
        np.array([-1, -1, 0, 0, 1, -1, 0, 0, 0, 0]),       ####When object is at the far end of a corner, thumb pulls it back in  - CORRECTIVE PRIMITIVE
        np.array([-1, 0, 0, 0, 1, -1, 0, -1, 0, 0]),
        np.array([-1, 1, 0, 0, 0, -1, 0, -1, 0, 0]),
        np.array([0, 1, 0, 0, -1, -1, 0, -1, 0, 0]),
        np.array([1, 1, -1, -1, -1, 0, 0, -1, 0, 0]),
        np.array([0, 1, -1, -1, -1, 0, 0, -1, 0, 0]),
        np.array([-1, 1, -1, -1, -1, 0, 0, -1, 0, 0]),
        np.array([-1, 0, -1, -1, 0, 0, 0, -1, 0, 1]),
        np.array([-1, -1,-1, -1, 0, 0, 0, -1, 0, 1]),
        np.array([0, -1,-1, -1, -1, 0, 0, -1, 0, 1]),
    ]

    p11 = [
        np.array([ 0, -1, -1, -1, 0, 0, 0, -1, 0, 1]),       ####Reorients the object from the ring side - REORIENTATION PRIMITIVE
        np.array([ 0, -1, -1, -1, 0, 0, 0, -1, 1, 1]),
        np.array([ 0, -1, -1, -1, 0, 0, 0, 0, 1, 1]),
        np.array([ 0, -1, -1, -1, 0, 0, 0, 0, 1, 1]),
        #np.array([ 0, -1, -1, -1, 0, 0, 0, 0, 1, 1]),
        #np.array([ 0, -1, -1, -1, 0, 0, 0, -1, 1, 1]),
    ]

    p12 = [
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

    p14 = [
        np.array([0, -1, -1, 0, -1, 0, 0, -1, 0, 0]),        ####When object is laterally disbalanced and about to fall off from the edge of the palm, thumb pushes it back in - CORRECTIVE PRIMITIVE
        np.array([0, -1, -1, -1, -1, 0, -1, -1, 0, 1]),
        np.array([0, -1, -1, -1, -1, 0, -1, -1, 0, 0]),
        np.array([0, -1,-1, -1, -1, 0, -1, -1, 0, 1]),
        np.array([0, -1, 0, -1, -1, 0, -1, -1, 0, 1]),
    ]

    p15 = [
        np.array([ 0, -1, -1, 0, -1, -1, -1, -1, 0, 0]),      ####When object at the corner of index finger, index finger pushes the object in - CORRECTIVE PRIMITIVE
        np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 0, 0]),
        np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 0, 0]),
        np.array([ 0, -1, -1, -1, -1, 1, -1, -1, 0, 0]),
        np.array([ 0, -1, -1, -1, -1, 1, -1, -1, 0, 0]),
    ]

    p16 = [
        np.array([ 0, -1, -1, -1, -1, 0, 0, -1, -1, 0]),      ####When object at the corner of ring finger, ring finger pushes the object in - CORRECTIVE PRIMITIVE
        np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 0, 0]),
        np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 0, 0]),
        np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 1, 0]),
        np.array([ 0, -1, -1, -1, -1, 0, -1, -1, 1, 0]),
    ]

    p17 = [
        np.array([ 0, -1, -1, 0, -1, -1, 0, -1, -1, 0]),     ###rotates object front to back Y axis (according to mujoco) - ROTATORY PRIMITIVE
        np.array([ 0, -1, -1, 0, 0, -1, 0, 0, -1, 0]),
        np.array([ 0, -1, -1, 0, 0, 0, 0, 0, 0, 0]),
        np.array([ 0, -1, -1, 0, 0, 0, 0, 0, 0, 0]),
        np.array([ 0, -1, -1, 0, 0, 0, 0, 0, 0, -1]),
        np.array([ 0, 0, -1, 0, 0, 0, 0, 0, 0, -1]),
        np.array([ 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]),
    ]

    # Convert to validated (L, n) arrays
    P = [
        to_primitive(p1, n_act),
        to_primitive(p2, n_act),
        to_primitive(p3, n_act),
        to_primitive(p4, n_act),
        #to_primitive(p5, n_act),
        to_primitive(p6, n_act),
        #to_primitive(p7, n_act),
        to_primitive(p8, n_act),
        to_primitive(p9, n_act),
        to_primitive(p10, n_act),
        to_primitive(p11, n_act),
        to_primitive(p12, n_act),
        #to_primitive(p13, n_act),
        to_primitive(p14, n_act),
        to_primitive(p15, n_act),
        to_primitive(p16, n_act),
        to_primitive(p17, n_act),
    ]

    # P = [
    #     to_primitive(p4, n_act),
    # ]


    print(f"[info] Loaded {len(P)} primitives with actuator width n={n_act}")
    for i, prim in enumerate(P):
        print(f"  - p{i+1}: shape={prim.shape}, values={sorted(np.unique(prim).tolist())}")

    # Wrap env
    env = PrimitiveMacroWrapper(env_base, P)
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,                  # ReLU is common for robotics
        net_arch=dict(pi=[256, 256, 256],             # Actor: 3 hidden layers of 256
                    vf=[256, 256, 256]),           # Critic: 3 hidden layers of 256
        # optional: orthogonal_init=False,            # sometimes helps stability with ReLU
    )

    # PPO over Discrete(K) actions + Dict observations
    model = PPO(
        policy="MultiInputPolicy",     # works with Dict observations (CombinedExtractor)
        env=env,
        n_steps=2048,                  # longer rollouts usually help complex hands
        batch_size=512,                # keep batch_size <= n_steps * n_envs
        n_epochs=10,                   # 5–15 is typical; 10 is a good start
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,            # try 1e-4..5e-4 if unstable
        clip_range=0.2,
        ent_coef=0.0,                  # try 0.005–0.02 to encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="auto",
    )

    #callback = RealTimeRenderCallback(env, render_freq=3)
    model.learn(total_timesteps= 512000)  # , callback=callback
    model.save(r"C:\CARL_Summer_Research\Bipolar-Control-for-Three-Finger-Robot-Hand-main (1)\Bipolar-Control-for-Three-Finger-Robot-Hand-main\Flexibletip_shadowhand\Runs\onPalm_primitives")
    env.close()
