# eval_primitives.py
import os, sys, time, collections, importlib.util
import numpy as np

# === EDIT THESE THREE PATHS AS NEEDED ===
ENV_PY_PATH   = os.path.abspath("all_primitivesv2_reduced_success_reward.py")  # your file with PrimitiveOnlyEnv, P, XML
MODEL_PATH    = os.path.abspath("all_primitives_break_success.zip")
#MODEL_PATH    = os.path.abspath("best_model.zip")
VECNORM_PATH  = os.path.abspath("all_primitives_break_success.pkl")

# === RUN SETTINGS ===
NUM_EPISODES      = 1000
DETERMINISTIC     = True
RENDER            = False      # if False, uses EGL headless
FRAME_HOLD_SEC    = 1.5            # keep last frame visible when episode ends (if rendering)

# ----------------------------------------

if RENDER:
    os.environ["MUJOCO_GL"] = "glfw"
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

# --- Dynamically import the env module from a filepath ---
def import_module_from_path(path, module_name="user_env_module"):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod = import_module_from_path(ENV_PY_PATH)

# Required symbols from your file
try:
    PrimitiveOnlyEnv = mod.PrimitiveOnlyEnv
    P = mod.P           # list of primitives
except AttributeError as e:
    raise RuntimeError("Your env file must define `PrimitiveOnlyEnv` and `P` at module level.") from e

# XML path: Prefer mod.XML if provided; else fallback to scene_right_op.xml next to env file
if hasattr(mod, "XML"):
    XML_PATH = mod.XML
else:
    XML_PATH = os.path.join(os.path.dirname(ENV_PY_PATH), "scene_right_op.xml")
XML_PATH = os.path.abspath(XML_PATH)

# --- Build a single env and wrap with DummyVecEnv + Monitor + VecNormalize ---
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

def make_eval_env():
    env = PrimitiveOnlyEnv(XML_PATH, primitives=P, render=RENDER, frame_skip=100)
    # Expose success flag in Monitor info for logging convenience
    env = Monitor(env, info_keywords=("is_success",))
    return env

# Important: for SB3 we need a VecEnv even for a single env
venv = DummyVecEnv([make_eval_env])

# Load VecNormalize stats into this venv
if os.path.isfile(VECNORM_PATH):
    venv = VecNormalize.load(VECNORM_PATH, venv)
else:
    print(f"[WARN] VecNormalize file not found at {VECNORM_PATH}. "
          f"Proceeding without loaded normalization stats.")
# Evaluation mode: do not update stats; do not normalize rewards
venv.training = False
venv.norm_reward = False

# Load the PPO policy with the vec-normalized env attached
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model = PPO.load(MODEL_PATH, device="auto", env=venv, print_system_info=True)

def hold_final_frame(seconds=1.0, fps=60):
    if not RENDER:
        return
    t_end = time.time() + seconds
    dt = 1.0 / fps
    # Pump the GL loop via env.render() calls
    while time.time() < t_end:
        # Dummy step to trigger render loop—our env exposes a render() method
        try:
            venv.envs[0].render()
        except Exception:
            break
        time.sleep(dt)

# --- Evaluation loop ---
primitive_hist_all = collections.Counter()
episode_results = []

obs = venv.reset()
for ep in range(NUM_EPISODES):
    done = False
    ep_reward = 0.0
    ep_len = 0
    chosen_prims = []

    while not done:
        # Predict one primitive ID per Env step (which internally unrolls its sequence)
        action, _ = model.predict(obs, deterministic=False)
        # `action` is shape (n_envs,) for DummyVecEnv; extract scalar
        chosen_prim = int(action[0])
        chosen_prims.append(chosen_prim)

        obs, reward, dones, infos = venv.step(action)
        ep_reward += float(reward[0])
        ep_len += 1
        done = bool(dones[0])

        # (Optional) If you want to see intermediate rendering in real time:
        if RENDER:
            venv.envs[0].render()

    # final info from Monitor
    info = infos[0]
    is_success = int(info.get("is_success", 0))

    # import time
    # if is_success == 1:
    #     time.sleep(5)

    # stats
    primitive_hist_all.update(chosen_prims)
    episode_results.append({
        "episode": ep + 1,
        "return": ep_reward,
        "length": ep_len,
        "success": is_success,
        "primitives": chosen_prims,
    })

    print(f"\nEpisode {ep+1}: return={ep_reward:.3f}  len={ep_len}  success={is_success}")
    print(f"  primitives used: {chosen_prims}")

    hold_final_frame(FRAME_HOLD_SEC)

# --- Summary ---
print("\n=== Evaluation Summary ===")
succ_rate = np.mean([r["success"] for r in episode_results]) if episode_results else 0.0
avg_ret   = np.mean([r["return"]  for r in episode_results]) if episode_results else 0.0
avg_len   = np.mean([r["length"]  for r in episode_results]) if episode_results else 0.0

print(f"Episodes: {len(episode_results)}")
print(f"Success rate: {succ_rate:.2f}")
print(f"Avg return:   {avg_ret:.3f}")
print(f"Avg length:   {avg_len:.2f}")

print("\nPrimitive usage (all episodes):")
for k, v in primitive_hist_all.most_common():
    print(f"  primitive {k}: {v} times")
