"""Where does the BC policy diverge from the expert?"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import FrankaCubeLift
from collect_demos import observe, GRIPPER_SCALE
from controller import ScriptedGrasp
from train_bc import MLPPolicy
DEC = 10

ck = torch.load("../models/bc_policy.pt", weights_only=False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
pol = MLPPolicy(ck["obs_dim"], ck["act_dim"], hidden=ck["hidden"]).to(dev)
pol.load_state_dict(ck["state_dict"]); pol.eval()
om, os_, am, as_ = ck["norm"]

env = FrankaCubeLift()
xy = np.array([0.55, 0.0])

# 1) open-loop action agreement on an EXPERT trajectory (no compounding)
env.reset(cube_xy=xy); exp = ScriptedGrasp(env)
errs = []
for _ in range(700):
    o = observe(env); q0 = env.arm_qpos
    at, g = exp.act()
    a_exp = np.concatenate([at - q0, [g / GRIPPER_SCALE]])
    with torch.no_grad():
        x = torch.as_tensor((o-om)/os_, dtype=torch.float32, device=dev).unsqueeze(0)
        a_bc = pol(x).squeeze(0).cpu().numpy()*as_ + am
    errs.append(np.abs(a_exp - a_bc))
    env.step(at, g, n_substeps=DEC)
    if exp.done: break
errs = np.array(errs)
print(f"on-expert-trajectory action error (mean abs):")
print(f"  dq   {np.round(errs[:, :7].mean(0), 5)}")
print(f"  grip {errs[:, 7].mean():.4f}   (expert grip is 0 or 1)")

# 2) closed-loop policy rollout
env.reset(cube_xy=xy)
print(f"\nclosed-loop rollout:")
print(f"{'step':>5s} {'tcp-cube':>9s} {'tcp_z':>7s} {'cube_z':>7s} {'grip_cmd':>9s} {'grip_w':>7s}")
for t in range(700):
    o = observe(env)
    with torch.no_grad():
        x = torch.as_tensor((o-om)/os_, dtype=torch.float32, device=dev).unsqueeze(0)
        a = pol(x).squeeze(0).cpu().numpy()*as_ + am
    tgt = np.clip(env.arm_qpos + a[:7], env.arm_limits[:,0], env.arm_limits[:,1])
    grip = float(np.clip(a[7], 0, 1))*GRIPPER_SCALE
    env.step(tgt, grip, n_substeps=DEC)
    if t % 60 == 0 or t == 699:
        d = np.linalg.norm(env.tcp_pos - env.cube_pos)
        print(f"{t:5d} {d*1000:8.1f}m {env.tcp_pos[2]:7.4f} {env.cube_pos[2]:7.4f} "
              f"{grip:9.1f} {env.gripper_width*1000:6.1f}m")
print(f"lifted={env.lifted()}")
env.close()
