"""Second domain (pre-registration v2, section 20): IsaacGymEnvs tasks built without hydra, from the task YAML files of
a local clone (third_party/IsaacGymEnvs, not pip-installed so that the Aerial Gym environment stays unchanged)."""
import os as _os
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_IGE_ROOT = _os.environ.get("ISAACGYMENVS_ROOT", "/opt/IsaacGymEnvs")
import os
import sys

IGE = _IGE_ROOT + ""
if IGE not in sys.path:
    sys.path.insert(0, IGE)

import isaacgym  # noqa: E402,F401
import types  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

# Stub packages so that the package __init__ files (which import hydra and every task) are not executed.
for _name, _sub in (("isaacgymenvs", "isaacgymenvs"), ("isaacgymenvs.tasks", "isaacgymenvs/tasks")):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = [os.path.join(IGE, _sub)]
        sys.modules[_name] = _m

for name, fn in (("eq", lambda x, y: x.lower() == y.lower()), ("contains", lambda x, y: x.lower() in y.lower()),
                 ("if", lambda pred, a, b: a if pred else b),
                 ("resolve_default", lambda default, arg: default if arg in ["", None] else arg)):
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, fn)

def task_cfg(task, num_envs, seed=0, overrides=None):
    base = OmegaConf.load(os.path.join(IGE, "isaacgymenvs/cfg/task", f"{task}.yaml"))
    top = OmegaConf.create({"task": base, "num_envs": num_envs, "seed": seed, "sim_device": "cuda:0",
                            "rl_device": "cuda:0", "physics_engine": "physx", "pipeline": "gpu",
                            "headless": True, "graphics_device_id": 0, "device_id": 0, "multi_gpu": False,
                            "capture_video": False, "num_threads": 4, "solver_type": 1, "num_subscenes": 4})
    cfg = OmegaConf.to_container(top, resolve=True)["task"]
    cfg["env"]["numEnvs"] = num_envs
    cfg["sim"]["use_gpu_pipeline"] = True
    cfg["sim"]["physx"]["use_gpu"] = True
    for k, v in (overrides or {}).items():
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d[kk]
        d[ks[-1]] = v
    return cfg

TASKS = {"Anymal": ("isaacgymenvs.tasks.anymal", "Anymal"), "Ant": ("isaacgymenvs.tasks.ant", "Ant"),
         "AnymalTerrain": ("isaacgymenvs.tasks.anymal_terrain", "AnymalTerrain")}  # section 29: same robot, terrain

def make(task, num_envs, seed=0, overrides=None):
    import importlib
    mod, cls = TASKS[task]
    cfg = task_cfg(task, num_envs, seed, overrides)
    return getattr(importlib.import_module(mod), cls)(cfg=cfg, rl_device="cuda:0", sim_device="cuda:0", graphics_device_id=-1,
                                   headless=True, virtual_screen_capture=False, force_render=False)
