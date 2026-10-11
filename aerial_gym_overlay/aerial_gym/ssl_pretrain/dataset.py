"""Windows of consecutive states and commands from the flight data (HDF5), preloaded into memory. Each objective defines its own target from a window."""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class TrajectoryWindowDataset(Dataset):
    """Random-start sliding window over multi-robot HDF5 trajectory file.

    """

    def __init__(self, h5_path, window_length=32, robots=None, preload=True,
                 max_data_steps=0):
        """
        """
        self.h5_path = h5_path
        self.window = window_length
        self._h5 = None
        self.preload = preload
        self._cache = None  # {robot: {"states": np.ndarray, "actions": np.ndarray}}
        self.max_data_steps = int(max_data_steps)

        with h5py.File(h5_path, "r") as f:
            available_robots = list(f.keys())
            if robots is None:
                robots = available_robots
            self.robots = [r for r in robots if r in available_robots]
            full_lengths = {r: f[r]["states"].shape[0] for r in self.robots}
            if self.max_data_steps > 0:
                per_robot_cap = max(1, self.max_data_steps // max(len(self.robots), 1))
                self.lengths = {r: min(full_lengths[r], per_robot_cap)
                                for r in self.robots}
            else:
                self.lengths = full_lengths
            if preload:
                self._cache = {}
                for r in self.robots:
                    n = self.lengths[r]
                    self._cache[r] = {
                        "states": f[r]["states"][:n],
                        "actions": f[r]["actions"][:n],
                    }

        self.cumlen = [0]
        for r in self.robots:
            n_valid = max(0, self.lengths[r] - self.window - 1)
            self.cumlen.append(self.cumlen[-1] + n_valid)
        self._total_len = self.cumlen[-1]

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        # Locate which robot this idx belongs to (linear search; R is tiny)
        for i, end_cum in enumerate(self.cumlen[1:]):
            if idx < end_cum:
                r = self.robots[i]
                start = idx - self.cumlen[i]
                break
        end = start + self.window + 1
        if self.preload:
            states = self._cache[r]["states"][start:end]
            actions = self._cache[r]["actions"][start:end]
        else:
            self._ensure_open()
            grp = self._h5[r]
            states = grp["states"][start:end]
            actions = grp["actions"][start:end]
        return {
            "state": torch.from_numpy(np.asarray(states, dtype=np.float32)),
            "action": torch.from_numpy(np.asarray(actions, dtype=np.float32)),
        }

def collate_window(batch):
    """Stack windows."""
    return {
        "state": torch.stack([b["state"] for b in batch], dim=0),    # (B, T+1, 13)
        "action": torch.stack([b["action"] for b in batch], dim=0),  # (B, T+1, 4)
    }

def make_input_seq(state_seq, action_seq=None):
    """Backbone input (B, T+1, 13) = state-only (PositionSetpointTask obs).

    """
    return state_seq  # (B, T+1, 13)
