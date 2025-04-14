import traci
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box


class SumoIntersectionEnv(gym.Env):
    def __init__(self, sumo_cfg_path, use_gui=False, max_steps=1000):
        super().__init__()
        self.sumo_cfg = sumo_cfg_path
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.step_count = 0

        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.tls_id = "gneJ1"  # traffic light ID from your .net.xml
        self._start_sumo()

        # Action space = number of phases (SUMO will tell us)
        self.num_phases = len(traci.trafficlight.getAllProgramLogics(self.tls_id)[0].phases)
        self.action_space = Discrete(self.num_phases)

        # Observation: e.g. waiting time per controlled lane
        self.controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        self.observation_space = Box(low=0, high=1000, shape=(len(self.controlled_lanes),), dtype=np.float32)

    def _start_sumo(self):
        traci.start([
            self.sumo_binary,
            "-c", self.sumo_cfg,
            "--start",  # prevents SUMO from waiting for user input
            "--step-length", "1.0",  # controls simulation step duration (default: 1s)
            "--quit-on-end"  # cleanly shuts SUMO after last vehicle
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        traci.close()
        self._start_sumo()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        traci.trafficlight.setPhase(self.tls_id, action)

        # Advance simulation (e.g., 10 steps per decision)
        for _ in range(10):
            traci.simulationStep()
            self.step_count += 1

        obs = self._get_obs()
        reward = -np.sum(obs)  # Minimize total waiting time
        terminated = self.step_count >= self.max_steps
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([traci.lane.getWaitingTime(l) for l in self.controlled_lanes], dtype=np.float32)

    def render(self):
        pass  # GUI does this

    def close(self):
        traci.close()
