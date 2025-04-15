"""
This SumoIntersectionEnv class takes in parameter a SUMO config file (like an intersection)
and converts it into a gym environment that the reinforce agent can interract with
"""

import traci
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

from scheduler import TlScheduler
scheduler = TlScheduler(tp_min=15, tl_ids=["gneJ1"])


class SumoIntersectionEnv(gym.Env):
    def __init__(self, sumo_cfg_path, use_gui=False, max_steps=1000):
        super().__init__()
        self.sumo_cfg = sumo_cfg_path
        self.use_gui = use_gui # for visualization
        self.max_steps = max_steps # max steps in 1 episode
        self.step_count = 0
        self.vmax = 13.89 # max speed 50km/h default
        self.last_tsd = 1e-6

        self.sumo_binary = "sumo-gui" if use_gui else "sumo" # choice of visualization
        self.tls_id = "gneJ1"  # traffic light ID ("C" for center traffic light)
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
            "--step-length", "1.0",  # simulation step duration (default: 1s)
            "--quit-on-end"  # cleanly shuts SUMO after last vehicle
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        traci.close()
        self._start_sumo()
        self.step_count = 0
        scheduler.reset()
        return self._get_obs(), {}

    def step(self, action):
        # Check if cooldown phase is terminated
        if scheduler.can_act(self.tls_id):
            traci.trafficlight.setPhase(self.tls_id, action)
            scheduler.set_cooldown(self.tls_id)

        # Advance the simulation
        for _ in range(10):  # Simulate 10 steps per decision
            scheduler.step()  # Update cooldowns every real step
            traci.simulationStep()
            self.step_count += 1

        obs = self._get_obs()

        # ---------------------
        # Computing total squared delay (TSD) reward at every step
        # ---------------------
        vehicle_ids = traci.vehicle.getIDList()
        tsd = 0.0

        for vid in vehicle_ids:
            try:
                v = traci.vehicle.getSpeed(vid) # getting vehicle speed at particular step
                delay = 1.0 - (v / self.vmax) # delay rate compared to max possible speed
                tsd += delay ** 2 # compute sum of squares
            except traci.TraCIException:
                pass  # in case a vehicle disappears this step

        tsd_max = max(tsd, self.last_tsd)
        self.last_tsd = tsd_max  # update memory

        reward = 1.0 - (tsd / tsd_max if tsd_max > 0 else 0.0)
        terminated = self.step_count >= self.max_steps
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([traci.lane.getWaitingTime(l) for l in self.controlled_lanes], dtype=np.float32)

    def render(self):
        pass  # GUI does this

    def close(self):
        traci.close()
