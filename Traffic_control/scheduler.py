class TlScheduler:
    def __init__(self, tp_min, tl_ids):
        self.tp_min = tp_min # min n° of simulation steps before a TL can change phase
        self.tl_ids = tl_ids # IDs of traffic lights

        # Dictionary that tracks cooldown time (in steps) for each traffic light.
        # "cooldown" = time in between two phases
        self.cooldowns = {tl_id: 0 for tl_id in tl_ids}

    def step(self):
        """
        Called once per simulation step.
        Decrements the cooldown timers for all traffic lights that are currently waiting.
        """
        for tl_id in self.cooldowns:
            if self.cooldowns[tl_id] > 0:
                self.cooldowns[tl_id] -= 1

    def can_act(self, tl_id):
        """
        Checks if a given traffic light is allowed to change its phase.
        Returns True if the cooldown has expired and the light can change.
        """
        return self.cooldowns[tl_id] <= 0

    def set_cooldown(self, tl_id):
        """
        Resets the cooldown timer for a traffic light after it changes phase.
        """
        self.cooldowns[tl_id] = self.tp_min

    def reset(self):
        """
        Resets all cooldowns to 0 (ex: at the beginning of an episode)
        """
        for tl_id in self.cooldowns:
            self.cooldowns[tl_id] = 0
