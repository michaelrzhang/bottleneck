import numpy as np

class FrozenSimpleEnv(object):
    def __init__(self, randomness=0.75):
        self._state = 0
        self.randomness = randomness

    def step(self, action):
        
        rand_val = np.random.random()
        if action == 0:
            if rand_val > self.randomness:
                self._state -= 1
            else:
                self._state += 1
        else:
            if rand_val > self.randomness:
                self._state += 1
            else:
                self._state -= 1

        if self._state == 3:
            rew = 1
            done = True
        elif self._state == -3:
            rew = -1
            done = True
        else:
            rew = 0
            done = False
        
        return self._state, float(rew), done, {}

    def reset(self):
        self._state = 0
        return self._state
