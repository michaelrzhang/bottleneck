import numpy as np

# todo: rename state position

class FrozenSimpleEnv(object):
    def __init__(self, randomness=0.75, constant_direction = True):
        self._state = 0
        self.randomness = randomness
        self.constant_direction = constant_direction

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
            if self._direction == 1:
                rew = 1
            else:
                rew = -1
            done = True
        elif self._state == -3:
            if self._direction == 1:
                rew = -1
            else:
                rew = 1
            done = True
        else:
            rew = 0
            done = False
        
        return np.array((self._state, self._direction)), float(rew), done, {}

    def reset(self):
        self._state = 0
        if self.constant_direction:
            self._direction = 1
        else:
            self._direction = int(np.random.random() < 0.5)
        return np.array((self._state, self._direction))
