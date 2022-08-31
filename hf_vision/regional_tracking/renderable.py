import numpy as np


class Renderable:

    def draw(self, image: np.array):
        assert (False, 'must be implemented in sub-class')
        pass
