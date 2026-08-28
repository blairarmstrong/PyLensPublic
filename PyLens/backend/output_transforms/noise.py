from .modifying import Modifying
from ..array_factory import Array_factory as af


class Noise(Modifying):
    """This makes the output noisy. The type of noise is determined by the group's noiseProc and noiseRange parameters.
    """
    noise_proc = "addGaussianNoise"
    noise_range = None
    #func_deriv = None


    def __init__(self, group, noise_proc="addGaussianNoise", noise_range=0.1):
        super().__init__("Noise", group)
        self.noise_range = self.group.network.noise_range if af.isnan(self.group.noise_range) else self.group.noise_range
        self.noise_proc = self.group.noise_proc
       # self.func_deriv = elementwise_grad(self.func)

    def func(self, x):
        self.unitHistoryData[self.group.curr_tick] = x

        if self.noise_proc == "addGaussianNoise": #other option is addUniformNoise
            return x + af.random_normal(0, self.noise_range, x.shape)
        elif self.noise_proc == "multiplyGaussianNoise":
            return x * af.random_normal(0, self.noise_range, x.shape)
        elif self.noise_proc == "addUniformNoise":
            return x + af.random_uniform(-self.noise_range, self.noise_range, x.shape)
        elif self.noise_proc == "multiplyUniformNoise":
            return x * af.random_uniform(-self.noise_range, self.noise_range, x.shape)

    def forward(self, x):
        return self.func(x)

    def backward(self, x, y):
        return self.unitHistoryData[self.group.curr_tick]
