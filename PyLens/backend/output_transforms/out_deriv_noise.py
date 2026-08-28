from .modifying import Modifying
from ..array_factory import Array_factory as af
from .noise_proc import *


class Out_Deriv_Noise(Modifying):
    """This crops the output to within the range [minOutput, maxOutput]. You may want to use this after OUT_NOISE to
     prevent outputs outside of this range.
    """

    func_deriv = None


    def __init__(self, group):
        super().__init__("Out_Deriv_Noise", group)
        self.maxOutput = self.group.maxOutput
        self.minOutput = self.group.minOutput
        self.noise_proc = self.group.noise_proc

    # def forward(self, x):
    #     scale = af.sum(x)
    #     self.group.groupHistoryData[self.group.curr_tick] = scale
    #     if scale != 0:
    #         return x * scale
    #     return x


    def backward(self, x, output_derivs):
        noise_range = self.group.network.noise_range if af.isnan(self.group.noise_range) else self.group.noise_range
        dim = output_derivs.shape
        if self.noise_proc == "addGaussianNoise":
            self.group.output_derivs = output_derivs + gaussian_noise(0, noise_range, dim)
        elif self.noise_proc == "multiplyGaussianNoise":
            self.group.output_derivs = output_derivs * gaussian_noise(0, noise_range, dim)
        elif self.noise_proc == "addUniformNoise":
            self.group.output_derivs = output_derivs + uniform_noise(-noise_range, noise_range, dim)
        elif self.noise_proc == "multiplyUniformNoise":
            self.group.output_derivs = output_derivs * gaussian_noise(-noise_range, noise_range, dim)
