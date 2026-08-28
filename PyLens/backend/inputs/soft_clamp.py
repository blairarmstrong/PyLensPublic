""" CLen Implementation: file  act.c
static void softClampInput(Group G, GroupProc P) {
  real initOutput = chooseValue(G->initOutput, Net->initOutput),
    gain = chooseValue(G->gain, Net->gain),
    strength = chooseValue(G->clampStrength, Net->clampStrength), val;
  FOR_EACH_UNIT(G, {
    if (!isNaN(U->externalInput)) {
      val = initOutput + strength * (U->externalInput - initOutput);
      U->input += INV_SIGMOID(val, gain);
    }
  });
}

#define INV_SIGMOID(y,g) (((y) <= 0.0) ? -LARGE_VAL : \
			  ((y) >= 1.0) ? LARGE_VAL : \
			  (LOG((y) / (1-(y))) / (g)))

/* There is no softClampInputBack */
"""

from ..clamping import Clamping
from ..array_factory import Array_factory as af

class Soft_Clamp(Clamping):
    clamp_strength = 0.5
    gain = 1
    init_output = 0.5

    def __init__(self, group, clamp_strength=0.5, gain=1, init_output=0.5):
        """
        Initializes the Soft_Clamp transformation.

        Args:
            group: The neural network group to which this transformation is applied.
            clamp_strength (float, optional): Strength of the clamping. Defaults to 0.5.
            gain (float, optional): Gain factor for the sigmoid transformation. Defaults to 1.
            init_output (float, optional): Initial output value for the transformation. Defaults to 0.5.
        """
        super().__init__("Soft_clamp", group)
        self.clamp_strength = self.group.network.clamp_strength if af.isnan(self.group.clamp_strength) else self.group.clamp_strength
        self.gain = [self.group.network.gain] if af.isnan(self.group.gain).any() else \
            (self.group.gain if isinstance(self.group.gain, list) else [self.group.gain])
        self.gain = af.array(self.gain)
        self.init_output = self.group.network.initOutput if af.isnan(self.group.initOutput) else self.group.initOutput

    def _inverse_sigmoid(self, y, g):
        """
        Computes the inverse sigmoid function.

        Args:
            y (af.array): Input values.
            g (float): Gain factor for the sigmoid transformation.
        
        Returns:
            af.array: Transformed values using the inverse sigmoid function.
        """
        infty_mask = (y<=0) * -1e8 + (y>=1) * 1e8 # this ensure af.log(y / (1-y)) gives [+/-]1e8 for invalid values
        y[y<=0] = 0.5 # this makes af.log(y / (1-y)) of invalid entries (<=0 or >=1) be 0
        y[y>=1] = 0.5 # this makes af.log(y / (1-y)) of invalid entries (<=0 or >=1) be 0
        result = af.log(y / (1-y)) / g + infty_mask
        return result

    def forward(self, x):
        """
        Applies the soft clamping transformation to the input.

        Args:
            x (af.array): Input values.
        
        Returns:
            af.array: Transformed values after applying soft clamping.
        """
        val = self.init_output + self.clamp_strength * (self.group.external_input - self.init_output)
        return self._inverse_sigmoid(val, self.gain) #should be return x+self._inverse_sigmoid(val, self.gain)?

    def backward(self, x, output_derivs):
        """
        Computes the backward pass for the soft clamping transformation.

        Args:
            x (af.array)
            output_derivs (af.array): Derivative of the output with respect to the input.
        
        Returns:
            af.array: The backpropagated derivatives.
        """
        return output_derivs
