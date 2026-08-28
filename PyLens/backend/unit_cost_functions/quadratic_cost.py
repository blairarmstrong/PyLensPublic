from PyLens.backend.unit_cost_functions.unit_cost import UnitCost
import numpy as np

class QuadraticCost(UnitCost):
    """
    Represents the Quadratic cost function for calculating cost and its derivative
    based on the outputs using a quadratic formula.

    This class extends the UnitCost base class and provides functionality to compute
    the forward pass (cost) and backward pass (derivatives) for a set of output values,
    particularly for use in neural networks or similar systems.

    Parameters:
        group (optional): The group to which this cost function belongs.

    Attributes:
        group: The group to which this cost function belongs.

    Methods:
        forward: Calculates the cost for a given set of outputs.
        backward: Calculates the derivative for a given set of outputs.
    """

    def __init__(self, group):
        super().__init__("Quadratic Cost", group)

    def forward(self, outputs):
        """
        Calculates the cost for the given outputs using the quadratic function.

        The cost is computed by evaluating a quadratic function on each element of the output,
        scaled by network-specific parameters like `output_cost_scale` and `ticks_per_interval`.
        The function handles cases where minimum and maximum output values are specified,
        adjusting the calculation accordingly.

        Parameters:
            outputs (ndarray): The output value for which the cost needs to be calculated.

        Returns:
            float: The total cost calculated for the given outputs.
        """
        cost = 0.0
        min = self.group.minOutput
        max = self.group.maxOutput
        ticks_per_interval = self.group.network.ticks_per_interval
        output_cost_scale = self.group.output_cost_scale
        if min is np.nan or max is np.nan:
            cost = np.square(outputs)
        else:
            p = self.group.network.output_cost_peak if np.isnan(self.group.output_cost_peak) else self.group.output_cost_peak
            invP = 1.0 / (p - min)
            inv1mP = 1.0 / (max - p)
            cost = np.where(outputs <= p, (outputs - min) * invP, (max - outputs) * inv1mP)
            cost = np.square(cost)
        total_cost = sum(cost)
        total_cost *= output_cost_scale / ticks_per_interval
        return total_cost

    def backward(self, outputs):
        """
        Calculates the derivative of the cost with respect to the outputs.

        This function computes the gradient of the quadratic cost function with respect to
        the output values, which is essential for backpropagation and optimization tasks. 
        The derivative computation varies depending on whether minimum and maximum output values 
        are defined.

        Parameters:
            outputs (ndarray)

        Returns:
            ndarray: The calculated derivatives corresponding to each output value.
        """
        output_cost_scale = self.group.output_cost_scale
        output_cost_strength = self.group.network.output_cost_strength
        ticks_per_interval = self.group.network.ticks_per_interval
        strength = output_cost_strength * output_cost_scale * 2.0 / ticks_per_interval
        min = self.group.minOutput
        max = self.group.maxOutput
        if min is np.nan or max is np.nan:
            output_derivs = strength * outputs
        else:
            p = self.group.network.output_cost_peak if np.isnan(self.group.output_cost_peak) else self.group.output_cost_peak
            left_scalue = strength / np.square(p - min)
            right_scale = - strength / np.square(max - p)
            output_derivs = np.where(outputs < p, left_scalue * (outputs - min), np.where(outputs > p, right_scale * (max - outputs), 0.0))
        return output_derivs
    
# static void quadraticCost(Group G, GroupProc P) {
#   real cost = 0.0, min = G->minOutput, max = G->maxOutput;
#   if (isNaN(min) || isNaN(max)) {
#     FOR_EACH_UNIT(G, cost += SQUARE(U->output));
#   } else {
#     real p = chooseValue(G->outputCostPeak, Net->outputCostPeak), 
#       invP = 1.0 / (p - min), inv1mP = 1.0 / (max - p), v;
#     FOR_EACH_UNIT(G, {
#       v = (U->output <= p) ? ((U->output - min) * invP) : (max - U->output) * inv1mP;
#       cost += SQUARE(v);
#     });
#   }
#   cost *= G->outputCostScale / Net->ticksPerInterval;
#   G->outputCost += cost;
#   Net->outputCost += cost;
# }

# static void quadraticCostBack(Group G, GroupProc P) {
#   real strength = Net->outputCostStrength * G->outputCostScale * 2.0 /
#     Net->ticksPerInterval, min = G->minOutput, max = G->maxOutput;
#   if (isNaN(min) || isNaN(max)) {
#     FOR_EACH_UNIT(G, U->outputDeriv += strength * U->output);
#   } else {
#     real p = chooseValue(G->outputCostPeak, Net->outputCostPeak),
#       leftScale = strength / SQUARE(p - min), 
#       rightScale = -strength / SQUARE(max - p);
#     FOR_EACH_UNIT(G, {
#       U->outputDeriv += (U->output < p) ? leftScale * (U->output - min) :
# 	(U->output > p) ? rightScale * (max - U->output) : 0.0;
#     });
#   }
# }

