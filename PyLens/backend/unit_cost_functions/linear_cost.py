from PyLens.backend.unit_cost_functions.unit_cost import UnitCost
import numpy as np


class LinearCost(UnitCost):
    """
    Represents the Cosine Linear function.

    This cost function calculates the cost and derivative for a given set of outputs using the Linear formula.

    Parameters:
        group (optional): The group to which this cost function belongs.

    Attributes:
        group: The group to which this cost function belongs.

    Methods:
        forward: Calculates the cost for a given set of outputs.
        backward: Calculates the derivative for a given set of outputs.
    """

    def __init__(self, group):
        super().__init__("Linear Cost", group)

    def forward(self, outputs):
        """
        Calculates the cost for a given set of outputs using the linear formula.

        If the minimum and maximum values for outputs are not set, it calculates the absolute value cost.
        If they are set, the cost is calculated based on the linear scaling of the output range.

        Parameters:
            outputs (ndarray): The output values from the network layer for which the cost is to be computed.

        Returns:
            float: The total computed cost for the outputs.
        """
        cost = 0.0
        min = self.group.minOutput
        max = self.group.maxOutput
        ticks_per_interval = self.group.network.ticks_per_interval
        output_cost_scale = self.group.output_cost_scale
        if min is np.nan or max is np.nan:
            cost = np.abs(outputs)
        else:
            p = self.group.network.output_cost_peak if np.isnan(self.group.output_cost_peak) else self.group.output_cost_peak
            invP = 1.0 / (p - min)
            inv1mP = 1.0 / (max - p)
            cost = np.where(outputs <= p, (outputs - min) * invP, (max - outputs) * inv1mP)
        total_cost = sum(cost)
        total_cost *= output_cost_scale / ticks_per_interval
        return total_cost

    def backward(self, outputs):
        """
        Calculates the derivative of the linear cost function with respect to the outputs.

        The derivative is calculated based on whether the minimum and maximum output values are set.
        If not set, the derivative is computed using a simple threshold logic.

        Parameters:
            outputs (ndarray)

        Returns:
            ndarray: The calculated derivatives for the given outputs.
        """
        output_cost_scale = self.group.output_cost_scale
        output_cost_strength = self.group.network.output_cost_strength
        ticks_per_interval = self.group.network.ticks_per_interval
        strength = output_cost_strength * output_cost_scale / ticks_per_interval
        min = self.group.minOutput
        max = self.group.maxOutput
        if min is np.nan or max is np.nan:
            output_derivs = np.where(outputs > 0, strength, np.where(outputs < 0, -strength, 0.0))
        else:
            p = self.group.network.output_cost_peak if np.isnan(self.group.output_cost_peak) else self.group.output_cost_peak
            left_deriv = strength / (p - min)
            right_deriv = strength / (p - max)
            output_derivs = np.where(outputs < p, left_deriv, np.where(outputs > p, right_deriv, 0.0))
        return output_derivs


# static void linearCost(Group G, GroupProc P) {
#   real cost = 0.0, min = G->minOutput, max = G->maxOutput;
#   if (isNaN(min) || isNaN(max)) {
#     FOR_EACH_UNIT(G, cost += ABS(U->output));
#   } else {
#     real p = chooseValue(G->outputCostPeak, Net->outputCostPeak), 
#       invP = 1.0 / (p - min), inv1mP = 1.0 / (max - p);
#     FOR_EACH_UNIT(G, cost += (U->output <= p) ? (U->output - min) * invP : 
# 		  (max - U->output) * inv1mP);
#   }
#   cost *= G->outputCostScale / Net->ticksPerInterval;
#   G->outputCost += cost;
#   Net->outputCost += cost;
# }

# static void linearCostBack(Group G, GroupProc P) {
#   real strength = Net->outputCostStrength * G->outputCostScale /
#     Net->ticksPerInterval, min = G->minOutput, max = G->maxOutput;
#   if (isNaN(min) || isNaN(max)) {
#     FOR_EACH_UNIT(G, U->outputDeriv += (U->output > 0) ? strength : 
# 		  (U->output < 0) ? -strength : 0.0);
#   } else {
#     real p = chooseValue(G->outputCostPeak, Net->outputCostPeak),
#       leftDeriv = strength / (p - min),
#       rightDeriv = strength / (p - max);
#     FOR_EACH_UNIT(G, {
#       U->outputDeriv += (U->output < p) ? leftDeriv : 
# 	(U->output > p) ? rightDeriv : 0.0;
#     });
#   }
# }
