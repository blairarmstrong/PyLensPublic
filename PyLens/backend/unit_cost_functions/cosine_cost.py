from PyLens.backend.unit_cost_functions.unit_cost import UnitCost
import numpy as np

class CosineCost(UnitCost):
    """
    Represents the Cosine Cost function, which computes a cost and its derivative
    for a given set of outputs based on a cosine function. This cost function is 
    used in network training for optimization purposes.

    Parameters:
        group (optional): The group to which this cost function belongs.

    Attributes:
        group: The group to which this cost function belongs.

    Methods:
        forward: Calculates the cost for a given set of outputs.
        backward: Calculates the derivative for a given set of outputs.
    """

    def __init__(self, group):
        super().__init__("Cosine Cost", group)

    def forward(self, outputs):
        """
        Calculates the total cost for the given set of output values using the Cosine cost formula.

        The formula used is:
        `cost = 1 - cos(π * invp * output)` when `output <= p`
        or
        `cost = 1 - cos(π * inv1mp * (output - 2 * p + 1))` when `output > p`

        where `p` is the output cost peak, `invp` is the inverse of `p`, and `inv1mp` is the inverse of `1 - p`.

        Parameters:
            outputs (ndarray): The output values from the network layer for which the cost is computed.

        Returns:
            float: The total computed cost.
        """
        cost = 0.0
        ticks_per_interval = self.group.network.ticks_per_interval
        output_cost_scale = self.group.output_cost_scale
        p = self.group.network.output_cost_peak if np.isnan(self.group.output_cost_peak) else self.group.output_cost_peak
        invp = 1.0 / p
        inv1mp = 1.0 / (1.0 - p)
        cost = np.where(outputs <= p, 1 - np.cos(np.pi * invp * outputs),
                        1 - np.cos(np.pi * inv1mp * (outputs - 2 * p + 1)))
        total_cost = sum(cost)
        total_cost *= output_cost_scale / ticks_per_interval
        return total_cost

    def backward(self, outputs):
        """
        Computes the derivative of the Cosine cost function with respect to the output values.

        The formula used is:
        `derivative = strength * invp * sin(π * invp * output)` when `output <= p`
        or
        `derivative = strength * inv1mp * sin(π * inv1mp * (output - 2 * p + 1))` when `output > p`

        where `strength` is based on various group/network parameters.

        Parameters:
            outputs (ndarray)

        Returns:
            ndarray: The computed derivatives of the cost function with respect to the output values.
        """
        output_cost_scale = self.group.output_cost_scale
        output_cost_strength = self.group.network.output_cost_strength
        ticks_per_interval = self.group.network.ticks_per_interval
        strength = output_cost_strength * output_cost_scale * np.pi * 0.5 / ticks_per_interval
        p = self.group.network.output_cost_peak if np.isnan(self.group.output_cost_peak) else self.group.output_cost_peak
        invp = 1.0 / p
        inv1mp = 1.0 / (1.0 - p)
        output_derivs = np.where(outputs <= p, strength * invp * np.sin(np.pi * invp * outputs),
                                strength * inv1mp * np.sin(np.pi * inv1mp * (outputs - 2 * p + 1))
                                )
        return output_derivs


# static void cosineCost(Group G, GroupProc P) {
#   real cost = 0.0, 
#     p = chooseValue(G->outputCostPeak, Net->outputCostPeak), 
#     invp = 1.0 / p,
#     inv1mp = 1.0 / (1.0 - p);
#   FOR_EACH_UNIT(G, {
#     cost += (U->output <= p) ? (1 - cos(PI * invp * U->output)) :
#       (1 - cos(PI * inv1mp * (U->output - 2 * p + 1)));
#   });
#   cost *= G->outputCostScale / Net->ticksPerInterval;
#   G->outputCost += cost;
#   Net->outputCost += cost;
# }

# static void cosineCostBack(Group G, GroupProc P) {
#   real strength = Net->outputCostStrength * G->outputCostScale *
#     PI * 0.5 / Net->ticksPerInterval,
#     p = chooseValue(G->outputCostPeak, Net->outputCostPeak), 
#     invp = 1.0 / p,
#     inv1mp = 1.0 / (1.0 - p);
#   FOR_EACH_UNIT(G, {
#     U->outputDeriv += (U->output <= p) ?
#       strength * invp * sin(PI * invp * U->output) :
#       strength * inv1mp * sin(PI * inv1mp * (U->output - 2 * p + 1));
#   });
# }
