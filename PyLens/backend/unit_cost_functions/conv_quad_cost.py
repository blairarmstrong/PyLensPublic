from PyLens.backend.unit_cost_functions.unit_cost import UnitCost
import numpy as np

class ConvQuadCost(UnitCost):
    """
    Represents the Convex Quadratic (Conv-Quad) Cost function, which computes a cost and its derivative 
    for a given set of outputs based on a convex quadratic formula. This cost function is primarily used 
    in network training for optimization purposes.
    
    Parameters:
        group (optional): The group to which this cost function belongs.

    Attributes:
        group: The group to which this cost function belongs.

    Methods:
        forward: Calculates the cost for a given set of outputs.
        backward: Calculates the derivative for a given set of outputs.
    """

    def __init__(self, group):
        super().__init__("Conv-Quad Cost", group)
        
    def forward(self, outputs):
        """
        Calculates the total cost for the given set of output values using the Conv-Quad cost formula.

        The formula used is:
        `cost = (output * (p * 2 - output) - min) * scale`
        where `p` is the output cost peak, `min` is a value based on `p`, and `scale` is computed 
        using `p` and other group/network parameters.

        Parameters:
            outputs (ndarray): The output values from the network layer for which the cost is computed.

        Returns:
            float: The total computed cost.
        """
        cost = 0.0
        ticks_per_interval = self.group.network.ticks_per_interval
        output_cost_scale = self.group.output_cost_scale
        p = self.group.network.output_cost_peak if np.isnan(self.group.output_cost_peak) else self.group.output_cost_peak
        min = p * 2 - 1 if p < 0.5 else 0.0
        scale = 1.0 / (p * p - min)
        cost = (outputs * (p * 2 - outputs) - min) * scale
        total_cost = sum(cost)
        total_cost *= output_cost_scale / ticks_per_interval
        return total_cost

    def backward(self, outputs):
        """
        Computes the derivative of the Conv-Quad cost function with respect to the output values.

        The formula used is:
        `derivative = (p - output) * scale`
        where `scale` is a factor computed using group/network parameters.

        Parameters:
            outputs (ndarray)

        Returns:
            ndarray: The computed derivatives of the cost function with respect to the output values.
        """
        output_cost_scale = self.group.output_cost_scale
        p = self.group.network.output_cost_peak if np.isnan(self.group.output_cost_peak) else self.group.output_cost_peak
        ticks_per_interval = self.group.network.ticks_per_interval
        min = p * 2 - 1 if p < 0.5 else 0.0
        output_cost_strength = self.group.network.output_cost_strength
        scale = output_cost_strength * output_cost_scale * 2.0 / (ticks_per_interval * (p * p - min))
        output_derivs = (p - outputs) * scale
        return output_derivs

# static void convexQuadraticCost(Group G, GroupProc P) {
#   real cost = 0.0, 
#     p = chooseValue(G->outputCostPeak, Net->outputCostPeak),
#     min = (p < 0.5) ? p * 2 - 1 : 0.0,
#     scale = 1.0 / (p * p - min);
#   FOR_EACH_UNIT(G, cost += (U->output * (p * 2 - U->output) - min) * scale);
#   cost *= G->outputCostScale / Net->ticksPerInterval;
#   G->outputCost += cost;
#   Net->outputCost += cost;
# }

# static void convexQuadraticCostBack(Group G, GroupProc P) {
#   real p = chooseValue(G->outputCostPeak, Net->outputCostPeak),
#     min = (p < 0.5) ? 2 * p - 1 : 0.0,
#     scale = Net->outputCostStrength * G->outputCostScale * 2.0 /
#     (Net->ticksPerInterval * (p * p - min));
#   FOR_EACH_UNIT(G, U->outputDeriv += (p - U->output) * scale);
# }
