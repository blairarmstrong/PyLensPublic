from PyLens.backend.unit_cost_functions.unit_cost import UnitCost
import numpy as np

class LogisticCost(UnitCost):
    """
    Represents the Logistic cost function for calculating cost and its derivative
    based on the outputs using a logistic formula.

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
        super().__init__("Logistic Cost", group)

    def forward(self, outputs):
        """
        Calculates the cost for the given outputs using the logistic function.

        The cost is computed by evaluating the logistic function on each element of the output,
        scaled by network-specific parameters like `output_cost_scale` and `ticks_per_interval`.
        The function also handles boundary cases where outputs are 0 or 1, avoiding undefined values.

        Parameters:
            outputs (ndarray): The output value, typically in the range [0, 1],
            for which the cost needs to be calculated.

        Returns:
            float: The total cost calculated for the given outputs.
        """
        cost = 0.0
        ticks_per_interval = self.group.network.ticks_per_interval
        output_cost_scale = self.group.output_cost_scale
        p = self.group.network.output_cost_peak if np.isnan(self.group.output_cost_peak) else self.group.output_cost_peak
        logp = np.log(p)
        log1mp = np.log(1.0 - p)
        scale = 1.0 / logp if p <= 0.5 else 1.0 / log1mp
        cost = np.where(outputs <= 0.0, -log1mp * scale + 1.0, 
                        np.where(outputs >= 1.0, -logp * scale + 1.0, 
                                 (outputs*(np.log(outputs)-logp) + (1-outputs)*(np.log(1-outputs)-log1mp)) * scale + 1.0))
        total_cost = sum(cost)
        total_cost *= output_cost_scale / ticks_per_interval
        return total_cost

    def backward(self, outputs):
        """
        Calculates the derivative of the cost with respect to the outputs.

        This function computes the gradient of the logistic cost function with respect to
        the output values, which is essential for backpropagation and optimization tasks. 
        It handles boundary cases to prevent division by zero or logarithmic errors.

        Parameters:
            outputs (ndarray)

        Returns:
            ndarray: An array of derivatives corresponding to each output value.
        """
        output_cost_scale = self.group.output_cost_scale
        output_cost_strength = self.group.network.output_cost_strength
        ticks_per_interval = self.group.network.ticks_per_interval
        strength = output_cost_strength * output_cost_scale / ticks_per_interval
        p = self.group.network.output_cost_peak if np.isnan(self.group.output_cost_peak) else self.group.output_cost_peak
        logp = np.log(p)
        log1mp = np.log(1.0 - p)
        scale = strength / logp if p <= 0.5 else strength / log1mp
        output_derivs = np.where(outputs < 1e-6, 1e-6, 
                                 np.where(outputs > (1.0 - 1e-6), 1.0 - 1e-6, 
                                          (np.log(outputs) - logp - np.log(1-outputs) + log1mp) * scale))
        return output_derivs

    
# static void logisticCost(Group G, GroupProc P) {
#   real cost = 0.0, 
#     p = chooseValue(G->outputCostPeak, Net->outputCostPeak),
#     logp = LOG(p), log1mp = LOG(1.0 - p),
#     scale = (p <= 0.5) ? (real) 1.0 / logp : (real) 1.0 / log1mp, x;
#   FOR_EACH_UNIT(G, {
#     x = U->output;
#     if (x <= 0.0) cost += -log1mp * scale + 1.0;
#     else if (x >= 1.0) cost += -logp * scale + 1.0;
#     else cost += (x*(LOG(x)-logp) + (1-x)*(LOG(1-x)-log1mp)) * scale + 1.0;
#   });
#   cost *= G->outputCostScale / Net->ticksPerInterval;
#   G->outputCost += cost;
#   Net->outputCost += cost;
# }

# static void logisticCostBack(Group G, GroupProc P) {
#   real strength = Net->outputCostStrength * G->outputCostScale /
#     Net->ticksPerInterval,
#     p = chooseValue(G->outputCostPeak, Net->outputCostPeak),
#     logp = LOG(p), log1mp = LOG(1.0-p),
#     scale = (p <= 0.5) ? strength / logp : strength / log1mp, x;
#   FOR_EACH_UNIT(G, {
#     x = U->output;
#     if (x < 1e-6) x = 1e-6;
#     else if (x > (1.0 - 1e-6)) x = 1.0 - 1e-6;
#     U->outputDeriv += (LOG(x) - logp - LOG(1-x) + log1mp) * scale;
#   });
# }
