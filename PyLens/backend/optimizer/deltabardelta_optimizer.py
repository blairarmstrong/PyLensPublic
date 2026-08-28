from .optimizer import Optimizer
from ..array_factory import Array_factory as af


class DeltaBarDeltaOptimizer(Optimizer):
    """
    Delta-bar-delta maintains an individual learning-rate multiplier for each
    weight. The multiplier increases when the previous weight update points in
    the current negative-gradient direction and decreases otherwise.
    """

    def __init__(self, network, lr):
        super().__init__(network, lr)

    def update_weights(self, report_request=None):
        """
        Updates the weights of all groups in the network using the delta-bar-delta update rule.

        Args:
            report_request (bool, optional): If True, gathering/reporting of statistics
                                             will be requested during the update.
        """
        if report_request:
            report_req = self.network.stats_plotter.report_stats
        else:
            report_req = None
        # Iterate over groups in reverse (e.g., from output layer back to input layer)
        for group in reversed(self.network.groups):
            #if group.name != "input":
            if group.weight_elimination is not None:
                self.weight_elimination = group.weight_elimination
            # Perform delta-bar-delta weight updates on this group
            self.deltabardelta_update_weights(
                    group, 
                    self.rate_increment, 
                    self.rate_decrement, 
                    self.learning_rate,
                    self.momentum, 
                    self.weight_decay, 
                    self.weight_elimination, 
                    report_stats=report_req
                    )

    def deltabardelta_update_weights(self, group, rate_increment, rate_decrement, learning_rate, momentum,
                                     weight_decay=0,
                                     weight_elimination=0, report_stats=None):
        """
        Applies delta-bar-delta updates to the weights of a given group.

        Args:
            group (Group): The neural network group whose weights are being updated.
            rate_increment (float): The factor by which the link learning rate is increased
                                    if the gradient reverses direction.
            rate_decrement (float): The factor by which the link learning rate is multiplied
                                    if the gradient continues in the same direction.
            learning_rate (float): Base learning rate scaling factor for overall adjustments.
            momentum (float): Momentum factor.
            weight_decay (float, optional): Weight decay (L2 regularization) parameter.
            weight_elimination (float, optional): Weight elimination parameter for controlling
                                                  large weights.
            report_stats: If provided, used to record/report statistics for analysis.
        """
    
        w0sq = weight_elimination ** 2
        reporting = report_stats is not None
        if reporting:
            group.reset_report_params()

        # Iterate over each incoming link to apply delta-bar-delta updates
        for i in range(len(group.incoming_links)):
            link = group.incoming_links[i]

            if reporting:
                group.pre_update_report_stat(link)

            last_weight_delta = link.last_weight_delta
            deriv = link.weight_derivs
            link_learning_rate = link.link_learning_rate


            opposite_sign = af.logical_xor(
                deriv < 0,
                last_weight_delta < 0,
            )

            # increase for opposite signs; otherwise decrease
            link_learning_rate = af.where(
                opposite_sign,
                link_learning_rate + rate_increment,
                link_learning_rate * rate_decrement,
            )


            delta_weight = (
                -link_learning_rate * learning_rate * deriv
                + momentum * last_weight_delta
            )

            old_weights = af.copy(link.weights)

            # Apply optional weight decay / weight elimination
            if weight_decay > 0:
                if weight_elimination > 0:
                    x = 1.0 + (old_weights * old_weights) / w0sq
                    delta_weight -= (
                        weight_decay
                        * old_weights
                        / (w0sq * x * x)
                    )
                else:
                    delta_weight -= weight_decay * old_weights

            new_weights = old_weights + delta_weight

            # Apply min/max constraints elementwise
            if not af.isnan(link.min_weights):
                new_weights = af.maximum(new_weights, link.min_weights)

            elif not af.isnan(link.max_weights):
                new_weights = af.minimum(new_weights, link.max_weights)

            link.update_weight(new_weights)

            link.last_weight_delta = new_weights - old_weights
            link.link_learning_rate = link_learning_rate

            if reporting:
                group.post_update_report_stat(link)
        if reporting:
            group.aggregate_report_stats(report_stats)
