from .optimizer import Optimizer
from ..array_factory import Array_factory as af


class MomentumOptimizer(Optimizer):

    def __init__(self, network, lr):
        super().__init__(network, lr)

    def update_weights(self, report_request=None):
        """
        Updates the weights of all groups in the network using the momentum update rule.

        Args:
            report_request (bool, optional): If True, optimization statistics are collected 
                                             during updates.
        """
        if report_request:
            report_req = self.network.stats_plotter.report_stats
        else:
            report_req = None
            
        # Update each group in reverse order
        for group in reversed(self.groups):
            #if group.group_type != "input":
            if group.weight_elimination is not None:
                self.weight_elimination = group.weight_elimination
            self.momentum_update_weights(group, self.learning_rate, self.momentum,
                                         self.weight_decay, report_stats=report_req)

    def momentum_update_weights(self, group, learning_rate, momentum, weight_decay=0, weight_elimination=0,
                                report_stats=None):
        """
        Applies the momentum-based update rule to the incoming links of a given group.

        The weight update equation is:
            delta_weight = -learning_rate * gradient + momentum * previous_delta_weight
        Then, the new weight is:
            new_weight = old_weight + delta_weight

        Args:
            group (Group): The group whose weights are being updated.
            learning_rate (float): Base learning rate for gradient scaling.
            momentum (float): Momentum factor.
            weight_decay (float, optional): Weight decay (L2 regularization) term.
            weight_elimination (float, optional): Weight elimination term to reduce
                                                  large weights.
            report_stats: If provided, used to record/report optimization statistics.
        """
        w0sq = weight_elimination ** 2
        reporting = report_stats is not None
        if reporting:
            group.reset_report_params()
        for i in range(len(group.incoming_links)):
            if reporting:
                group.pre_update_report_stat(group.incoming_links[i])
            # Calculate the delta weight using momentum
            delta_weight = -learning_rate * group.incoming_links[i].weight_derivs + momentum * \
                           group.incoming_links[i].last_weight_delta
            if weight_decay > 0:
                if weight_elimination > 0:
                    x = 1.0 + ((group.incoming_links[i].weights * group.incoming_links[i].weights) / w0sq)
                    delta_weight -= weight_decay * group.incoming_links[i].weights / (x * x * w0sq)
                else:
                    delta_weight -= weight_decay * group.incoming_links[i].weights
            group.incoming_links[i].update_weight(group.incoming_links[i].weights + delta_weight)
            # Apply max/min constraints if defined
            max_weights = group.incoming_links[i].max_weights
            min_weights = group.incoming_links[i].min_weights
            max_truth_np = af.greater(group.incoming_links[i].weights, group.incoming_links[i].max_weights)
            min_truth_np = af.less(group.incoming_links[i].weights, group.incoming_links[i].min_weights)
            if not af.isnan(max_weights) and not False in max_truth_np:
                group.incoming_links[i].update_weight(group.incoming_links[i].max_weights)
            if not af.isnan(min_weights) and not False in min_truth_np:
                group.incoming_links[i].update_weight(group.incoming_links[i].min_weights)
            group.incoming_links[i].last_weight_delta = delta_weight
            # moved reset weight derivs to network train example beginning of iteration for GUI visualization
            # group.incoming_links[i].reset_weight_derivs()
            if reporting:
                group.post_update_report_stat(group.incoming_links[i])
        if reporting:
            group.aggregate_report_stats(report_stats)
