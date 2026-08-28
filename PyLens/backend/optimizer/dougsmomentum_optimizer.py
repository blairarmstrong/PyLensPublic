from .optimizer import Optimizer
from ..array_factory import Array_factory as af



class DougsMomentumOptimizer(Optimizer):
    """

    dougsMomentum is exactly like momentum with the exception that the pre-momentum
    weight step vector is bounded so that its length cannot exceed 1.0. After the momentum
    is added, the length of the resulting weight change vector can grow as high as 1 / momentum.
    This change allows stable behavior with much higher initial learning rates, resulting in
    less need to adjust the learning rate as training progresses. It is usually safe to start
    training with high momentum in dougsMomentum, but not in standard momentum descent.
    """

    def __init__(self, network, lr):
        super().__init__(network, lr)
        self.sum = None

    def update_weights(self, report_request=False):
        """
        Updates the weights of all groups in the network using the Dougs Momentum update rule.

        Args:
            report_request (bool, optional): If True, statistics are gathered/reported
                                             during updates.
        """
        if report_request:
            report_req = self.network.stats_plotter.report_stats
        else:
            report_req = None
        # Iterate over groups in reverse, applying Dougs Momentum
        for group in reversed(self.groups):
            #if group.group_type != "input":
            if group.weight_elimination is not None:
                self.weight_elimination = group.weight_elimination

            self.dougs_momentum_update_weights(group, self.learning_rate, self.momentum,
                                               self.weight_decay, self.weight_elimination, report_stats=report_req)

        self.sum = None

    def dougs_momentum_update_weights(self, group, learning_rate, momentum, weight_decay=0, weight_elimination=0,
                                      report_stats=None):
        """
        Applies the Dougs Momentum weight update rule for a single group in the network.

        Args:
            group (Group): The group (layer) whose weights will be updated.
            learning_rate (float): Base learning rate for scaling the gradient.
            momentum (float): Momentum factor.
            weight_decay (float, optional): Weight decay (L2 regularization) term.
            weight_elimination (float, optional): Weight elimination parameter for reducing 
                                                  the impact of large weights.
            report_stats: If provided, used to record/report optimization statistics.
        """
        # TODO deal with frozen units and groups

        if not self.sum:
            self.sum = 0
            for g in self.groups:
                for link in g.incoming_links:
                    self.sum += af.sum(af.square(link.weight_derivs))


        if self.sum > 1.0:
            scale = 1 / (self.sum ** 0.5)
        else:
            scale = 1.0

        w0sq = weight_elimination ** 2
        reporting = report_stats is not None
        if reporting:
            group.reset_report_params()
        for i in range(len(group.incoming_links)):
            if reporting:
                group.pre_update_report_stat(group.incoming_links[i])

            delta_weight = -learning_rate * scale * group.incoming_links[i].weight_derivs + momentum * \
                           group.incoming_links[i].last_weight_delta

            if weight_decay > 0:
                if weight_elimination > 0:
                    x = 1.0 + ((group.incoming_links[i].weights * group.incoming_links[i].weights) / w0sq)
                    delta_weight -= weight_decay * group.incoming_links[i].weights / (x * x * w0sq)
                else:
                    delta_weight -= weight_decay * group.incoming_links[i].weights

            group.incoming_links[i].update_weight(group.incoming_links[i].weights + delta_weight)

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
