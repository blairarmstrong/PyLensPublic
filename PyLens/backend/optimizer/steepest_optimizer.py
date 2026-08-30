from .optimizer import Optimizer
from ..array_factory import Array_factory as af

class SteepestOptimizer(Optimizer):

    def __init__(self, network, lr):
        super().__init__(network, lr)

    def update_weights(self, report_request=None):
        """
        Updates the weights of all groups in the network using steepest gradient descent.

        Args:
            report_request (bool, optional): If True, reporting/statistics are gathered
                                             during the update step.
        """
        if report_request:
            report_req = self.network.stats_plotter.report_stats
        else:
            report_req = None
        for group in reversed(self.groups):
            #if group.group_type != "input":
            if group.weight_elimination is not None:
                self.weight_elimination = group.weight_elimination
            self.steepest_descent_update_weights(group, self.learning_rate, self.weight_decay,
                                                    report_stats=report_req)

    def steepest_descent_update_weights(self, group, learning_rate, weight_decay=0, weight_elimination=0,
                                        report_stats=None):
        """
        Applies a steepest (gradient) descent update rule to the weights of the specified group.

        The update rule is:
            delta_weight = -learning_rate * gradient
        An optional weight decay (L2) or weight elimination term can also be subtracted
        from the update.

        Args:
            group (Group): The group (layer) whose weights are being updated.
            learning_rate (float): The base learning rate for gradient descent.
            weight_decay (float, optional): Weight decay (L2 regularization) factor.
            weight_elimination (float, optional): Weight elimination parameter, for
                                                  controlling large weights.
            report_stats: If provided, used to record/report optimization statistics.
        """
        w0sq = weight_elimination ** 2
        reporting = report_stats is not None
        if reporting:
            group.reset_report_params()
        for i in range(len(group.incoming_links)):
            if reporting:
                group.pre_update_report_stat(group.incoming_links[i])
            delta_weight = -learning_rate * group.incoming_links[i].weight_derivs
            # Optional weight decay or weight elimination
            if weight_decay > 0:
                if weight_elimination > 0:
                    x = 1.0 + ((group.incoming_links[i].weights * group.incoming_links[i].weights) / w0sq)
                    delta_weight -= weight_decay * group.incoming_links[i].weights / (x * x * w0sq)
                else:
                    delta_weight -= weight_decay * group.incoming_links[i].weights
            group.incoming_links[i].update_weight(group.incoming_links[i].weights + delta_weight)
            # Enforce max/min weight constraints if defined
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
            #group.incoming_links[i].reset_weight_derivs()
            if reporting:
                group.post_update_report_stat(group.incoming_links[i])
        if reporting:
            group.aggregate_report_stats(report_stats)
