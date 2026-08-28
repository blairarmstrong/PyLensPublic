from .optimizer import Optimizer
from ..array_factory import Array_factory as af


class AdamOptimizer(Optimizer):
    """

    The AdamOptimizer class implements the Adam optimization algorithm for updating
    neural network weights.

    Adam combines the advantages of AdaGrad (adaptive learning rates) and RMSProp 
    (exponential moving average of squared gradients), making it effective for 
    non-stationary objectives and problems with noisy and/or sparse gradients.
    """

    def __init__(self, network, lr):
        super().__init__(network, lr)
        self.beta_1 = self.optimizer_params.PAR_O_AdamBeta_1
        self.beta_2 = self.optimizer_params.PAR_O_AdamBeta_2
        self.epsilon = self.optimizer_params.PAR_O_AdamEpsilon
        self.moving_grad_average = None
        self.avg_gradient = None
        self.t = 0

    def update_weights(self, report_request=False):
        """
        Updates the neural network weights for all groups using the Adam optimization algorithm.

        Args:
            report_request (bool, optional): If True, reporting statistics are collected 
                                             during the update step.
        """
        if report_request:
            report_req = self.network.stats_plotter.report_stats
        else:
            report_req = None
        
        # Initialize moving averages of gradients if this is the first update
        if self.t == 0:
            self.moving_grad_average = [[af.zeros_like(group.incoming_links[i].weights) for i in
                                         range(len(group.incoming_links))] for group in self.groups]
            self.avg_gradient = [[af.zeros_like(group.incoming_links[i].weights) for i in
                                  range(len(group.incoming_links))] for group in self.groups]
        self.t += 1
        counter = len(self.groups) - 1
        for group in reversed(self.groups):
            #if group.group_type != "input":
            if group.weight_elimination is not None:
                self.weight_elimination = group.weight_elimination
            self.adam_update_weights(group, counter, report_stats=report_req)
            counter -= 1

    def adam_update_weights(self, group, group_index, report_stats=None):
        """
        Applies the Adam weight update rule to a specific group of the network.

        This includes:
         - Updating the moving average of gradients (m).
         - Updating the moving average of squared gradients (v).
         - Computing the bias-corrected estimates.
         - Updating the group's weights accordingly.
         - Clamping weights if `max_weights` or `min_weights` are defined.

        Args:
            group (Group): The group (layer) in the network whose weights are being updated.
            group_index (int): The index of this group in the overall network group list.
            report_stats: If provided, used to record/report optimization statistics.
        """

        reporting = report_stats is not None
        if reporting:
            group.reset_report_params()

        alpha = self.learning_rate * (af.sqrt(1 - af.power(self.beta_2, self.t)) / (1 - af.power(self.beta_1, self.t)))

        for i in range(len(group.incoming_links)):
            if reporting:
                group.pre_update_report_stat(group.incoming_links[i])

            # first moment
            self.moving_grad_average[group_index][i] = self.beta_1 * self.moving_grad_average[group_index][i] + (
                    1 - self.beta_1) * group.incoming_links[i].weight_derivs

            # second moment
            self.avg_gradient[group_index][i] = self.beta_2 * self.avg_gradient[group_index][i] + (
                        1 - self.beta_2) * af.square(group.incoming_links[i].weight_derivs)

            # Apply weight update
            delta_weight = (
                -alpha * self.moving_grad_average[group_index][i]
                / (af.sqrt(self.avg_gradient[group_index][i]) + self.epsilon)
            )
            group.incoming_links[i].update_weight(
                group.incoming_links[i].weights + delta_weight
            )

            # Optional weight clamping
            max_weights = group.incoming_links[i].max_weights
            min_weights = group.incoming_links[i].min_weights
            max_truth_np = af.greater(group.incoming_links[i].weights, group.incoming_links[i].max_weights)
            min_truth_np = af.less(group.incoming_links[i].weights, group.incoming_links[i].min_weights)

            if not af.isnan(max_weights) and not False in max_truth_np:
                group.incoming_links[i].update_weight(group.incoming_links[i].max_weights)
            if not af.isnan(min_weights) and not False in min_truth_np:
                group.incoming_links[i].update_weight(group.incoming_links[i].min_weights)

            group.incoming_links[i].last_weight_delta = delta_weight


            if reporting:
                group.post_update_report_stat(group.incoming_links[i])
        if reporting:
            group.aggregate_report_stats(report_stats)
