import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator
# NEGATION NETWORK TEST
# momentum = 0.9, lr = 0.1, using steepest

def negation_test():
    updates = 2
    batch_size = 0
    report_interval = 1
    learning_rate=0.1
    update_method = "steepest"

    # create simulator
    sim = Simulator(name="sim_neg")

    # create sample network
    negation_net = sim.create_net(name="negation_net")
    negation_net.toggle_plots(plots=False)
    negation_net.toggle_keyboard(use=False)
    # create input layer with 4 units
    negation_net.add_group(
        4,
        name="input",
        group_type="input",
        input_transforms=[],
        output_transforms=[])

    # create hidden layer with 3 units; used sigmoid output, dot product input
    negation_net.add_group(
        3, name="hidden",
        group_type="hidden",
        input_transforms=["dot"],
        output_transforms=["sigmoid"])

    # create output layer with 3 units; used sigmoid output, dot product input, cross entropy error
    negation_net.add_group(
        3,
        name="output",
        group_type="output",
        input_transforms=["dot"],
        output_transforms=["sigmoid"],
        error_function="cross_entropy")

    # connects the input layer to the hidden layer; uniform links; full projection
    negation_net.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full")

    # connects the hidden layer to the output layer; uniform links; full projection
    negation_net.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        link_type="uniform",
        proj_type="full")

    # connects the input layer to the output layer; uniform links; full projection
    negation_net.connect_groups(
        outgoing_group="input",
        incoming_group="output",
        link_type="uniform",
        proj_type="full")


    # load example set
    negation_net.load_example_set("./examples/lens_example_input/negation.ex")

    # load sample weights from clens random weight seed
    negation_net.load_clen_weight("./examples/example_networks/negation_examples/negation_example_one_weights.wt")

    # set learning rate and update method
    negation_net.set_update_method(learning_rate, update_method)

    # set epochs=100, batch size=0, report interval=10
    negation_net.train(updates, batch_size, report_interval)


    # COMPARE ERRORS AND WEIGHT COST
    # clens - error: 31.94658 weight cost: 11.80113
    clens_error = round(34.48818, 2)
    clens_weight_cost = round(4.96856, 3)

    # pylens - error: 31.94589 weight cost: 11.80113
    pylens_error = round(negation_net.stats_plotter.progress_stats['error'][-1], 2)
    pylens_weight_cost = round(negation_net.stats_plotter.progress_stats["weight_cost"][-1], 3)

    print("testing negation:")
    print("     Error: ... ", end="")
    if pylens_error != clens_error:
        print("Failed")
        print("Errors differ by: {}".format(abs(pylens_error-clens_error)))
        return False
    print("passed")
    print("     Weight Costs: ... ", end="")
    if pylens_weight_cost != clens_weight_cost:
        print("Failed")
        print("Weight costs differ by: {}".format(abs(pylens_weight_cost-clens_weight_cost)))
        return False
    print("passed")
    return True

if __name__ == '__main__':
    result = negation_test()
    print("done test")
    if result:
        exit(0)
    else:
        exit(1)
