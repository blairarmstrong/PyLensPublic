import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator

def xor_runner():
    # hyperparameters
    batch_size = 0
    num_updates = 10
    report_interval = 1
    learning_rate = 0.1

    update_method = "dougs momentum"
    # create simulator
    sim = Simulator(name="sim_xor")

    # create sample network
    xor_net_one = sim.create_net(name="xor")
    xor_net_one.toggle_plots(plots=False)
    xor_net_one.toggle_keyboard(use=False)
    # create input layer with 2  units
    xor_net_one.add_group(
        2,
        name="input",
        group_type="input",
        input_transforms=[],
        output_transforms=[]
    )

    # create hidden layer with 2 units; used sigmoid output, dot product input
    xor_net_one.add_group(
        2,
        name="hidden",
        group_type="hidden",
        input_transforms=["dot"],
        output_transforms=["soft_max"]
    )

    # create output layer with 1 units; used sigmoid output, dot product input, cross entropy error
    xor_net_one.add_group(
        1,
        name="output",
        group_type="output",
        input_transforms=["dot"],
        output_transforms=["soft_max"],
        error_function="cross_entropy"
    )

    # connects the input layer to the hidden layer; uniform links; full projection
    xor_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="full"
    )

    # connects the hidden layer to the output layer; uniform links; full projection
    xor_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        initialization="uniform",
        proj_type="full"
    )

    # load example set
    xor_net_one.load_example_set("./examples/lens_example_input/xor.ex")

    # load sample weights from clens random weight seed
    xor_net_one.load_clen_weight("./examples/example_networks/xor_examples/xor_example_one_weights.wt")

    # set learning rate and update method
    xor_net_one.set_update_method(learning_rate, update_method)

    # set epochs=10, batch size=0, report interval=10
    xor_net_one.train(num_updates, batch_size, report_interval)
    # COMPARE ERRORS AND WEIGHT COST

    decimal = 3
    clens_error = round(200000000, decimal)
    clens_weight_cost = round(1.93923, decimal)

    pylens_error = round(xor_net_one.stats_plotter.progress_stats['error'][-1], decimal)
    pylens_weight_cost = round(xor_net_one.stats_plotter.progress_stats["weight_cost"][-1], decimal)

    print("testing xor:")
    print("     Error: ... ", end="")
    sim.delete_all_nets()
    if pylens_error != clens_error:
        print("Failed")
        print("Errors differ by: {}".format(abs(pylens_error-clens_error)))
        return 0
    print("passed")
    print("     Weight Costs: ... ", end="")
    if pylens_weight_cost != clens_weight_cost:
        print("Failed")
        print("Weight costs differ by: {}".format(abs(pylens_weight_cost-clens_weight_cost)))
        return 0
    print("passed")
    return True


def test_xor():
    result = xor_runner()
    assert result == 1


test_xor()


