import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator

def xor_runner(parallel):
    # Run to Test Example pylens error against clens error
    # STEEPEST, 100 UPDATES

    batch_size = 0
    num_updates = 2
    report_interval = 1
    learning_rate = 0.5
    update_method = "steepest"

    # create simulator
    sim = Simulator(name="sim_xor")

    # create sample network
    xor_net_one = sim.create_net(name="xor")
    # xor_net_one.toggle_print_reports(print_reports=False)
    # xor_net_one.toggle_live_update(live_update=False)
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
        output_transforms=["sigmoid"]
    )

    # create output layer with 1 units; used sigmoid output, dot product input, cross entropy error
    xor_net_one.add_group(
        1,
        name="output",
        group_type="output",
        input_transforms=["dot"],
        output_transforms=["sigmoid"],
        error_function="cross_entropy", 
        unit_cost_function="linear"
    )

    # connects the input layer to the hidden layer; uniform links; full projection
    xor_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full"
    )

    # connects the hidden layer to the output layer; uniform links; full projection
    xor_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        link_type="uniform",
        proj_type="full"
    )

    # load example set
    # xor_net_one.load_example_set("C:/Users/rajan/OneDrive/Desktop/PyLens Workstudy/PYLens/examples/lens_example_input/xor.ex")
    xor_net_one.load_example_set("./examples/lens_example_input/xor_dense.ex")

    # load sample weights from clens random weight seed
    # xor_net_one.load_clen_weight("C:/Users/rajan/OneDrive/Desktop/PyLens Workstudy/PYLens/examples/example_networks/xor_examples/xor_example_one_weights.wt")
    xor_net_one.load_clen_weight("./examples/example_networks/xor_examples/xor_example_one_weights.wt")

    # set learning rate and update method
    xor_net_one.set_update_method(learning_rate, update_method)

    # to use GUI, uncomment
    # sim.use_gui(xor_net_one)

    # set epochs=100, batch size=0, report interval=10
    if parallel == True:
        xor_net_one.train(num_updates, batch_size, report_interval, parallel_mode=True, num_worker=3)
    else:
        xor_net_one.train(num_updates, batch_size, report_interval)

    # COMPARE ERRORS AND WEIGHT COST

    # clens - error: 2.30689 weight cost: 15.02701
    # Below is old
    clens_error = round(2.76677, 3)
    clens_weight_cost = round(1.74981, 3)
    clens_unit_cost = round(3.94316, 3)
    # clens_error = round(2.78054, 3)
    # clens_weight_cost = round(1.78510, 3)

    # pylens - error: 2.30684 weight cost: 15.02701
    pylens_error = round(xor_net_one.stats_plotter.progress_stats['error'][-1], 3)
    pylens_weight_cost = round(xor_net_one.stats_plotter.progress_stats["weight_cost"][-1], 3)
    pylens_unit_cost = round(xor_net_one.stats_plotter.progress_stats["unit_cost"][-1], 3)

    print("testing xor:")
    print("     Error: ... ", end="")
    sim.delete_all_nets()
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
    print("     Unit Costs: ... ", end="")
    if pylens_unit_cost != clens_unit_cost:
        print("Failed")
        print("Unit costs differ by: {}".format(abs(pylens_unit_cost - clens_unit_cost)))
        return False
    print("passed")
    return True


def test_xor(parallel=False):
    result = xor_runner(parallel)
    if result:
        x = 1
    else:
        x = 0

    # assert x == 1

if __name__ == '__main__':
    test_xor()
    test_xor(parallel=True)

# if result:
#     print("Final Passed")
#     exit(0)
# else:
#     print("Final Failed")
#     exit(1)
