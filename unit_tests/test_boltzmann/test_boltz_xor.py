from PyLens.simulator import Simulator
from test_utils import *

def boltz_xor_unit_test(parallel=False):
    # Run to Test Example pylens error against clens error
    # STEEPEST, 100 UPDATES
    num_updates = 100
    batch_size = 0
    report_interval = 1
    learning_rate = 0.1
    update_method = "steepest"

    # create simulator
    sim_one = Simulator(name="simulator")

    # create sample network
    xor_net_one = sim_one.create_net(name='dbm', time_intervals=4,
                                    ticks_per_interval=5, type='boltzmann')
    xor_net_one.plot = False

    # create input layer with 2 units
    xor_net_one.add_group(
        2,
        name="input",
        group_type="input",
    )

    # create hidden layer with 2 units
    xor_net_one.add_group(
        2,
        name="hidden",
        group_type="hidden",
    )

    # create output layer with 1 units; cross entropy error is standard for BMs
    xor_net_one.add_group(
        1,
        name="output",
        group_type="output",
    )

    # Connect the groups:
    # connects the input layer to the hidden layer; uniform links; full projection; bidirectional
    xor_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full",
    )
    xor_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="input",
        link_type="uniform",
        proj_type="full"
    )
    # connects the hidden layer laterally; uniform links; full projection
    xor_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full"
    )
    # connects the hidden layer to the output layer; uniform links; full projection; bidirectional
    xor_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        link_type="uniform",
        proj_type="full",
    )
    xor_net_one.connect_groups(
        outgoing_group="output",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full",
    )
    # connects the output layer laterally; uniform links; full projection
    xor_net_one.connect_groups(
        outgoing_group="output",
        incoming_group="output",
        link_type="uniform",
        proj_type="full"
    )

    xor_net_one.load_example_set("./examples/lens_example_input/xor_dense.ex")

    # set network and example parameters
    xor_net_one.set_properties(
        group_criterion_threshold=0.001,
        test_group_criterion_threshold=0.001,
        clamp_strength=1.0,
        init_gain=0.1,
        final_gain=1.0,
        anneal_time=1.0,
        training_sets=[
            {"max_time": 3.0, "min_time": 0.0, "grace_time": 1.0}
        ]
    )

    # load sample weights from clens random weight seed
    xor_net_one.load_clen_weight("./examples/example_networks/boltzmann_examples/boltz_xor.wt")

    # set learning rate and update method
    xor_net_one.set_update_method(learning_rate, update_method)


    # set epochs=100, batch size=0, report interval=10
    if parallel == True:
        xor_net_one.train(num_updates, batch_size, report_interval=report_interval, parallel_mode=True, num_worker=3)
    else:
        xor_net_one.train(num_updates, batch_size, report_interval=report_interval)

    # to use GUI, uncomment
    # sim_one.use_gui(xor_net_one)

    check_reports(xor_net_one)


    """
    clens errors round to 3 sig digs :  
    boltz_xor:
    error = 0.55894
    weight_cost = 33.1917

    boltz_big:
    error = 1.88509
    weight_cost = 721.033


    """


    actual_weight_cost = xor_net_one.stats_plotter.progress_stats['weight_cost'][-1]
    actual_error = xor_net_one.stats_plotter.progress_stats['error'][-1]
    sim_one.delete_all_nets()
    
    expected_weight_cost = 33.1917
    check_difference(actual_weight_cost, expected_weight_cost)
    
    expected_error = 0.55894
    check_difference(actual_error, expected_error)

if __name__ == '__main__':
    boltz_xor_unit_test()
    boltz_xor_unit_test(parallel=True)
