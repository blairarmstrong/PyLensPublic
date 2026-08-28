# A "big" deterministic Boltzmann machine is trained to solve the simple digits task.
from PyLens.simulator import Simulator

def boltz_big_unit_test(parallel=False):
    # Run to Test Example pylens error against clens error
    # STEEPEST, 100 UPDATES
    num_updates = 100
    batch_size = 0
    report_interval = 1
    learning_rate = 0.1
    update_method = "steepest"

    # create simulator
    sim_one = Simulator(name="simulator", baseType='numpy')

    # create sample network
    digits_net_one = sim_one.create_net(name='dbm', time_intervals=4,
                                    ticks_per_interval=5, type='boltzmann')
    digits_net_one.plot = False

    # create input layer with 20 units
    digits_net_one.add_group(
        20,
        name="input",
        group_type="input",
    )

    # create hidden layer with 10 units
    digits_net_one.add_group(
        10,
        name="hidden",
        group_type="hidden",
    )

    # create output layer with 3 units; cross entropy error is standard for BMs
    digits_net_one.add_group(
        3,
        name="output",
        group_type="output",
    )

    # Connect the groups:
    # connects the input layer to the hidden layer; uniform links; full projection; bidirectional
    digits_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full",
        bidirectional=True
    )
    # connects the hidden layer laterally; uniform links; full projection
    digits_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full"
    )
    # connects the hidden layer to the output layer; uniform links; full projection; bidirectional
    digits_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        link_type="uniform",
        proj_type="full",
        bidirectional=True
    )
    # connects the output layer laterally; uniform links; full projection
    digits_net_one.connect_groups(
        outgoing_group="output",
        incoming_group="output",
        link_type="uniform",
        proj_type="full"
    )

    digits_net_one.load_example_set("./examples/lens_example_input/digits.ex")

    digits_net_one.set_properties(
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
    # # set network parameters
    # digits_net_one.group_criterion_threshold = 0.001
    # digits_net_one.test_group_criterion_threshold = 0.001
    # digits_net_one.clamp_strength = 1.0
    # digits_net_one.init_gain = 0.1
    # digits_net_one.final_gain = 1.0
    # digits_net_one.anneal_time = 1.0

    # # load example set
    # digits_net_one.training_sets[0].max_time = 3.0
    # digits_net_one.training_sets[0].min_time = 0.0
    # digits_net_one.training_sets[0].grace_time = 1.0
    # load sample weights from clens random weight seed
    digits_net_one.load_clen_weight("./examples/example_networks/boltzmann_examples/boltz_big.wt")

    # set learning rate and update method
    digits_net_one.set_update_method(learning_rate, update_method)

    # set epochs=100, batch size=0, report interval=10
    if parallel == True:
        digits_net_one.train(num_updates, batch_size, report_interval, parallel_mode=True, num_worker=3)
    else:
        digits_net_one.train(num_updates, batch_size, report_interval)

    # to use GUI, uncomment
    # sim_one.use_gui(digits_net_one)

    actual_weight_cost = round(float(digits_net_one.stats_plotter.progress_stats['weight_cost'][-1]), 2)
    actual_error = round(float(digits_net_one.stats_plotter.progress_stats['error'][-1]), 2)
    
    expected_weight_cost = round(721.033, 2)
    expected_error = round(1.88509, 2)

    assert actual_error == expected_error
    assert actual_weight_cost == expected_weight_cost

if __name__ == '__main__':
    boltz_big_unit_test()
    boltz_big_unit_test(parallel=True)
