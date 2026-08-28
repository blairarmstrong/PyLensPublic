# These Boltzmann machines are used to perform the pattern completion task on
# boltz.ex data.
from PyLens.simulator import Simulator
from test_utils import *

def boltz_complete2_unit_test(parallel=False):
    # Run to Test Example pylens error against clens error
    # STEEPEST, 100 UPDATES
    num_updates = 10
    batch_size = 0
    report_interval = 10
    learning_rate = 0.1
    update_method = "steepest"

    # create simulator
    sim_one = Simulator(name="simulator")

    # create sample network
    reconstruction_net_one = sim_one.create_net(name='dbm', time_intervals=4,
                                    ticks_per_interval=10, type='boltzmann')

    # create input layer with 20 units
    reconstruction_net_one.add_group(
        20,
        name="input",
        group_type="input",
        input_transforms=["boltzmann"],
        output_transforms=["boltzmann"]
    )

    # create output layer with 20 units
    reconstruction_net_one.add_group(
        20,
        name="output",
        group_type="output",
    )


    # connect input layer laterally
    reconstruction_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="input",
        link_type="uniform",
        proj_type="full",
    )

    # connects the input layer to the output layer; uniform links; full projection; bidirectional
    reconstruction_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="output",
        link_type="uniform",
        proj_type="full",
        bidirectional=True
    )

    # connect input layer laterally
    reconstruction_net_one.connect_groups(
        outgoing_group="output",
        incoming_group="output",
        link_type="uniform",
        proj_type="full",
    )

    reconstruction_net_one.load_example_set("./examples/lens_example_input/boltz.ex")
    # set network and example parameters
    reconstruction_net_one.set_properties(
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
    # reconstruction_net_one.group_criterion_threshold = 0.001
    # reconstruction_net_one.test_group_criterion_threshold = 0.001
    # reconstruction_net_one.clamp_strength = 1.0
    # reconstruction_net_one.init_gain = 0.1
    # reconstruction_net_one.final_gain = 1.0
    # reconstruction_net_one.anneal_time = 1.0

    # # load example set
    # reconstruction_net_one.training_sets[0].max_time = 3.0
    # reconstruction_net_one.training_sets[0].min_time = 0.0
    # reconstruction_net_one.training_sets[0].grace_time = 1.0
    # load sample weights from clens random weight seed
    reconstruction_net_one.load_clen_weight("./examples/example_networks/boltzmann_examples/boltz_complete2.wt")

    # set learning rate and update method
    reconstruction_net_one.set_update_method(learning_rate, update_method)

    # set epochs=100, batch size=0, report interval=10
    if parallel == True:
        reconstruction_net_one.train(num_updates, batch_size, report_interval, parallel_mode=True, num_worker=4)
    else:
        reconstruction_net_one.train(num_updates, batch_size, report_interval)

    # to use GUI, uncomment
    # sim_one.use_gui(reconstruction_net_one)

    actual_weight_cost = reconstruction_net_one.stats_plotter.progress_stats['weight_cost'][-1]
    actual_error = reconstruction_net_one.stats_plotter.progress_stats['error'][-1]
    sim_one.delete_all_nets()
    
    expected_weight_cost = 1131.39
    check_difference(actual_weight_cost, expected_weight_cost)

    expected_error = 29.4884
    check_difference(actual_error, expected_error)

if __name__ == '__main__':
    boltz_complete2_unit_test()
    boltz_complete2_unit_test(parallel=True)
