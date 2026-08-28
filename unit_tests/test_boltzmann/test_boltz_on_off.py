# Multi Event on_off example with a Boltzmann Machine.
# In the "on" example set, the network is first given an input and must activate the correct output. 
# In the second event, the input is removed, but the output must be maintained.
from PyLens.simulator import Simulator

# Run to Test Example pylens error against clens error
# STEEPEST, 100 UPDATES
def boltz_multi_event_unit_test():
    num_updates = 100
    batch_size = 0
    report_interval = 10
    learning_rate = 0.1
    update_method = "steepest"

    # create simulator
    sim_one = Simulator(name="simulator")

    # create sample network
    on_off_net = sim_one.create_net(name='dbm', time_intervals=4,
                                    ticks_per_interval=10, type='boltzmann')
    on_off_net.plot = False

    # create input layer with 2 units
    on_off_net.add_group(
        6,
        name="input",
        group_type="input",
        input_transforms=["dot", "in_integr"],
        biased=True
    )

    # create hidden layer with 2 units
    on_off_net.add_group(
        10,
        name="hidden",
        group_type="hidden",
        input_transforms=["in_integr"]
    )

    # create output layer with 1 units; cross entropy error is standard for BMs
    on_off_net.add_group(
        6,
        name="output",
        group_type="output",
        input_transforms=["in_integr"]
    )

    # Connect the groups:
    # connects the input layer to the hidden layer; uniform links; full projection; bidirectional
    on_off_net.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full",
        bidirectional=True
    )

    # connects the hidden layer laterally; uniform links; full projection
    on_off_net.connect_groups(
        outgoing_group="hidden",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full"
    )
    # connects the hidden layer to the output layer; uniform links; full projection; bidirectional
    on_off_net.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        link_type="uniform",
        proj_type="full",
        bidirectional=True
    )

    # connects the output layer laterally; uniform links; full projection
    on_off_net.connect_groups(
        outgoing_group="output",
        incoming_group="output",
        link_type="uniform",
        proj_type="full"
    )

    # set network parameters
    on_off_net.group_criterion_threshold = 0.05
    on_off_net.test_group_criterion_threshold = 0.1
    on_off_net.clamp_strength = 0.9
    on_off_net.min_criterion_batches = 5
    on_off_net.init_gain = 1.0
    on_off_net.final_gain = 1.0
    on_off_net.anneal_time = 1.0

    # load example set
    on_off_net.load_example_set("./examples/lens_example_input/on-off2.ex")
    #on_off_net.training_sets[0].max_time = 3.0
    #on_off_net.training_sets[0].min_time = 0.0
    #on_off_net.training_sets[0].grace_time = 1.0

    # load sample weights from clens random weight seed
    on_off_net.load_clen_weight("./examples/example_networks/boltzmann_examples/boltz_on_off.wt")

    # set learning rate and update method
    on_off_net.set_update_method(learning_rate, update_method)

    # set epochs=100, batch size=0, report interval=10
    on_off_net.train(1, batch_size, report_interval)

    # to use GUI, uncomment
    # sim_one.use_gui(on_off_net)

    events = on_off_net.example_sets[0].example[0].event


    assert abs(4.47658-on_off_net.stats_plotter.progress_stats['error'][-1]) < 0.01
    print("passed")

if __name__ == '__main__':
    boltz_multi_event_unit_test()
