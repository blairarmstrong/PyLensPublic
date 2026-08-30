# These Boltzmann machines are used to perform the pattern completion task on
# boltz.ex data. 
from PyLens.simulator import Simulator

num_updates = 100
batch_size = 0
report_interval = 10
learning_rate = 0.1
update_method = "steepest"

sim_one = Simulator(name="simulator")

reconstruction_net_one = sim_one.create_net(name='dbm', time_intervals=4,
                                 ticks_per_interval=10, type='boltzmann')
reconstruction_net_one.plot = False

reconstruction_net_one.add_group(
    20,
    name="input",
    group_type="input",
    num_cols=4,
    input_transforms=["boltzmann"],
    output_transforms=["boltzmann"],
)

reconstruction_net_one.add_group(
    20,
    name="output",
    group_type="output",
    num_cols=4
)

reconstruction_net_one.connect_groups(
    outgoing_group="input",
    incoming_group="input",
    initialization="uniform",
    proj_type="full",
)

reconstruction_net_one.connect_groups(
    outgoing_group="input",
    incoming_group="output",
    initialization="uniform",
    proj_type="full",
    bidirectional=True
)

reconstruction_net_one.connect_groups(
    outgoing_group="output",
    incoming_group="output",
    initialization="uniform",
    proj_type="full",
)


# set network parameters
reconstruction_net_one.group_criterion_threshold = 0.001
reconstruction_net_one.test_group_criterion_threshold = 0.001
reconstruction_net_one.clamp_strength = 1.0
reconstruction_net_one.init_gain = 0.1
reconstruction_net_one.final_gain = 1.0
reconstruction_net_one.anneal_time = 1.0

# load example set
reconstruction_net_one.load_example_set("./examples/lens_example_input/boltz.ex")
reconstruction_net_one.training_sets[0].max_time = 3.0
reconstruction_net_one.training_sets[0].min_time = 0.0
reconstruction_net_one.training_sets[0].grace_time = 1.0
# load sample weights from clens random weight seed
reconstruction_net_one.load_clen_weight("./examples/example_networks/boltzmann_examples/boltz_complete2.wt")

reconstruction_net_one.set_update_method(learning_rate, update_method)

reconstruction_net_one.train(100, batch_size, report_interval)

sim_one.use_gui(reconstruction_net_one)


