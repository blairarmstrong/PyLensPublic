# A "big" (including a hidden layer) deterministic Boltzmann machine is trained to solve the simple digits task.
from PyLens.simulator import Simulator

num_updates = 100
batch_size = 0
report_interval = 10
learning_rate = 0.1
update_method = "steepest"

sim_one = Simulator(name="simulator")

digits_net_one = sim_one.create_net(name='dbm', time_intervals=4,
                                 ticks_per_interval=5, type='boltzmann')
digits_net_one.plot = False

digits_net_one.add_group(
    20,
    name="input",
    group_type="input",
    num_cols=4,
)

digits_net_one.add_group(
    10,
    name="hidden",
    group_type="hidden",
)

digits_net_one.add_group(
    3,
    name="output",
    group_type="output",
)

digits_net_one.connect_groups(
    outgoing_group="input",
    incoming_group="hidden",
    initialization="uniform",
    proj_type="full",
    bidirectional=True
)

digits_net_one.connect_groups(
    outgoing_group="hidden",
    incoming_group="hidden",
    initialization="uniform",
    proj_type="full"
)

digits_net_one.connect_groups(
    outgoing_group="hidden",
    incoming_group="output",
    initialization="uniform",
    proj_type="full",
    bidirectional=True
)

digits_net_one.connect_groups(
    outgoing_group="output",
    incoming_group="output",
    initialization="uniform",
    proj_type="full"
)

# set network parameters
digits_net_one.group_criterion_threshold = 0.001
digits_net_one.test_group_criterion_threshold = 0.001
digits_net_one.clamp_strength = 1.0
digits_net_one.init_gain = 0.1
digits_net_one.final_gain = 1.0
digits_net_one.anneal_time = 1.0

# load example set
digits_net_one.load_example_set("./examples/lens_example_input/digits.ex")
digits_net_one.training_sets[0].max_time = 3.0
digits_net_one.training_sets[0].min_time = 0.0
digits_net_one.training_sets[0].grace_time = 1.0

# load sample weights from clens random weight seed
digits_net_one.load_clen_weight("./examples/example_networks/boltzmann_examples/boltz_big.wt")

# set learning rate and update method
digits_net_one.set_update_method(learning_rate, update_method)

digits_net_one.train(num_updates, batch_size, report_interval)

