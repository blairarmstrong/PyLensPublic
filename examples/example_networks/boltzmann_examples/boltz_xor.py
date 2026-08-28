# The xor problem tackled with a small Boltzmann Machine
from PyLens.simulator import Simulator

num_updates = 100
batch_size = 0
report_interval = 10
learning_rate = 0.1
update_method = "steepest descent"

sim_one = Simulator(name="simulator")

xor_net_one = sim_one.create_net(name='dbm', time_intervals=4,
                                 ticks_per_interval=5, type='boltzmann')

xor_net_one.add_group(
    2,
    name="input",
    group_type="input",
)

xor_net_one.add_group(
    2,
    name="hidden",
    group_type="hidden",
)

xor_net_one.add_group(
    1,
    name="output",
    group_type="output",
)

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
xor_net_one.connect_groups(
    outgoing_group="hidden",
    incoming_group="hidden",
    link_type="uniform",
    proj_type="full"
)
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
xor_net_one.connect_groups(
    outgoing_group="output",
    incoming_group="output",
    link_type="uniform",
    proj_type="full"
)

# set network parameters
xor_net_one.group_criterion_threshold = 0.001
xor_net_one.test_group_criterion_threshold = 0.001
xor_net_one.clamp_strength = 1.0
xor_net_one.init_gain = 0.1
xor_net_one.final_gain = 1.0
xor_net_one.anneal_time = 1.0

# load example set
xor_net_one.load_example_set("./examples/lens_example_input/xor_dense.ex")
xor_net_one.training_sets[0].max_time = 3.0
xor_net_one.training_sets[0].min_time = 0.0
xor_net_one.training_sets[0].grace_time = 1.0

# load sample weights from clens random weight seed
xor_net_one.load_clen_weight("./examples/example_networks/boltzmann_examples/boltz_xor.wt")

# set learning rate and update method
xor_net_one.set_update_method(learning_rate, update_method)

xor_net_one.train(100, batch_size, report_interval)
