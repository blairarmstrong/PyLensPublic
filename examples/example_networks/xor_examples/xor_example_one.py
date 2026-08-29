# A classic network to solve XOR
from PyLens.simulator import Simulator

# Create simulator
sim = Simulator(name="sim_xor")

# Create network
xor_net_one = sim.create_net(name="xor")

# Add the input layer (2 units)
xor_net_one.add_group(
    2,
    name="input",
    group_type="input",
    input_transforms=[],
    output_transforms=[]
)

# Add the hidden layer (2 units, sigmoid activation)
xor_net_one.add_group(
    2,
    name="hidden",
    group_type="hidden",
    input_transforms=["dot"],
    output_transforms=["sigmoid"]
)

# Add the output layer (1 unit, sigmoid + cross_entropy error)
xor_net_one.add_group(
    1,
    name="output",
    group_type="output",
    input_transforms=["dot"],
    output_transforms=["sigmoid"],
    error_function="cross_entropy"
)

# Connect layers (full projection)
xor_net_one.connect_groups(
    outgoing_group="input",
    incoming_group="hidden",
    initialization="uniform",
    proj_type="full"
)

xor_net_one.connect_groups(
    outgoing_group="hidden",
    incoming_group="output",
    initialization="uniform",
    proj_type="full"
)

xor_net_one.load_example_set("examples/lens_example_input/xor_dense.ex")
xor_net_one.load_clen_weight("examples/example_networks/xor_examples/xor_example_one_weights.wt")

# Set hyperparameters
batch_size = 0
num_updates = 100 
report_interval = 1
learning_rate = 0.5
update_method = "momentum"

xor_net_one.set_update_method(learning_rate, update_method)

xor_net_one.train(num_updates, batch_size, report_interval)
sim.use_gui(xor_net_one)
