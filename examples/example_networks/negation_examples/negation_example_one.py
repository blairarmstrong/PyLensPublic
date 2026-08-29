# This is the "negation" problem introduced on page 346 of Parallel Distributed Processing, Explorations in the Microstructure of Cognition, Vol. 1, edited by David Rumelhart and James McClelland. 
#  If the first bit of the input is off, the output should be the same as the last three bits of the input. But if the first bit is on, the output should be the complement of the first three bits. 
from PyLens.simulator import Simulator

updates = 100
batch_size = 0
report_interval = 10
learning_rate = 0.1
update_method = "steepest"

sim = Simulator(name="simulator")

negation_net = sim.create_net(name="negation_net")

negation_net.add_group(
    4,
    name="input",
    group_type="input",
    input_transforms=[],
    output_transforms=[])

negation_net.add_group(
    3, name="hidden",
    group_type="hidden",
    input_transforms=["dot"],
    output_transforms=["sigmoid"])

negation_net.add_group(
    3,
    name="output",
    group_type="output",
    input_transforms=["dot"],
    output_transforms=["sigmoid"],
    error_function="cross_entropy")

negation_net.connect_groups(
    outgoing_group="input",
    incoming_group="hidden",
    initialization="uniform",
    proj_type="full")

negation_net.connect_groups(
    outgoing_group="hidden",
    incoming_group="output",
    initialization="uniform",
    proj_type="full")

negation_net.connect_groups(
    outgoing_group="input",
    incoming_group="output",
    initialization="uniform",
    proj_type="full")


negation_net.load_example_set("./examples/lens_example_input/negation.ex")

negation_net.load_clen_weight("./examples/example_networks/negation_examples/negation_example_one_weights.wt")

negation_net.set_update_method(learning_rate, update_method)

negation_net.train(updates, batch_size, report_interval)
sim.use_gui(negation_net)
