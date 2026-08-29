# This is a random-mapping task, but a CONTINUOUS network has been thrown at it here
from PyLens.simulator import Simulator

learning_rate = 0.5
update_method = "dougs momentum"

sim_one = Simulator(name="simulator")

rand10x40 = sim_one.create_net(name='rand', time_intervals=4, ticks_per_interval=5, type='continuous')

rand10x40.add_group(
    10,
    name="input",
    group_type="input",
    input_transforms=[],
    output_transforms=[],
)

rand10x40.add_group(
    50,
    name="hidden",
    group_type="hidden",
    input_transforms=["dot", "in_integr"],
    output_transforms=["sigmoid",] 
)

rand10x40.add_group(
    10,
    name="output",
    group_type="output",
    input_transforms=["dot"],
    output_transforms=["sigmoid", "out_integr"],
    error_function="cross_entropy"
)

rand10x40.connect_groups(
    outgoing_group="input",
    incoming_group="hidden",
    initialization="uniform",
    proj_type="full"
)

rand10x40.connect_groups(
    outgoing_group="output",
    incoming_group="hidden",
    initialization="uniform",
    proj_type="full"
)


rand10x40.connect_groups(
    outgoing_group="hidden",
    incoming_group="output",
    initialization="uniform",
    proj_type="full"
)

rand10x40.load_example_set("./examples/example_networks/rand10x40_examples/rand10x40.ex")
rand10x40.training_sets[0].max_time=2
rand10x40.load_clen_weight("./examples/example_networks/rand10x40_examples/rand10x40_weights.wt")

rand10x40.set_update_method(learning_rate, update_method)
sim_one.use_gui(rand10x40)
