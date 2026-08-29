# A continuous network thrown at XOR task
from PyLens.simulator import Simulator

batch_size = 0
num_updates = 100
report_interval = 10
learning_rate = 0.5
update_method = "steepest"

sim_one = Simulator(name="simulator")

xor_net_one = sim_one.create_net(name='xor', time_intervals=4, ticks_per_interval=5 ,type='continuous')

xor_net_one.add_group(
    2,
    name="input",
    group_type="input",
    input_transforms=[],
    output_transforms=[]
)

xor_net_one.add_group(
    2,
    name="hidden",
    group_type="hidden",
    input_transforms=["dot"], #, "in_integr"
    output_transforms=["sigmoid", "out_integr"] #
)

xor_net_one.add_group(
    1,
    name="output",
    group_type="output",
    input_transforms=["dot"],
    output_transforms=["sigmoid","out_integr"],
    error_function="cross_entropy"
)

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



xor_net_one.load_example_set("examples/example_networks/continuous_xor_examples/xor.ex")
xor_net_one.load_clen_weight("examples/example_networks/continuous_xor_examples/xor_weights.wt")

xor_net_one.set_update_method(learning_rate, update_method)

xor_net_one.train(num_updates, batch_size, 1)
sim_one.use_gui(xor_net_one)
