from PyLens.simulator import Simulator

batch_size = 0
num_updates = 2
report_interval = 1
learning_rate = 0.01
update_method = "dougs momentum"

sim_one = Simulator(name="simulator")

task_net_one = sim_one.create_net(name="task")

task_net_one.add_group(
    3,
    name="task",
    group_type="input",
    input_transforms=[],
    output_transforms=[]
)

task_net_one.add_group(
    6,
    name="taskHidden",
    group_type="hidden",
    input_transforms=["dot"],
    output_transforms=["sigmoid"]
)

task_net_one.add_group(
    6,
    name="input",
    group_type="input",
    input_transforms=["dot"],
    output_transforms=["sigmoid"]
)

task_net_one.add_group(
    6,
    name="hidden",
    group_type="hidden",
    input_transforms=["product"],
    output_transforms=["sigmoid"]
)

task_net_one.add_group(
    2,
    name="output",
    group_type="output",
    input_transforms=["dot"],
    output_transforms=["sigmoid"]
)


task_net_one.connect_groups(
    outgoing_group="task",
    incoming_group="taskHidden",
    link_type="uniform",
    proj_type="full"
)

task_net_one.connect_groups(
    outgoing_group="taskHidden",
    incoming_group="hidden",
    link_type="uniform",
    proj_type="one-to-one"
)

task_net_one.connect_groups(
    outgoing_group="input",
    incoming_group="hidden",
    link_type="uniform",
    proj_type="one-to-one"
)

task_net_one.connect_groups(
    outgoing_group="hidden",
    incoming_group="output",
    link_type="uniform",
    proj_type="full"
)

task_net_one.load_example_set("./examples/lens_example_input/task.ex")

task_net_one.load_clen_weight("./examples/example_networks/task_examples/task_example_one_weights.wt")

task_net_one.set_update_method(learning_rate, update_method)

sim_one.use_gui(task_net_one)
