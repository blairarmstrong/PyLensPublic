# Filler task with srbptt (simple-recurrent-backprop-through-time) network
from PyLens.simulator import Simulator

batch_size = 0
num_updates = 10
report_interval = 1

sim = Simulator(name="simulator")
filler_net = sim.create_net(name="filler", time_intervals=5, type="srbptt")
filler_net.set_update_method(0.05, "adam")
filler_net.add_group(
    4, name="chars", group_type="input", input_transforms=[], output_transforms=[]
)
filler_net.add_group(
    20,
    name="elman",
    group_type="elman",
    input_transforms=[],
    output_transforms=["sigmoid"],
)
filler_net.add_group(
    20,
    name="hidden",
    group_type="hidden",
    input_transforms=["dot"],
    output_transforms=["sigmoid"],
    lesion_rate=0,
    dropout_rate=0,
)
filler_net.add_group(
    20,
    name="output",
    group_type="output",
    input_transforms=["dot"],
    output_transforms=["sigmoid"],
    error_function="cross_entropy",
)

# Apparently this order matters for srbptt
# Or else weights get stored in the wrong order
filler_net.connect_groups(
    outgoing_group="elman",
    incoming_group="hidden",
    link_type="uniform",
    proj_type="full",
)
filler_net.connect_groups(
    outgoing_group="chars",
    incoming_group="hidden",
    link_type="uniform",
    proj_type="full",
    lesion_rate=0,
    dropout_rate=0,
    perma_lesion_rate=0,
)
filler_net.connect_groups(
    outgoing_group="hidden",
    incoming_group="output",
    link_type="uniform",
    proj_type="full",
)

filler_net.connect_groups(
    outgoing_group="hidden",
    incoming_group="elman",
    link_type="elman",
    proj_type="one-to-one",
)

# load example set
filler_net.load_example_set(
    "examples/lens_example_input/filler.ex"
)
filler_net.load_clen_weight(
    "examples/example_networks/filler_srbptt_examples/filler_srbptt_w1.wt"
)

filler_net.train(
    epochs=num_updates, batch_size=batch_size, report_interval=report_interval, 
    parallel_mode=True, num_worker=8
)
sim.use_gui(filler_net)
