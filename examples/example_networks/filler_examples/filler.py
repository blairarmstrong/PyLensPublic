# This is a memory test. The network is given a sequence of five patterns. Each pattern has one in four bits on. The network must remember all of the patterns it's seen and reproduce them after each step. The output is a slot-filler representation. The first pattern goes in the top slot, the second pattern in the next slot and so on. Along with just remembering the patterns, the network must keep track of where it is in the sequence so the next pattern can go in the right place. 
from PyLens.simulator import Simulator

sim = Simulator(name="simulator")
filler_net = sim.create_net(name="filler", time_intervals=5)
filler_net.set_update_method(0.01, "adam")
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

filler_net.connect_groups(
    outgoing_group="chars",
    incoming_group="hidden",
    initialization="uniform",
    proj_type="full",
    lesion_rate=0,
    dropout_rate=0,
    perma_lesion_rate=0,
)
filler_net.connect_groups(
    outgoing_group="elman",
    incoming_group="hidden",
    initialization="uniform",
    proj_type="full",
)
filler_net.connect_groups(
    outgoing_group="hidden",
    incoming_group="output",
    initialization="uniform",
    proj_type="full",
)

filler_net.connect_groups(
    outgoing_group="hidden",
    incoming_group="elman",
    proj_type="elman",
)

# Disable plots
filler_net.toggle_plots(plots=False)

# load example set
filler_net.load_example_set(
    "examples/lens_example_input/filler.ex"
)
filler_net.load_clen_weight("examples/example_networks/filler_examples/filler_w2.wt")

filler_net.train(10, batch_size=0, report_interval=1)
sim.use_gui(filler_net)
