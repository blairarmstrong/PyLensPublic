# simple recurrent network with two elman layers
from PyLens.simulator import Simulator

sim = Simulator(name="simulator")
srn = sim.create_net(name="srn", time_intervals=4)

srn.set_update_method(0.1, "steepest")

srn.add_group(
    2, name="input", group_type="input", input_transforms=[], output_transforms=[]
)
# !!! Note that the order of adding elman network is critical elman1 then elman2.
srn.add_group(
    2,
    name="elman1",
    group_type="elman",
    input_transforms=[],
    output_transforms=["sigmoid"],
)
srn.add_group(
    2,
    name="elman2",
    group_type="elman",
    input_transforms=[],
    output_transforms=["sigmoid"],
)
srn.add_group(
    2,
    name="hidden",
    group_type="hidden",
    input_transforms=["dot"],
    output_transforms=["sigmoid"],
    lesion_rate=0,
    dropout_rate=0,
)
srn.add_group(
    2,
    name="output",
    group_type="output",
    input_transforms=["dot"],
    output_transforms=["sigmoid"],
    error_function="cross_entropy",
)

srn.connect_groups(
    outgoing_group="input",
    incoming_group="hidden",
    initialization="uniform",
    proj_type="full",
    lesion_rate=0,
    dropout_rate=0,
    perma_lesion_rate=0,
)
srn.connect_groups(
    outgoing_group="elman2",
    incoming_group="hidden",
    initialization="uniform",
    proj_type="full",
)
srn.connect_groups(
    outgoing_group="hidden",
    incoming_group="output",
    initialization="uniform",
    proj_type="full",
)

srn.connect_groups(
    outgoing_group="hidden",
    incoming_group="elman1",
    proj_type="elman",
)
srn.connect_groups(
    outgoing_group="elman1",
    incoming_group="elman2",
    proj_type="elman",
)

# Disable plots
srn.toggle_plots(plots=False)

# load example set
srn.load_example_set("examples/example_networks/SRN_examples/examples_double.ex")
srn.load_clen_weight("examples/example_networks/SRN_examples/srn.wt")

srn.train(10, 0, report_interval=1)
sim.use_gui(srn)
