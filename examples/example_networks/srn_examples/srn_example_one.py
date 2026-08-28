# Example of simple recurrent network
from PyLens.simulator import Simulator

sim = Simulator(name="simulator")
srn = sim.create_net(name="srn", time_intervals=2)

srn.set_update_method(0.1, "steepest")

srn.add_group(
    2, name="input", group_type="input", input_transforms=[], output_transforms=[]
)
srn.add_group(
    2,
    name="elman",
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
    link_type="uniform",
    proj_type="full",
    lesion_rate=0,
    dropout_rate=0,
    perma_lesion_rate=0,
)
srn.connect_groups(
    outgoing_group="elman",
    incoming_group="hidden",
    link_type="uniform",
    proj_type="full",
)
srn.connect_groups(
    outgoing_group="hidden",
    incoming_group="output",
    link_type="uniform",
    proj_type="full",
)

srn.connect_groups(
    outgoing_group="hidden",
    incoming_group="elman",
    link_type="elman",
    proj_type="one-to-one",
)

# Disable plots
srn.toggle_plots(plots=False)

# load example set
srn.load_example_set("examples/example_networks/srn_examples/examples.ex")
srn.load_clen_weight("examples/example_networks/srn_examples/srn.wt")

srn.train(2, 0, report_interval=1)
sim.use_gui(srn)
