# A simple recurrent network that learns to add base-10 numbers.
from PyLens.simulator import Simulator

sim = Simulator(name="simulator")
adder_net = sim.create_net(name="adder10", time_intervals=7)
adder_net.set_update_method(0.02, "adam")

adder_net.add_group(20,
                     name="input",
                     group_type="input",
                     input_transforms=[],
                     output_transforms=[],
                     num_cols=10)

adder_net.add_group(12,
                     name="elman",
                     group_type="elman",
                     input_transforms=[],
                     output_transforms=["sigmoid"])

adder_net.add_group(12,
                     name="hidden",
                     group_type="hidden",
                     input_transforms=["dot"],
                     output_transforms=["sigmoid"],
                     dropout_rate=0)

adder_net.add_group(10,
                     name="output",
                     group_type="output",
                     input_transforms=["dot"],
                     output_transforms=["soft_max"],
                     error_function="divergence")


adder_net.connect_groups(outgoing_group="elman",
                          incoming_group="hidden",
                          link_type="uniform",
                          proj_type="full")

adder_net.connect_groups(outgoing_group="input",
                          incoming_group="hidden",
                          link_type="uniform",
                          proj_type="full")

adder_net.connect_groups(outgoing_group="hidden",
                          incoming_group="output",
                          link_type="uniform",
                          proj_type="full")

adder_net.connect_groups(outgoing_group="hidden",
                          incoming_group="elman",
                          link_type="elman",
                          proj_type="one-to-one")

# load example set
adder_net.load_example_set("examples/example_networks/adder_examples/adder10.ex")
adder_net.load_clen_weight("examples/example_networks/adder_examples/adder10_weights.wt")

adder_net.train(100, batch_size=0, report_interval=10)
sim.use_gui(adder_net)
