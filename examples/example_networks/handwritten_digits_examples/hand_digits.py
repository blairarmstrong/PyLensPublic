# Sovling a recognition task of handwritten digits
from PyLens.simulator import Simulator
sim_one = Simulator(name="simulator")

digits_net = sim_one.create_net(name='digits')

digits_net.add_group(
        64,
        name="input",
        group_type="input",
        num_cols=8,
)
digits_net.add_group(
        20,
        name="hidden",
        group_type="hidden",
)
digits_net.add_group(
        10,
        name="output",
        group_type="output",
        output_transforms=['soft_max'],
        error_function="cross_entropy"
)

digits_net.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
)
digits_net.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
)

digits_net.load_example_set("./examples/lens_example_input/hand-digits.trn.ex", "hand-digits.trn")
digits_net.load_example_set("./examples/lens_example_input/hand-digits.tst.ex", "hand-digits.tst", training=False, testing=True)

digits_net.set_update_method(
        lr = 0.2,
        update_method = "adam"
)

digits_net.train(
        100,
        batch_size = 0,
        report_interval = 10
)
# sim_one.use_gui(digits_net)
