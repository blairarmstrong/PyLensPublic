# Sovling a digit recognition task 
from PyLens.simulator import Simulator
sim_one = Simulator(name="simulator")

digits_net = sim_one.create_net(name='digits')

digits_net.add_group(
        20,
        name="input",
        group_type="input",
        num_cols=4,
)
digits_net.add_group(
        20,
        name="hidden",
        group_type="hidden",
)
digits_net.add_group(
        20,
        name="hidden2",
        group_type="hidden",
        output_transforms=['sigmoid', 'noise'],
        unit_cost_function='cosine',
)
digits_net.add_group(
        3,
        name="output",
        group_type="output",
)

digits_net.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        proj_type='random',
        perma_lesion_rate=0.5
)
digits_net.connect_groups(
        outgoing_group="hidden",
        incoming_group="hidden2",
)
digits_net.connect_groups(
        outgoing_group="hidden2",
        incoming_group="output",
)

digits_net.load_example_set("./examples/lens_example_input/digits.ex", "digits")
digits_net.load_example_set("./examples/lens_example_input/digits2.ex", "digits_noise", training=False, testing=True)
digits_net.load_clen_weight("examples/example_networks/digits_examples/digits_random_connection.wt")

digits_net.set_update_method(
        lr = 0.1,
        update_method = "dougs momentum"
)

digits_net.train(
        100,
        batch_size = 0,
        report_interval = 1
)
# sim_one.use_gui(digits_net)

