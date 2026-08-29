# This is a sentence gestalt network based on St. John & McClelland (1990) and O’Reilly et al. (2024) Chapter 10: https://sims.compcogneuro.org/ch10/sg/
from PyLens.simulator import Simulator

batch_size = 0
num_updates = 2000
report_interval = 10
learning_rate = 0.05
update_method = "adam"

sim = Simulator(name="simulator")
sg = sim.create_net(name="sg", time_intervals=20)
sg.set_update_method(learning_rate, update_method)

sg.add_group(50, name="word", group_type="input", input_transforms=[], output_transforms=[])
sg.add_group(100, name="previous_gestalt", group_type="elman", input_transforms=[], output_transforms=["sigmoid"])
sg.add_group(100, name="hidden1",  group_type="hidden", input_transforms=["dot"], output_transforms=["sigmoid"])
sg.add_group(100, name="gestalt",  group_type="hidden", input_transforms=["dot"], output_transforms=["sigmoid"])

sg.add_group(9, name="role", group_type="input", input_transforms=[], output_transforms=[])
sg.add_group(100, name="hidden2",  group_type="hidden", input_transforms=["dot"],output_transforms=["sigmoid"])
sg.add_group(55, name="filler", group_type="output", input_transforms=["dot"], output_transforms=["sigmoid"], error_function="cross_entropy")

# Apparently this order matters for recurrent network
# Or else weights get stored in the wrong order
sg.connect_groups(outgoing_group="word", incoming_group="hidden1", initialization="uniform", proj_type="full")
sg.connect_groups(outgoing_group="previous_gestalt", incoming_group="hidden1", initialization="uniform", proj_type="full")
sg.connect_groups(outgoing_group="hidden1", incoming_group="gestalt", initialization="uniform", proj_type="full")
sg.connect_groups(outgoing_group="gestalt", incoming_group="previous_gestalt", proj_type="elman")

sg.connect_groups(outgoing_group="gestalt", incoming_group="hidden2", initialization="uniform", proj_type="full")
sg.connect_groups(outgoing_group="role", incoming_group="hidden2", initialization="uniform", proj_type="full")
sg.connect_groups(outgoing_group="gestalt", incoming_group="filler", initialization="uniform", proj_type="full")

sg.load_example_set("examples/example_networks/sentence_gestalt_examples/sentence_gestalt.ex")
sg.load_example_set("examples/example_networks/sentence_gestalt_examples/sentence_gestalt_test.ex", training=False, testing=True)

sg.test(10)
sg.train(epochs=num_updates, batch_size=batch_size, report_interval=report_interval, parallel_mode=True, num_worker=3)
sg.test(10)
sim.use_gui(sg)
