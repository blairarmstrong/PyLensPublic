# This is the classic encoder task, in which the network must map its input to its output through a constriction. This forces the network to use a compressed representation of the pattern at its hidden layer. 
from PyLens.simulator import Simulator

batch = 0
num_updates = 100
report_interval = 10
learning_rate = 0.1
weight_decay = 0.0001
zero_error_radius = 0.01
method = "steepest"

hidden_size = 3

sim = Simulator(name="simulator")

encoder = sim.create_net(name="encoder")

encoder.set_update_method(learning_rate, method)

encoder.optimizer.weight_decay = weight_decay
encoder.optimizer.optimizer_params.PAR_O_zeroErrorRadius = zero_error_radius

encoder.add_group(8,
                  name="input",
                  group_type="input",
                  input_transforms=[],
                  output_transforms=[])
encoder.add_group(hidden_size,
                  name="hidden",
                  group_type="hidden",
                  input_transforms=["dot"],
                  output_transforms=["sigmoid"])
encoder.add_group(8,
                  name="output",
                  group_type="output",
                  input_transforms=["dot"],
                  output_transforms=["sigmoid"],
                  error_function="cross_entropy")
encoder.connect_groups(outgoing_group="input",
                       incoming_group="hidden",
                       initialization="uniform",
                       proj_type="full")
encoder.connect_groups(outgoing_group="hidden",
                       incoming_group="output",
                       initialization="uniform",
                       proj_type="full")

encoder.load_example_set("examples/lens_example_input/encoder.sparse.ex")
encoder.load_clen_weight("examples/example_networks/encoder_examples/encoder_example_one_weights.wt")

encoder.train(epochs=num_updates, batch_size=batch, report_interval=report_interval)
sim.use_gui(encoder)
