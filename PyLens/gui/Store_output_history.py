from PyLens.backend.network import Network
from PyLens.gui.unit_viewer_tk import FrameExamplesProgram
from PyLens.simulator import Simulator
import pickle

def store_output_history(simulator, net):
    history = simulator.generate_constant_output_history(training_net)
    with open('history.pkl', 'wb') as output:
        pickle.dump(history, output, pickle.HIGHEST_PROTOCOL)
    output.close()

if __name__ == "__main__":
    sim = Simulator(name="my simulator ")
    training_net = sim.create_net(name="train", learning_rate=0.2)
    training_net.add_group(name="first", num_units=182, group_type="input", input_transforms=[], output_transforms=[])
    training_net.add_group(name="second", num_units=122, group_type="hidden", input_transforms=["dot"],
                           output_transforms=["sigmoid"])
    training_net.add_group(name="third", num_units=3698, group_type="output", input_transforms=["dot"],
                           output_transforms=["sigmoid"])
    training_net.connect_groups(outgoing_group="first", incoming_group="second", initialization="uniform")
    training_net.connect_groups(outgoing_group="second", incoming_group="third", initialization="uniform")
    training_net.load_example_set(
        example_set_name="TRAINING_EX", proc=False,
        file_name="../exp_english_distribution=uniform_frequencyMinimum=1_dropType=linear_distributionType=probabilistic_dropAmount=0_shuffle=True.ex.txt",
        num_examples_loaded=1)
    training_net.train(5, 0, report_interval=100)

    store_output_history(sim, training_net)

