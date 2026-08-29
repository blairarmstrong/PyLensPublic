
from PyLens.simulator import Simulator
import sys
import numpy as np

np.random.seed(100)
sys.path.insert(0, ".")

# Run to Test Example pylens error against clens error
# STEEPEST, 100 UPDATES


def test_lesion_hidden_unit():
    batch_size = 0
    num_updates = 10
    report_interval = 10
    learning_rate = 0.5
    update_method = "steepest"

    # create simulator
    sim_one = Simulator(name="simulator")

    # create sample network
    xor_net_one = sim_one.create_net(name="xor")
    xor_net_one.toggle_plots(plots=False)
    xor_net_one.toggle_keyboard(use=False)
    
    # create input layer with 2 units
    xor_net_one.add_group(
        2,
        name="input",
        group_type="input",
        input_transforms=[],
        output_transforms=[]
    )

    # create hidden layer with 2 units; used sigmoid output, dot product input
    xor_net_one.add_group(
        2,
        name="hidden",
        group_type="hidden",
        input_transforms=["dot"],
        output_transforms=["sigmoid"]
    )

    # create output layer with 1 units; used sigmoid output, dot product input, cross entropy error
    xor_net_one.add_group(
        1,
        name="output",
        group_type="output",
        input_transforms=["dot"],
        output_transforms=["sigmoid"],
        error_function="cross_entropy"
    )

    # connects the input layer to the hidden layer; uniform links; full projection
    xor_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="full"
    )

    # connects the hidden layer to the output layer; uniform links; full projection
    xor_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        initialization="uniform",
        proj_type="full"
    )

    # load example set
    xor_net_one.load_example_set(
        "./examples/lens_example_input/xor.ex")

    # load sample weights from clens random weight seed
    xor_net_one.load_clen_weight(
        "./examples/example_networks/xor_examples/xor_example_one_weights.wt")

    # set learning rate and update method
    xor_net_one.set_update_method(learning_rate, update_method)
    # set epochs=100, batch size=0, report interval=10
    xor_net_one.train(num_updates, batch_size, report_interval)
    # load example set for testing again
    xor_net_one.load_example_set(
        "./examples/lens_example_input/xor.ex", testing=True)

    return xor_net_one, batch_size, sim_one


def test_pylens_error(xor_net_one, batch_size, clen_error, sim_one):
    # Test PYLens error
    pylens_error = xor_net_one.test(batch_size)
    print("clens error: ", str(clen_error))
    print("pylens error: ", str(pylens_error))
    # Compare PYLens error with Clen error
    assert (abs(clen_error - pylens_error) <= 0.0001)
    print("Hidden unit lesion Test COMPLETED SUCCESSFULLY")
    sim_one.delete_all_nets()


def test_lesion_hidden_unit_lesion_by_index_0():
    xor_net_one, batch_size, sim_one = test_lesion_hidden_unit()
    xor_net_one.lesion_specific_group("hidden", lesion_units=[0])
    clen_error = 2.81585
    test_pylens_error(xor_net_one, batch_size, clen_error, sim_one)


def test_lesion_hidden_unit_lesion_by_index_1():
    xor_net_one, batch_size, sim_one = test_lesion_hidden_unit()
    xor_net_one.lesion_specific_group("hidden", lesion_units=[1])
    clen_error = 3.09504
    test_pylens_error(xor_net_one, batch_size, clen_error, sim_one)


def test_lesion_hidden_unit_lesion_by_indices():
    xor_net_one, batch_size, sim_one = test_lesion_hidden_unit()
    xor_net_one.lesion_specific_group("hidden", lesion_units=[0, 1])
    clen_error = 2.88622
    test_pylens_error(xor_net_one, batch_size, clen_error, sim_one)


def test_lesion_hidden_unit_lesion_by_proportion_0():
    xor_net_one, batch_size, sim_one = test_lesion_hidden_unit()
    xor_net_one.lesion_specific_group("hidden", lesion_rate=0)
    clen_error = 2.75995
    test_pylens_error(xor_net_one, batch_size, clen_error, sim_one)


def test_lesion_hidden_unit_lesion_by_proportion_1():
    xor_net_one, batch_size, sim_one = test_lesion_hidden_unit()
    xor_net_one.lesion_specific_group("hidden", lesion_rate=1)
    clen_error = 2.88622
    test_pylens_error(xor_net_one, batch_size, clen_error, sim_one)


def test_lesion_hidden_unit_lesion_by_proportion_half():
    xor_net_one, batch_size, sim_one = test_lesion_hidden_unit()
    xor_net_one.lesion_specific_group("hidden", lesion_rate=0.5)
    clen_error = 2.81585
    pylens_error = xor_net_one.test(batch_size)
    print("clens error: approximately ", str(clen_error))
    print("pylens error: ", str(pylens_error))
    assert (abs(clen_error - pylens_error) <= 0.3)
    print("Input to Hidden Unit Lesion Test COMPLETED SUCCESSFULLY")
    sim_one.delete_all_nets()

def test_lesion_hidden_unit_heal_by_index_0():
    xor_net_one, batch_size, sim_one = test_lesion_hidden_unit()
    xor_net_one.lesion_specific_group("hidden", lesion_units=[0])
    xor_net_one.heal_specific_group("hidden", heal_units=[0])
    clen_error = 2.75995
    test_pylens_error(xor_net_one, batch_size, clen_error, sim_one)


def test_lesion_hidden_unit_heal_by_indices():
    xor_net_one, batch_size, sim_one = test_lesion_hidden_unit()
    xor_net_one.lesion_specific_group("hidden", lesion_units=[0, 1])
    xor_net_one.heal_specific_group("hidden", heal_units=[0, 1])
    clen_error = 2.75995
    test_pylens_error(xor_net_one, batch_size, clen_error, sim_one)


def test_lesion_hidden_unit_heal_by_proportion_0():
    xor_net_one, batch_size, sim_one = test_lesion_hidden_unit()
    xor_net_one.lesion_specific_group("hidden", lesion_rate=1)
    xor_net_one.heal_specific_group("hidden", heal_rate=0)
    clen_error = 2.88622
    test_pylens_error(xor_net_one, batch_size, clen_error, sim_one)


def test_lesion_hidden_unit_heal_by_proportion_1():
    xor_net_one, batch_size, sim_one = test_lesion_hidden_unit()
    xor_net_one.lesion_specific_group("hidden", lesion_rate=1)
    xor_net_one.heal_specific_group("hidden", heal_rate=1)
    clen_error = 2.75995
    test_pylens_error(xor_net_one, batch_size, clen_error, sim_one)


def test_lesion_hidden_unit_heal_by_proportion_half():
    xor_net_one, batch_size, sim_one = test_lesion_hidden_unit()
    xor_net_one.lesion_specific_group("hidden", lesion_rate=1)
    xor_net_one.heal_specific_group("hidden", heal_rate=0.5)
    clen_error = 2.88622
    pylens_error = xor_net_one.test(batch_size)
    print("clens error: approximately ", str(clen_error))
    print("pylens error: ", str(pylens_error))
    assert (abs(clen_error - pylens_error) <= 0.03)
    print("Input to Hidden Unit Lesion Test COMPLETED SUCCESSFULLY")
    sim_one.delete_all_nets()


# test_lesion_hidden_unit_lesion_by_index_0()
# test_lesion_hidden_unit_lesion_by_index_1()
# test_lesion_hidden_unit_lesion_by_indices()
# test_lesion_hidden_unit_lesion_by_proportion_0()
# test_lesion_hidden_unit_lesion_by_proportion_1()
test_lesion_hidden_unit_lesion_by_proportion_half()
# test_lesion_hidden_unit_heal_by_index_0()
# test_lesion_hidden_unit_heal_by_indices()
# test_lesion_hidden_unit_heal_by_proportion_0()
# test_lesion_hidden_unit_heal_by_proportion_1()
# test_lesion_hidden_unit_heal_by_proportion_half()

