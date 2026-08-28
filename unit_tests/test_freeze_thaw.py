

from PyLens.simulator import Simulator
import sys
sys.path.insert(0, ".")
import numpy as np

np.random.seed(100)
sys.path.insert(0, ".")
# Run to Test Example pylens error against clens error
# STEEPEST, 100 UPDATES

num_updates = 10
report_interval = 1
batch_size = 0


def test_freeze_weights():

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
        link_type="uniform",
        proj_type="full"
    )

    # connects the hidden layer to the output layer; uniform links; full projection
    xor_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        link_type="uniform",
        proj_type="full"
    )

    # load example set
    xor_net_one.load_example_set(
        "./examples/lens_example_input/xor.ex"
    )

    # load sample weights from clens random weight seed
    xor_net_one.load_clen_weight(
        "./examples/example_networks/xor_examples/xor_example_one_weights.wt")

    # set learning rate and update method
    xor_net_one.set_update_method(learning_rate, update_method)

    # to use GUI, uncomment
    # sim_one.use_gui(xor_net_one)

    return xor_net_one, sim_one


def test_pylens_error(xor_net_one, clen_error, sim_one):
    # Test PYLens error
    pylens_error = xor_net_one.test(batch_size)
    print("clens error: ", str(clen_error))
    print("pylens error: ", str(pylens_error))
    # Compare PYLens error with Clen error
    assert (abs(clen_error - pylens_error) <= 0.0001)
    sim_one.delete_all_nets()


def train_network(xor_net_one):
    # set epochs=100, batch size=0, report interval=10
    xor_net_one.train(num_updates, batch_size, report_interval)
    # load example set for testing again
    xor_net_one.load_example_set(
        "./examples/lens_example_input/xor.ex", testing=True)


def test_freeze_group_inputs_all():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_group_inputs(freeze_all=True)
    train_network(xor_net_one)

    clen_error = 3.19316
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_freeze_group_inputs_unit_indices_none():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_group_inputs("hidden", units_indices=[])
    train_network(xor_net_one)

    clen_error = 2.75995
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_freeze_group_inputs_unit_indices_all():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_group_inputs("hidden", units_indices=[
                                    0, 1], bias_indices=[0, 1])
    train_network(xor_net_one)

    clen_error = 2.76845
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_freeze_group_inputs_link_indices_none():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_group_inputs("hidden", link_indices=[])
    train_network(xor_net_one)

    clen_error = 2.75995
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_freeze_group_inputs_link_indices_all():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_group_inputs("hidden", link_indices=[(
        1, 0), (1, 1), (0, 0), (0, 1)], bias_indices=[(1, 0), (1, 1), (0, 0), (0, 1)])
    train_network(xor_net_one)

    clen_error = 2.76845
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_thaw_group_inputs_all_before_training():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_group_inputs("hidden", freeze_all=True)
    xor_net_one.thaw_group_inputs("hidden", thaw_all=True)
    train_network(xor_net_one)

    clen_error = 2.75995
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_thaw_group_inputs_all_after_training():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_group_inputs(freeze_all=True)
    xor_net_one.train(num_updates, batch_size, report_interval)

    xor_net_one.thaw_group_inputs(thaw_all=True)
    xor_net_one.toggle_plots(False)
    train_network(xor_net_one)

    clen_error = 2.75995
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_thaw_group_inputs_unit_indices_none():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_group_inputs("hidden", units_indices=[])
    xor_net_one.train(num_updates, batch_size, report_interval)

    xor_net_one.thaw_group_inputs("hidden", units_indices=[])
    xor_net_one.toggle_plots(False)
    train_network(xor_net_one)

    clen_error = 2.75103
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_thaw_group_inputs_unit_indices_all():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_group_inputs("hidden", units_indices=[
                                    0, 1], bias_indices=[0, 1])
    xor_net_one.train(num_updates, batch_size, report_interval)

    xor_net_one.thaw_group_inputs("hidden", units_indices=[
                                  0, 1], bias_indices=[0, 1])
    xor_net_one.toggle_plots(False)
    train_network(xor_net_one)

    clen_error = 2.76141
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_thaw_group_inputs_link_indices_none():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_group_inputs("hidden", link_indices=[])
    xor_net_one.train(num_updates, batch_size, report_interval)

    xor_net_one.thaw_group_inputs("hidden", link_indices=[])
    xor_net_one.toggle_plots(False)
    train_network(xor_net_one)

    clen_error = 2.75103
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_thaw_group_inputs_link_indices_all():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_group_inputs("hidden", link_indices=[(
        1, 0), (1, 1), (0, 0), (0, 1)], bias_indices=[(1, 0), (1, 1), (0, 0), (0, 1)])
    xor_net_one.train(num_updates, batch_size, report_interval)

    xor_net_one.thaw_group_inputs("hidden", link_indices=[(
        1, 0), (1, 1), (0, 0), (0, 1)], bias_indices=[(1, 0), (1, 1), (0, 0), (0, 1)])
    xor_net_one.toggle_plots(False)
    train_network(xor_net_one)

    clen_error = 2.76141
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_freeze_weights_all():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_weights()
    train_network(xor_net_one)

    clen_error = 3.19316
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_freeze_weights_unit_indices_none():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_weights(groups="hidden", units_indices=[])
    train_network(xor_net_one)

    clen_error = 2.75995
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_freeze_weights_unit_indices_all():
    xor_net_one, sim_one = test_freeze_weights()
    # Match legacy freeze_group_inputs(..., bias_indices=[0, 1]) behavior:
    # freeze incoming links to hidden including bias link.
    xor_net_one.freeze_weights(groups="hidden", units_indices=[0, 1])
    train_network(xor_net_one)

    clen_error = 2.76845
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_freeze_weights_link_indices_none():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_weights(groups="hidden", link_indices=[])
    train_network(xor_net_one)

    clen_error = 2.75995
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_freeze_weights_link_indices_all():
    xor_net_one, sim_one = test_freeze_weights()
    # Freeze full main (uniform) hidden input matrix
    xor_net_one.freeze_weights(
        groups="hidden",
        link_indices=[(1, 0), (1, 1), (0, 0), (0, 1)],
        link_type="uniform",
    )
    # Add bias-link freezing to mirror legacy test's bias_indices effect.
    xor_net_one.freeze_weights(groups="hidden", units_indices=[0, 1])
    train_network(xor_net_one)

    clen_error = 2.76845
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_thaw_weights_all_before_training():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_weights()
    xor_net_one.thaw_weights()
    train_network(xor_net_one)

    clen_error = 2.75995
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_thaw_weights_all_after_training():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_weights()
    xor_net_one.train(num_updates, batch_size, report_interval)

    xor_net_one.thaw_weights()
    xor_net_one.toggle_plots(False)
    train_network(xor_net_one)

    clen_error = 2.75995
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_thaw_weights_unit_indices_none():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_weights(groups="hidden", units_indices=[])
    xor_net_one.train(num_updates, batch_size, report_interval)

    xor_net_one.thaw_weights(groups="hidden", units_indices=[])
    xor_net_one.toggle_plots(False)
    train_network(xor_net_one)

    clen_error = 2.75103
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_thaw_weights_unit_indices_all():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_weights(groups="hidden", units_indices=[0, 1])
    xor_net_one.train(num_updates, batch_size, report_interval)

    xor_net_one.thaw_weights(groups="hidden", units_indices=[0, 1])
    xor_net_one.toggle_plots(False)
    train_network(xor_net_one)

    clen_error = 2.76141
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_thaw_weights_link_indices_none():
    xor_net_one, sim_one = test_freeze_weights()
    xor_net_one.freeze_weights(groups="hidden", link_indices=[])
    xor_net_one.train(num_updates, batch_size, report_interval)

    xor_net_one.thaw_weights(groups="hidden", link_indices=[])
    xor_net_one.toggle_plots(False)
    train_network(xor_net_one)

    clen_error = 2.75103
    test_pylens_error(xor_net_one, clen_error, sim_one)


def test_thaw_weights_link_indices_all():
    xor_net_one, sim_one = test_freeze_weights()
    # Freeze full main (uniform) hidden input matrix
    xor_net_one.freeze_weights(
        groups="hidden",
        link_indices=[(1, 0), (1, 1), (0, 0), (0, 1)],
        link_type="uniform",
    )
    # Freeze bias-link columns too for parity with legacy behavior.
    xor_net_one.freeze_weights(groups="hidden", units_indices=[0, 1])
    xor_net_one.train(num_updates, batch_size, report_interval)

    # Thaw the same paths in two steps.
    xor_net_one.thaw_weights(
        groups="hidden",
        link_indices=[(1, 0), (1, 1), (0, 0), (0, 1)],
        link_type="uniform",
    )
    xor_net_one.thaw_weights(groups="hidden", units_indices=[0, 1])
    xor_net_one.toggle_plots(False)
    train_network(xor_net_one)

    clen_error = 2.76141
    test_pylens_error(xor_net_one, clen_error, sim_one)


# test_freeze_group_inputs_all()
# test_freeze_group_inputs_unit_indices_none()
# test_freeze_group_inputs_unit_indices_all()
# test_freeze_group_inputs_link_indices_none()
# test_freeze_group_inputs_link_indices_all()

# test_thaw_group_inputs_all_before_training()
# test_thaw_group_inputs_all_after_training()
# test_thaw_group_inputs_unit_indices_none()
# test_thaw_group_inputs_unit_indices_all()
# test_thaw_group_inputs_link_indices_none()
# test_thaw_group_inputs_link_indices_all()

# test_freeze_weights_all()
# test_freeze_weights_unit_indices_none()
# test_freeze_weights_unit_indices_all()
# test_freeze_weights_link_indices_none()
test_freeze_weights_unit_indices_all()

# test_thaw_weights_all_before_training()
# test_thaw_weights_all_after_training()
# test_thaw_weights_unit_indices_none()
# test_thaw_weights_unit_indices_all()
# test_thaw_weights_link_indices_none()
# test_thaw_weights_link_indices_all()
