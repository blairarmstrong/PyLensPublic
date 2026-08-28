import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator
import numpy as np

def xor_runner(parallel):
    # Run to Test Example pylens error against clens error
    # STEEPEST, 100 UPDATES

    batch_size = 0
    num_updates = 2
    report_interval = 1
    learning_rate = 0.5
    update_method = "steepest"

    # create simulator
    sim = Simulator(name="sim_xor")

    # create sample network
    xor_net_one = sim.create_net(name="xor")
    # xor_net_one.toggle_print_reports(print_reports=False)
    # xor_net_one.toggle_live_update(live_update=False)
    xor_net_one.toggle_plots(plots=False)
    xor_net_one.toggle_keyboard(use=False)
    # create input layer with 2  units
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

    xor_net_one.load_example_set("./examples/lens_example_input/xor_dense.ex")
    

    # load sample weights from clens random weight seed
    # xor_net_one.load_clen_weight("C:/Users/rajan/OneDrive/Desktop/PyLens Workstudy/PYLens/examples/example_networks/xor_examples/xor_example_one_weights.wt")
    xor_net_one.load_clen_weight("./examples/example_networks/xor_examples/xor_example_one_weights.wt")
    
    xor_net_one.print_link_values("links.txt")

    # set learning rate and update method
    xor_net_one.set_update_method(learning_rate, update_method)
    
    ## test connect group to unit
    test_connect_group_to_unit(xor_net_one)
    ## revert the tste effect
    xor_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full"
    )
        
    # test connect units to units
    test_connect_units(xor_net_one)
    ## revert the tste effect
    xor_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full"
    )
    
    # test disconnect groups
    test_disconnect_groups(xor_net_one)
    ## revert the tste effect
    xor_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full"
    )
    
    # test diconnect units:
    test_disconnect_units(xor_net_one)
    
    # test diconnect group to units:
    test_disconnect_group_to_unit(xor_net_one)
    
    return True


def test_xor(parallel=False):
    result = xor_runner(parallel)
    if result:
        x = 1
    else:
        x = 0

    assert x == 1
    
def test_connect_group_to_unit(net):
    net.connect_group_to_unit("input", "hidden", [0])
    g_out, _ = net._group_pair_from_names("input", "hidden")
    link = g_out.outgoing_links[0]
    connection_mask = link.connection_mask
    expected = np.array([[1.0, 0.0],
                         [1.0, 0.0]], dtype=float)
    assert connection_mask.shape == expected.shape
    np.testing.assert_allclose(connection_mask, expected)
    print("Pass - connect group to unit")
    
def test_connect_units(net):
    net.connect_units("input", [0], "hidden", [1], bidirectional=True)
    g_out, _ = net._group_pair_from_names("input", "hidden")
    connection_mask = g_out.outgoing_links[0].connection_mask
    reverse_connection_mask = g_out.incoming_links[0].connection_mask
    
    expected = np.array([[0.0, 1.0],
                         [0.0, 0.0]], dtype=float)
    expected_reverse = np.array([[0.0, 0.0],
                         [1.0, 0.0]], dtype=float)

    assert connection_mask.shape == expected.shape
    assert reverse_connection_mask.shape == expected_reverse.shape
    np.testing.assert_allclose(connection_mask, expected)
    np.testing.assert_allclose(reverse_connection_mask, expected_reverse)
    print("Pass - connect units")

def test_disconnect_groups(net):
    g_out, _ = net._group_pair_from_names("input", "hidden")
    net.disconnect_groups("input", "hidden")
    assert len(g_out.outgoing_links) == 0
    print("Pass - disconnect groups")
    
def test_disconnect_units(net):
    g_out, g_in = net._group_pair_from_names("input", "hidden")
    net.disconnect_units("input", [0], "hidden", [1])
    link = g_out.outgoing_links[0]
    connection_mask = link.connection_mask
    expected = np.array([[1.0, 0.0],
                         [1.0, 1.0]], dtype=float)
    assert connection_mask.shape == expected.shape
    np.testing.assert_allclose(connection_mask, expected)
    print("Pass - disconnect units")
    
def test_disconnect_group_to_unit(net):
    g_out, g_in = net._group_pair_from_names("input", "hidden")
    net.disconnect_group_units("input", "hidden", [1])
    link = g_out.outgoing_links[0]
    connection_mask = link.connection_mask
    expected = np.array([[1.0, 0.0],
                         [1.0, 0.0]], dtype=float)
    assert connection_mask.shape == expected.shape
    np.testing.assert_allclose(connection_mask, expected)
    print("Pass - disconnect group to units")
    
    

if __name__ == '__main__':
    test_xor()
