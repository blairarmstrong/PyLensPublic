import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator

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
        name="hidden1",
        group_type="hidden",
        input_transforms=["dot"],
        output_transforms=["sigmoid"]
    )
    
    xor_net_one.add_group(
        2,
        name="hidden2",
        group_type="elman",
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
        incoming_group="hidden1",
        initialization="uniform",
        proj_type="full"
    )
    
    xor_net_one.connect_groups(
        outgoing_group="hidden1",
        incoming_group="hidden2",
        initialization="uniform",
        proj_type="full"
    )

    # connects the hidden layer to the output layer; uniform links; full projection
    xor_net_one.connect_groups(
        outgoing_group="hidden2",
        incoming_group="output",
        initialization="uniform",
        proj_type="full"
    )
    
    xor_net_one.load_example_set("./examples/lens_example_input/xor_dense.ex")
    
    xor_net_one.elman_connect("hidden1", "hidden2")
    
    xor_net_one.print_link_values("links.txt")

    # set learning rate and update method
    xor_net_one.set_update_method(learning_rate, update_method)
    
    xor_net_one.train(num_updates, batch_size, report_interval)
    return True


def test_xor(parallel=False):
    result = xor_runner(parallel)
    if result:
        x = 1
    else:
        x = 0

    assert x == 1

if __name__ == '__main__':
    test_xor()
