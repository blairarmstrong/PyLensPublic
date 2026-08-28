import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator


def filler_srbptt_runner(parallel):
    # Example network for srbptt using the filler network

    batch_size = 0
    num_updates = 100
    report_interval = 1
    learning_rate = 0.05
    update_method = "steepest"

    dropout_rate = 0
    perma_lesion_rate = 0

    sim = Simulator(name="simulator", baseType='pytorch')
    filler_net = sim.create_net(name="filler", time_intervals=5, type='srbptt')
    # disable plots and keyboard input
    filler_net.toggle_plots(plots=False)
    filler_net.toggle_keyboard(use=False)


    # build network
    filler_net.add_group(4, name="chars", group_type="input", input_transforms=[], output_transforms=[])
    filler_net.add_group(20, name="elman", group_type="elman", input_transforms=[], output_transforms=["sigmoid"])
    filler_net.add_group(20, name="hidden",  group_type="hidden", input_transforms=["dot"],
                         output_transforms=["sigmoid"], dropout_rate=0)
    filler_net.add_group(20, name="output", group_type="output", input_transforms=["dot"],
                         output_transforms=["sigmoid"], error_function="cross_entropy")

    # Apparently this order matters for srbptt
    # Or else weights get stored in the wrong order
    filler_net.connect_groups(outgoing_group="elman", incoming_group="hidden", link_type="uniform", proj_type="full")
    filler_net.connect_groups(outgoing_group="chars", incoming_group="hidden", link_type="uniform", proj_type="full",
                              dropout_rate=dropout_rate, perma_lesion_rate=perma_lesion_rate)
    filler_net.connect_groups(outgoing_group="hidden", incoming_group="output", link_type="uniform", proj_type="full")

    filler_net.connect_groups(outgoing_group="hidden", incoming_group="elman", link_type="elman", proj_type="one-to-one")

    filler_net.toggle_plots(False)
    # load example set
    filler_net.load_example_set("./examples/example_networks/filler_srbptt_examples/filler_srbptt.ex")
    filler_net.load_clen_weight("./examples/example_networks/filler_srbptt_examples/filler_srbptt_w1.wt")
    # lr = 0.05, method = steepest
    filler_net.set_update_method(learning_rate, update_method)

    if parallel == True:
        filler_net.train(epochs=num_updates, batch_size=batch_size, report_interval=report_interval, parallel_mode=True, num_worker=4)
    else:
        filler_net.train(epochs=num_updates, batch_size=batch_size, report_interval=report_interval)
    clens_error = round(16.7977, 3)
    clens_weight_cost = round(455.841, 3)


    # #
    pylens_error = round(float(filler_net.stats_plotter.progress_stats['error'][-1]), 3)
    pylens_weight_cost = round(float(filler_net.stats_plotter.progress_stats['weight_cost'][-1]), 3)

    print("testing filler:")
    print("     Error: ... ", end="")
    if pylens_error != clens_error:
        print("Failed")
        print("Errors differ by: {}".format(abs(pylens_error-clens_error)))
        return False
    print("passed")
    print("     Weight Costs: ... ", end="")
    if pylens_weight_cost != clens_weight_cost:
        print("Failed")
        print("Weight costs differ by: {}".format(abs(pylens_weight_cost-clens_weight_cost)))
        return False
    print("passed")
    return True

def test_filler_srbptt(parallel=False):
    result = filler_srbptt_runner(parallel)
    if result:
        x = 1
    else:
        x = 0

    assert x == 1

if __name__ == '__main__':
    test_filler_srbptt()
    test_filler_srbptt(parallel=True)
