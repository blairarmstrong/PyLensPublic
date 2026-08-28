import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator


def filler_test(parallel):
    lesion_rate = 0
    dropout_rate = 0
    perma_lesion_rate = 0

    sim = Simulator(name="simulator")
    filler_net = sim.create_net(name="filler_ex2", time_intervals=5)
    filler_net.toggle_plots(plots=False)
    filler_net.toggle_keyboard(use=False)
    filler_net.set_update_method(0.05, "steepest")

    filler_net.add_group(4,
                         name="chars",
                         group_type="input",
                         input_transforms=[],
                         output_transforms=[])

    filler_net.add_group(20,
                         name="elman",
                         group_type="elman",
                         input_transforms=[],
                         output_transforms=["sigmoid"])

    filler_net.add_group(20,
                         name="hidden",
                         group_type="hidden",
                         input_transforms=["dot"],
                         output_transforms=["sigmoid"],
                         dropout_rate=0)

    filler_net.add_group(20,
                         name="output",
                         group_type="output",
                         input_transforms=["dot"],
                         output_transforms=["sigmoid"],
                         error_function="cross_entropy")

    filler_net.connect_groups(outgoing_group="chars",
                              incoming_group="hidden",
                              link_type="uniform",
                              proj_type="full",
                              dropout_rate=dropout_rate,
                              perma_lesion_rate=perma_lesion_rate)

    filler_net.connect_groups(outgoing_group="elman",
                              incoming_group="hidden",
                              link_type="uniform",
                              proj_type="full")

    filler_net.connect_groups(outgoing_group="hidden",
                              incoming_group="output",
                              link_type="uniform",
                              proj_type="full")

    filler_net.connect_groups(outgoing_group="hidden",
                              incoming_group="elman",
                              link_type="elman",
                              proj_type="one-to-one")

    # load example set
    filler_net.load_example_set("./examples/example_networks/filler_examples/filler_example2.ex")
    filler_net.load_clen_weight("./examples/example_networks/filler_examples/filler_w2.wt")
    # print(filler_net.training_sets[0].print_out_example_set())
    # filler_net.store_weight('filler_py_w1', weight_only=True, format='json')
    if parallel == True:
        filler_net.train(2, batch_size=5, report_interval=1, parallel_mode=True, num_worker=3)
    else:
        filler_net.train(2, batch_size=5, report_interval=1)

    clens_error = round(102.703, 3)
    clens_weight_cost = round(175.046, 3)

    pylens_error = round(filler_net.stats_plotter.progress_stats['error'][-1], 3)
    pylens_weight_cost = round(filler_net.stats_plotter.progress_stats['weight_cost'][-1], 3)

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



def test_filler(parallel=False):
    result = filler_test(parallel)
    print("done test")
    if result:
        x = 1
    else:
        x = 0

    assert x == 1


if __name__ == '__main__':
    test_filler()
    test_filler(parallel=True)

# filler_net.store_weight('test_store', weight_only=True, format='json')
# filler_net.do_test(0)

# sim.use_gui(filler_net)

# 100 epochs, batch size = 10
# pylens:
#   error: 3.95462
#   weight cost: 509.26169
# clens:
#   error: 3.95462
#   weight cost: 509.262
#
# 2 epochs, batch size = 5
# pylens:
#     error: 102.70274
#     weight cost: 175.04566
# clens:
#     error: 102.703
#     weight cost: 175.046

