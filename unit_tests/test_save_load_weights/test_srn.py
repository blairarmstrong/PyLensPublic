from PyLens.simulator import Simulator
import numpy as np

def srn_runner():

    lesion_rate = 0
    dropout_rate = 0
    perma_lesion_rate = 0

    sim = Simulator(name="simulator")
    srn = sim.create_net(name="srn", time_intervals=2)

    # srn.load_example_set("examples.ex")
    srn.set_update_method(0.1, "steepest")

    srn.add_group(2, name="input", group_type="input", input_transforms=[], output_transforms=[])
    srn.add_group(2, name="elman", group_type="elman", input_transforms=[], output_transforms=["sigmoid"])
    srn.add_group(2, name="hidden",  group_type="hidden", input_transforms=["dot"],
                         output_transforms=["sigmoid"], lesion_rate=0, dropout_rate=0)
    srn.add_group(2, name="output", group_type="output", input_transforms=["dot"],
                         output_transforms=["sigmoid"], error_function="cross_entropy")

    srn.connect_groups(outgoing_group="input", incoming_group="hidden", link_type="uniform", proj_type="full",
                              lesion_rate=lesion_rate, dropout_rate=dropout_rate, perma_lesion_rate=perma_lesion_rate)
    srn.connect_groups(outgoing_group="elman", incoming_group="hidden", link_type="uniform", proj_type="full")
    srn.connect_groups(outgoing_group="hidden", incoming_group="output", link_type="uniform", proj_type="full")

    srn.connect_groups(outgoing_group="hidden", incoming_group="elman", link_type="elman", proj_type="one-to-one")

    # Disable plots
    srn.toggle_plots(plots=False)

    # load example set
    srn.load_example_set("./examples/example_networks/srn_examples/examples_double.ex")
    srn.load_clen_weight("./examples/example_networks/srn_examples/srn.wt")

    srn.train(1, 0, report_interval=1)

    srn.store_links('test_srn_weight.json', weight_only=True, format='json')
    srn.reset_network()
    srn.load_links('test_srn_weight.json')

    srn.train(1, 0, report_interval=1)

    clens_error = round(5.76951, 3)
    clens_weight_cost = round(2.87860, 3)

    # pylens - error: 2.30684 weight cost: 15.02701
    pylens_error = round(srn.stats_plotter.progress_stats['error'][-1], 3)
    pylens_weight_cost = round(srn.stats_plotter.progress_stats["weight_cost"][-1], 3)

    assert clens_error == pylens_error
    assert clens_weight_cost == pylens_weight_cost

    print("pass")

if __name__ == '__main__':
    srn_runner()
