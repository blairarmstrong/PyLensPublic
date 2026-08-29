import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator


def adder_test(parallel):

    sim = Simulator(name="simulator")
    adder_net = sim.create_net(name="adder10", time_intervals=7)
    adder_net.set_update_method(0.2, "dougs momentum")

    adder_net.add_group(20,
                         name="input",
                         group_type="input",
                         input_transforms=[],
                         output_transforms=[])

    adder_net.add_group(12,
                         name="elman",
                         group_type="elman",
                         input_transforms=[],
                         output_transforms=["sigmoid"])

    adder_net.add_group(12,
                         name="hidden",
                         group_type="hidden",
                         input_transforms=["dot"],
                         output_transforms=["sigmoid"],
                         dropout_rate=0)

    adder_net.add_group(10,
                         name="output",
                         group_type="output",
                         input_transforms=["dot"],
                         output_transforms=["soft_max"],
                         error_function="divergence")


    adder_net.connect_groups(outgoing_group="elman",
                              incoming_group="hidden",
                              initialization="uniform",
                              proj_type="full")

    adder_net.connect_groups(outgoing_group="input",
                              incoming_group="hidden",
                              initialization="uniform",
                              proj_type="full")

    adder_net.connect_groups(outgoing_group="hidden",
                              incoming_group="output",
                              initialization="uniform",
                              proj_type="full")

    adder_net.connect_groups(outgoing_group="hidden",
                              incoming_group="elman",
                             proj_type="elman")

    # load example set
    adder_net.load_example_set("examples/example_networks/adder_examples/adder10.ex")
    adder_net.load_clen_weight("examples/example_networks/adder_examples/adder10_weights.wt")

    if parallel == True:
        adder_net.train(10, batch_size=0, report_interval=1, parallel_mode=True, num_worker=4)
    else:
        adder_net.train(10, batch_size=0, report_interval=1)

    clens_error = round(1735.337, 3)
    clens_weight_cost = round(82.4288, 3)
    clens_grad_linearity = round(-0.01767, 3)

    pylens_error = round(adder_net.stats_plotter.progress_stats['error'][-1], 3)
    pylens_weight_cost = round(adder_net.stats_plotter.progress_stats['weight_cost'][-1], 3)
    pylens_gradient_linearity = round(adder_net.stats_plotter.progress_stats["grad_lin"][-1], 3)

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
    print("     Gradient Linearity: ... ", end="")
    if pylens_weight_cost != clens_weight_cost:
        print("Failed")
        print("Gradient Linearity differ by: {}".format(abs(pylens_gradient_linearity-clens_grad_linearity)))
        return False
    print("passed")
    return True



def test_adder(parallel=False):
    result = adder_test(parallel)
    print("done test")
    if result:
        x = 1
    else:
        x = 0

    assert x == 1


if __name__ == '__main__':
    test_adder()
    test_adder(parallel=True)

