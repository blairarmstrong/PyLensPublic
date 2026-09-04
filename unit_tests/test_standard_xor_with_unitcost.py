import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator

def xor_runner(parallel, unit_cost):

    batch_size = 0
    num_updates = 100
    report_interval = 50
    learning_rate = 0.5
    update_method = "steepest"

    sim = Simulator(name="sim_xor")

    xor_net_one = sim.create_net(name="xor")
    xor_net_one.toggle_plots(plots=False)
    xor_net_one.toggle_keyboard(use=False)
    xor_net_one.add_group(
        2,
        name="input",
        group_type="input",
        input_transforms=[],
        output_transforms=[]
    )

    xor_net_one.add_group(
        2,
        name="hidden",
        group_type="hidden",
        input_transforms=["dot"],
        output_transforms=["sigmoid"]
    )

    xor_net_one.add_group(
        1,
        name="output",
        group_type="output",
        input_transforms=["dot"],
        output_transforms=["sigmoid"],
        error_function="cross_entropy", 
        unit_cost_function=unit_cost
    )

    xor_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="full"
    )

    xor_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        initialization="uniform",
        proj_type="full"
    )

    xor_net_one.load_example_set("./examples/lens_example_input/xor_dense.ex")

    xor_net_one.load_clen_weight("./examples/example_networks/xor_examples/xor_example_one_weights.wt")

    xor_net_one.set_update_method(learning_rate, update_method)

    if parallel == True:
        xor_net_one.train(num_updates, batch_size, report_interval, parallel_mode=True, num_worker=3)
    else:
        xor_net_one.train(num_updates, batch_size, report_interval)
    if unit_cost == 'linear':
        clens_error = round(2.28547, 3)
        clens_unit_cost = round(3.03676, 3)
        clens_weight_cost = round(15.8262, 3)
    elif unit_cost == 'quadratic':
        clens_error = round(2.27066, 3)
        clens_unit_cost = round(2.30235, 3)
        clens_weight_cost = round(16.4059, 3)
    elif unit_cost == 'conv_quad':
        clens_error = round(2.30337, 3)
        clens_unit_cost = round(3.73354, 3)
        clens_weight_cost = round(15.1567, 3)
    elif unit_cost == 'logistic':
        clens_error = round(2.30431, 3)
        clens_unit_cost = round(3.80457, 3)
        clens_weight_cost = round(15.1219, 3)
    elif unit_cost == 'cosine':
        clens_error = round(2.29883, 3)
        clens_unit_cost = round(6.78486, 3)
        clens_weight_cost = round(15.3267, 3)
 

    pylens_error = round(xor_net_one.stats_plotter.progress_stats['error'][-1], 3)
    pylens_weight_cost = round(xor_net_one.stats_plotter.progress_stats["weight_cost"][-1], 3)
    pylens_unit_cost = round(xor_net_one.stats_plotter.progress_stats["unit_cost"][-1], 3)

    print("testing xor:")
    print("     Error: ... ", end="")
    sim.delete_all_nets()
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
    print("     Unit Costs: ... ", end="")
    if pylens_unit_cost != clens_unit_cost:
        print("Failed")
        print("Unit costs differ by: {}".format(abs(pylens_unit_cost - clens_unit_cost)))
        return False
    print("passed")
    return True


def test_xor(parallel=False, unit_cost='linear'):
    result = xor_runner(parallel, unit_cost)
    if result:
        x = 1
    else:
        x = 0

    assert x == 1

if __name__ == '__main__':
    test_xor(unit_cost='linear')
    test_xor(unit_cost='quadratic')
    test_xor(unit_cost='conv_quad')
    test_xor(unit_cost='logistic')
    test_xor(unit_cost='cosine')

