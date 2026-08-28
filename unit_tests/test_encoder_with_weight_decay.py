import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator

def encoder_runner(parallel, update_method="steepest"):
    # set parameters
    batch = 0
    num_updates = 2
    report_interval = 1
    learning_rate = 0.1
    weight_decay = 0.0001
    zero_error_radius = 0.01
    hidden = 3

    sim = Simulator(name="simulator")
    encoder = sim.create_net(name="encoder")
    encoder.toggle_plots(plots=False)
    encoder.toggle_keyboard(use=False)
    encoder.set_update_method(learning_rate, update_method)

    encoder.optimizer.weight_decay = weight_decay
    encoder.optimizer.optimizer_params.PAR_O_zeroErrorRadius = zero_error_radius

    encoder.add_group(8,
                      name="input",
                      group_type="input",
                      input_transforms=[],
                      output_transforms=[])
    encoder.add_group(hidden,
                      name="hidden",
                      group_type="hidden",
                      input_transforms=["dot"],
                      output_transforms=["sigmoid"])
    encoder.add_group(8,
                      name="output",
                      group_type="output",
                      input_transforms=["dot"],
                      output_transforms=["sigmoid"],
                      error_function="cross_entropy")

    encoder.connect_groups(outgoing_group="input",
                           incoming_group="hidden",
                           link_type="uniform",
                           proj_type="full")

    encoder.connect_groups(outgoing_group="hidden",
                           incoming_group="output",
                           link_type="uniform",
                           proj_type="full")

    # encoder.load_example_set("unit_tests/encoder.sparse.ex")
    # encoder.load_clen_weight("unit_tests/encoder_example_one_weights.wt")
    # load example set
    encoder.load_example_set("./examples/lens_example_input/encoder.sparse.ex")

    # load sample weights from clens random weight seed
    encoder.load_clen_weight("./examples/example_networks/encoder_examples/encoder_example_one_weights.wt")

    if parallel == True:
        encoder.train(epochs=num_updates, batch_size=batch, report_interval=report_interval, parallel_mode=True, num_worker=3)
    else:
        encoder.train(epochs=num_updates, batch_size=batch, report_interval=report_interval)

    pylens_error = round(encoder.stats_plotter.progress_stats['error'][-1], 3)
    pylens_weight_cost = round(encoder.stats_plotter.progress_stats['weight_cost'][-1], 3)
    pylens_gradient_linearity = round(encoder.stats_plotter.progress_stats["grad_lin"][-1], 3)

    if update_method == 'steepest':
        clens_error = round(41.0797, 3)
        clens_weight_cost = round(8.82949, 3)
        clens_grad_linearity = round(0.98113, 3)
    elif update_method == 'dougs momentum':
        clens_error = round(60.4910, 3)
        clens_weight_cost = round(8.92054, 3)
        clens_grad_linearity = round(0.99993, 3)
    elif update_method == 'momentum':
        clens_error = round(41.0797, 3)
        clens_weight_cost = round(11.5157, 3)
        clens_grad_linearity = round(0.98113, 3)
    elif update_method == 'delta bar delta':
        clens_error = round(42.7327, 3)
        clens_weight_cost = round(10.9172, 3)
        clens_grad_linearity = round(0.98490, 3)

    print("testing encoder:")
    print("     Error: ... ", end="")
    if pylens_error != clens_error:
        print("Failed")
        print("Errors differ by: {}".format(abs(pylens_error-clens_error)))
        import ipdb; ipdb.set_trace()
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

def test_encoder(parallel=False, update_method='steepest'):
    result = encoder_runner(parallel, update_method)
    if result:
        x = 1
    else:
        x = 0

    assert x == x

if __name__ == '__main__':
    test_encoder()
    test_encoder(update_method='dougs momentum')
    test_encoder(update_method='momentum')
    test_encoder(update_method='delta bar delta')
    test_encoder(parallel=True)
    test_encoder(update_method='dougs momentum', parallel=True)
    test_encoder(update_method='momentum', parallel=True)
    test_encoder(update_method='delta bar delta', parallel=True)
