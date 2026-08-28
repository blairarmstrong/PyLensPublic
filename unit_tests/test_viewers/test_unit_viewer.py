import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator
from PyLens.gui.main_viewer_tk import main_viewer_tk
from PyLens.gui.unit_viewer_tk import FrameExamplesProgram
from PyLens.gui.unit_viewer_tk import PINK_HEX, YELLOW_HEX, LIGHT_GREY_HEX


def test_clicks(unit_viewer):
    unit_viewer.window.update()

    output = unit_viewer.nodes["output"][0]
    output_outline = unit_viewer.output_nodes_border[0]
    hidden = unit_viewer.nodes["hidden"][0]

    original_output_rim_color = unit_viewer.canvas.itemcget(output_outline, "outline")
    original_hidden_fill = unit_viewer.canvas.itemcget(hidden, "fill")
    original_hidden_outline = unit_viewer.canvas.itemcget(hidden, "outline")

    def click_center(item, button=1):
        x1, y1, x2, y2 = unit_viewer.canvas.coords(item)
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)

        unit_viewer.canvas.event_generate(
            f"<Button-{button}>",
            x=x,
            y=y,
        )
        unit_viewer.window.update()

    def click_rim(item, button=1):
        x1, y1, x2, y2 = unit_viewer.canvas.coords(item)
        x = int(x1)
        y = int((y1 + y2) / 2)

        unit_viewer.canvas.event_generate(
            f"<Button-{button}>",
            x=x,
            y=y,
        )
        unit_viewer.window.update()

    # LEFT CLICK: output center
    click_center(output)

    assert unit_viewer.selected_node == output
    assert unit_viewer.canvas.itemcget(output_outline, "outline") == PINK_HEX

    # LEFT CLICK: target rim
    click_rim(output_outline)

    assert unit_viewer.selected_node == output_outline
    assert unit_viewer.canvas.itemcget(output_outline, "outline") == PINK_HEX

    # LEFT CLICK: hidden
    click_center(hidden)

    assert unit_viewer.selected_node == hidden
    assert unit_viewer.canvas.itemcget(output_outline, "outline") == original_output_rim_color
    assert unit_viewer.canvas.itemcget(hidden, "outline") == PINK_HEX

    # LEFT CLICK hidden again: unselect
    click_center(hidden)

    assert unit_viewer.selected_node is None
    assert unit_viewer.canvas.itemcget(hidden, "outline") == original_hidden_outline

    # RIGHT CLICK: hidden
    click_center(hidden, button=3)

    assert unit_viewer.right_click is True
    assert unit_viewer.right_click_id == hidden
    assert unit_viewer.link_weight_flag
    assert unit_viewer.canvas.itemcget(hidden, "fill") == 'gray'

    # RIGHT CLICK same hidden again: exit link mode
    click_center(hidden, button=3)
    assert unit_viewer.canvas.itemcget(hidden, "fill") == original_hidden_fill

    assert unit_viewer.right_click is False
    assert not unit_viewer.link_weight_flag


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

    # load example set
    # xor_net_one.load_example_set("C:/Users/rajan/OneDrive/Desktop/PyLens Workstudy/PYLens/examples/lens_example_input/xor.ex")
    xor_net_one.load_example_set("./examples/lens_example_input/xor_dense.ex")

    # load sample weights from clens random weight seed
    # xor_net_one.load_clen_weight("C:/Users/rajan/OneDrive/Desktop/PyLens Workstudy/PYLens/examples/example_networks/xor_examples/xor_example_one_weights.wt")
    xor_net_one.load_clen_weight("./examples/example_networks/xor_examples/xor_example_one_weights.wt")

    # set learning rate and update method
    xor_net_one.set_update_method(learning_rate, update_method)

    # to use GUI, uncomment
    xor_net_one.simulator = sim
    xor_net_one.visualized = True
    program = main_viewer_tk(input_net=xor_net_one)
    sim.gui_program = program

    sim.gui_program.unit_viewer = FrameExamplesProgram(input_net=xor_net_one, parent=sim.gui_program.window)
    test_clicks(sim.gui_program.unit_viewer)

    sim.gui_program.train_report_int.set(report_interval)
    sim.gui_program.train_weight_updates.set(num_updates)
    sim.gui_program.train_batch_size.set(batch_size)
    sim.gui_program.start_training()
    sim.gui_program.training_thread.join()

    # COMPARE ERRORS AND WEIGHT COST

    # clens - error: 2.30689 weight cost: 15.02701
    # Below is old
    clens_error = round(2.76625, 3)
    clens_weight_cost = round(1.74944, 3)
    # clens_error = round(2.78054, 3)
    # clens_weight_cost = round(1.78510, 3)

    # pylens - error: 2.30684 weight cost: 15.02701
    pylens_error = round(xor_net_one.stats_plotter.progress_stats['error'][-1], 3)
    pylens_weight_cost = round(xor_net_one.stats_plotter.progress_stats["weight_cost"][-1], 3)

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
