import sys
sys.path.insert(0, ".")

from PyLens.simulator import Simulator
from PyLens.gui.main_viewer_tk import main_viewer_tk
from PyLens.gui.unit_viewer_tk import FrameExamplesProgram
from PyLens.gui.unit_viewer_tk import PINK_HEX

def build_xor_network():
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
        error_function="cross_entropy"
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

    xor_net_one.load_example_set(
        "./examples/lens_example_input/xor_dense.ex"
    )

    xor_net_one.load_clen_weight(
        "./examples/example_networks/xor_examples/xor_example_one_weights.wt"
    )


    return sim, xor_net_one


def build_task_net():

    # create simulator
    sim_one = Simulator(name="simulator")

    # create sample network
    task_net_one = sim_one.create_net(name="task")


    # create input layer with 3 units
    task_net_one.add_group(
        3,
        name="task",
        group_type="input",
        input_transforms=[],
        output_transforms=[]
    )

    # create hidden layer with 6 units; used sigmoid output, dot product input
    task_net_one.add_group(
        6,
        name="taskHidden",
        group_type="hidden",
        input_transforms=["dot"],
        output_transforms=["sigmoid"]
    )

    # create input layer with 6 units; used sigmoid output, dot product input
    task_net_one.add_group(
        6,
        name="input",
        group_type="input",
        input_transforms=["dot"],
        output_transforms=["sigmoid"]
    )

    # create input layer with 6 units; used sigmoid output, product input
    task_net_one.add_group(
        6,
        name="hidden",
        group_type="hidden",
        input_transforms=["product"],
        output_transforms=["sigmoid"]
    )

    # create output layer with 2 units; used sigmoid output, dot product input
    task_net_one.add_group(
        2,
        name="output",
        group_type="output",
        input_transforms=["dot"],
        output_transforms=["sigmoid"]
    )


    # connects the task layer to the taskHidden layer; uniform links; full projection
    task_net_one.connect_groups(
        outgoing_group="task",
        incoming_group="taskHidden",
        initialization="uniform",
        proj_type="full"
    )

    # connects the taskHidden layer to the hidden layer; uniform links; full projection
    task_net_one.connect_groups(
        outgoing_group="taskHidden",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="one-to-one"
    )

    # connects the input layer to the hidden layer; uniform links; full projection
    task_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="one-to-one"
    )

    # connects the hidden layer to the output layer; uniform links; full projection
    task_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        initialization="uniform",
        proj_type="full"
    )

    # load example set
    task_net_one.load_example_set("./examples/lens_example_input/task.ex")

    # load sample weights from clens random weight seed
    task_net_one.load_clen_weight("./examples/example_networks/task_examples/task_example_one_weights.wt")

    return sim_one, task_net_one

def build_digits_network():
    sim_one = Simulator(name="simulator")

    digits_net = sim_one.create_net(name='digits')

    digits_net.add_group(
            20,
            name="input",
            group_type="input",
            num_cols=4,
    )
    digits_net.add_group(
            20,
            name="hidden",
            group_type="hidden",
    )
    digits_net.add_group(
            3,
            name="output",
            group_type="output",
    )

    digits_net.connect_groups(
            outgoing_group="input",
            incoming_group="hidden",
    )
    digits_net.connect_groups(
            outgoing_group="hidden",
            incoming_group="output",
    )

    digits_net.load_example_set("./examples/lens_example_input/digits.ex", "digits")

    return sim_one, digits_net

def build_filler_network():
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
                              initialization="uniform",
                              proj_type="full",
                              dropout_rate=dropout_rate,
                              perma_lesion_rate=perma_lesion_rate)

    filler_net.connect_groups(outgoing_group="elman",
                              incoming_group="hidden",
                              initialization="uniform",
                              proj_type="full")

    filler_net.connect_groups(outgoing_group="hidden",
                              incoming_group="output",
                              initialization="uniform",
                              proj_type="full")

    filler_net.connect_groups(outgoing_group="hidden",
                              incoming_group="elman",
                              proj_type="elman")

    # load example set
    filler_net.load_example_set("./examples/example_networks/filler_examples/filler_example2.ex")
    filler_net.load_clen_weight("./examples/example_networks/filler_examples/filler_w2.wt")

    return sim, filler_net

def build_filler_srbptt_network():
    dropout_rate = 0
    perma_lesion_rate = 0

    sim = Simulator(name="simulator")
    filler_net = sim.create_net(name="filler", time_intervals=5, type='srbptt')

    # build network
    filler_net.add_group(4, name="chars", group_type="input", input_transforms=[], output_transforms=[])
    filler_net.add_group(20, name="elman", group_type="elman", input_transforms=[], output_transforms=["sigmoid"])
    filler_net.add_group(20, name="hidden",  group_type="hidden", input_transforms=["dot"],
                         output_transforms=["sigmoid"], dropout_rate=0)
    filler_net.add_group(20, name="output", group_type="output", input_transforms=["dot"],
                         output_transforms=["sigmoid"], error_function="cross_entropy")

    # Apparently this order matters for srbptt
    # Or else weights get stored in the wrong order
    filler_net.connect_groups(outgoing_group="elman", incoming_group="hidden", initialization="uniform", proj_type="full")
    filler_net.connect_groups(outgoing_group="chars", incoming_group="hidden", initialization="uniform", proj_type="full",
                              dropout_rate=dropout_rate, perma_lesion_rate=perma_lesion_rate)
    filler_net.connect_groups(outgoing_group="hidden", incoming_group="output", initialization="uniform", proj_type="full")

    filler_net.connect_groups(outgoing_group="hidden", incoming_group="elman", proj_type="elman")

    # load example set
    filler_net.load_example_set("./examples/example_networks/filler_srbptt_examples/filler_srbptt.ex")
    filler_net.load_clen_weight("./examples/example_networks/filler_srbptt_examples/filler_srbptt_w1.wt")

    return sim, filler_net

def build_rand_network():
    # create simulator
    sim_one = Simulator(name="simulator")

    # create sample network
    rand10x40 = sim_one.create_net(name='rand', time_intervals=4, ticks_per_interval=5,type='continuous')
    # rand10x40.plot = False
    rand10x40.toggle_plots(plots=False)
    rand10x40.toggle_keyboard(use=False)

    # create input layer with 10 units
    rand10x40.add_group(
        10,
        name="input",
        group_type="input",
        input_transforms=[],
        output_transforms=[]
    )

    # create hidden layer with 50 units; used sigmoid output, dot product input, and input integration
    rand10x40.add_group(
        50,
        name="hidden",
        group_type="hidden",
        input_transforms=["dot", "in_integr"],
        output_transforms=["sigmoid",]
    )

    # create output layer with 10 units; used sigmoid output, dot product input, cross entropy error
    rand10x40.add_group(
        10,
        name="output",
        group_type="output",
        input_transforms=["dot"],
        output_transforms=["sigmoid", "out_integr"],
        error_function="cross_entropy"
    )

    # connects the input layer to the hidden layer; uniform links; full projection
    rand10x40.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="full"
    )

    # connects the output layer to the hidden layer; uniform links; full projection
    rand10x40.connect_groups(
        outgoing_group="output",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="full"
    )


    # connects the hidden layer to the output layer; uniform links; full projection
    rand10x40.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        initialization="uniform",
        proj_type="full"
    )


    # load example set
    rand10x40.load_example_set("./examples/example_networks/rand10x40_examples/rand10x40.ex")
    # load sample weights from clens random weight seed
    rand10x40.load_clen_weight("./examples/example_networks/rand10x40_examples/rand10x40_weights.wt")

    return sim_one, rand10x40

def build_boltz_network():
    # create simulator
    sim_one = Simulator(name="simulator")

    # create sample network
    digits_net_one = sim_one.create_net(name='dbm', time_intervals=4,
                                    ticks_per_interval=5, type='boltzmann')
    digits_net_one.plot = False

    # create input layer with 20 units
    digits_net_one.add_group(
        20,
        name="input",
        group_type="input",
    )

    # create hidden layer with 10 units
    digits_net_one.add_group(
        10,
        name="hidden",
        group_type="hidden",
    )

    # create output layer with 3 units; cross entropy error is standard for BMs
    digits_net_one.add_group(
        3,
        name="output",
        group_type="output",
    )

    # Connect the groups:
    # connects the input layer to the hidden layer; uniform links; full projection; bidirectional
    digits_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="full",
        bidirectional=True
    )
    # connects the hidden layer laterally; uniform links; full projection
    digits_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="full"
    )
    # connects the hidden layer to the output layer; uniform links; full projection; bidirectional
    digits_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        initialization="uniform",
        proj_type="full",
        bidirectional=True
    )
    # connects the output layer laterally; uniform links; full projection
    digits_net_one.connect_groups(
        outgoing_group="output",
        incoming_group="output",
        initialization="uniform",
        proj_type="full"
    )

    digits_net_one.load_example_set("./examples/lens_example_input/digits.ex")

    digits_net_one.set_properties(
        group_criterion_threshold=0.001,
        test_group_criterion_threshold=0.001,
        clamp_strength=1.0,
        init_gain=0.1,
        final_gain=1.0,
        anneal_time=1.0,
        training_sets=[
            {"max_time": 3.0, "min_time": 0.0, "grace_time": 1.0}
        ]
    )
    digits_net_one.load_clen_weight("./examples/example_networks/boltzmann_examples/boltz_big.wt")

    return sim_one, digits_net_one

###############################################################
def test_clicks(unit_viewer):
    unit_viewer.window.update()

    output = unit_viewer.nodes["output"][0]
    output_outline = unit_viewer.output_nodes_border[0]
    hidden = unit_viewer.nodes["hidden"][0]

    original_output_rim_color = unit_viewer.canvas.itemcget(output_outline, "outline")
    original_hidden_fill = unit_viewer.canvas.itemcget(hidden, "fill")
    original_hidden_outline = unit_viewer.canvas.itemcget(hidden, "outline")

    def get_group(name):
        return next(
            group for group in unit_viewer.input_net.groups
            if group.name == name
        )

    def check_output_value(item, group_name, unit_idx):
        group = get_group(group_name)

        expected = group.output_history[unit_viewer._curr_tick_idx][unit_idx]
        if hasattr(expected, "item"):
            expected = expected.item()

        # value stored by unit viewer for this canvas item
        displayed = float(unit_viewer.tags[item][2])

        assert displayed == expected

        # value actually shown in bottom textbox
        assert unit_viewer.lower_textbox.get() == f"O: {float(expected):.8f}"

    # this check the target rim of output
    def check_target_value(item, group_name, unit_idx):
        group = get_group(group_name)

        # for multi-tick nets, move forward one tick first
        # because multi-tick net starts at tick 1
        moved_forward = False
        if unit_viewer._ticks_per_ex > 1:
            unit_viewer.btn_lst[3].invoke()  # >
            moved_forward = True

        expected = group.target_history[unit_viewer._curr_tick_idx][unit_idx]
        if hasattr(expected, "item"):
            expected = expected.item()

        displayed = float(unit_viewer.tags[item][3])

        assert displayed == expected

        # restore original tick
        if moved_forward:
            unit_viewer.btn_lst[2].invoke()  # <

    def click_center(item, button=1):
        x1, y1, x2, y2 = unit_viewer.canvas.coords(item)

        canvas_x = (x1 + x2) / 2
        canvas_y = (y1 + y2) / 2

        x = int(canvas_x - unit_viewer.canvas.canvasx(0))
        y = int(canvas_y - unit_viewer.canvas.canvasy(0))

        unit_viewer.canvas.event_generate(
            f"<Button-{button}>",
            x=x,
            y=y,
        )
        unit_viewer.window.update()

    def click_rim(item, button=1):
        x1, y1, x2, y2 = unit_viewer.canvas.coords(item)

        canvas_x = x1
        canvas_y = (y1 + y2) / 2

        x = int(canvas_x - unit_viewer.canvas.canvasx(0))
        y = int(canvas_y - unit_viewer.canvas.canvasy(0))

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
    check_output_value(output, "output", 0)

    # LEFT CLICK: target rim
    click_rim(output_outline)
    assert unit_viewer.selected_node == output_outline
    assert unit_viewer.canvas.itemcget(output_outline, "outline") == PINK_HEX
    check_target_value(output_outline, "output", 0)

    # LEFT CLICK: hidden
    click_center(hidden)
    assert unit_viewer.selected_node == hidden
    assert unit_viewer.canvas.itemcget(output_outline, "outline") == original_output_rim_color
    assert unit_viewer.canvas.itemcget(hidden, "outline") == PINK_HEX
    check_output_value(hidden, "hidden", 0)

    # LEFT CLICK hidden again: unselect
    click_center(hidden)
    assert unit_viewer.selected_node is None
    assert unit_viewer.canvas.itemcget(hidden, "outline") == original_hidden_outline
    check_output_value(hidden, "hidden", 0)

    # RIGHT CLICK: hidden
    click_center(hidden, button=3)
    assert unit_viewer.right_click is True
    assert unit_viewer.right_click_id == hidden
    assert unit_viewer.link_weight_flag
    # network such as boltz has connection to itself
    hidden_group = next(
        group for group in unit_viewer.input_net.groups
        if group.name == "hidden"
    )
    has_self_link = any(
        link.outgoing_group is hidden_group and
        link.incoming_group is hidden_group
        for link in hidden_group.outgoing_links
    )
    if not has_self_link:
        assert unit_viewer.canvas.itemcget(hidden, "fill") == "gray"

    # RIGHT CLICK same hidden again: exit link mode
    click_center(hidden, button=3)
    assert unit_viewer.canvas.itemcget(hidden, "fill") == original_hidden_fill
    assert unit_viewer.right_click is False
    assert not unit_viewer.link_weight_flag

def test_redrawing(unit_viewer):
    unit_viewer.window.update()

    # Save original canvas structure
    original_counts = {
        group_name: len(nodes)
        for group_name, nodes in unit_viewer.nodes.items()
    }

    original_total_items = len(unit_viewer.canvas.find_all())

    # REFRESH: should not create duplicate nodes
    unit_viewer.refresh_canvas()
    unit_viewer.window.update()

    assert {
        group_name: len(nodes)
        for group_name, nodes in unit_viewer.nodes.items()
    } == original_counts

    assert len(unit_viewer.canvas.find_all()) == original_total_items

    # CELL SIZE: nodes should actually become larger
    group_name = next(
        name for name, nodes in unit_viewer.nodes.items()
        if nodes
    )

    node = unit_viewer.nodes[group_name][0]
    x1, y1, x2, y2 = unit_viewer.canvas.coords(node)
    original_width = x2 - x1

    original_cell_size = unit_viewer.cell_size

    unit_viewer.update_cell_size(original_cell_size + 2)
    unit_viewer.window.update()

    node = unit_viewer.nodes[group_name][0]
    x1, y1, x2, y2 = unit_viewer.canvas.coords(node)

    assert x2 - x1 > original_width

    # Restore
    unit_viewer.update_cell_size(original_cell_size)
    unit_viewer.window.update()

    assert {
        group_name: len(nodes)
        for group_name, nodes in unit_viewer.nodes.items()
    } == original_counts

def test_time_buttons(unit_viewer):
    unit_viewer.window.update()

    # >
    unit_viewer._curr_tick_idx = 0
    unit_viewer.update_textboxes()

    unit_viewer.btn_lst[3].invoke()

    assert unit_viewer._curr_tick_idx == 1

    # <
    unit_viewer.btn_lst[2].invoke()

    assert unit_viewer._curr_tick_idx == 0

    # >>>
    unit_viewer.btn_lst[5].invoke()

    assert unit_viewer._curr_tick_idx == unit_viewer._ticks_per_ex - 1
    assert unit_viewer.end_of_example()

    # <<<
    unit_viewer.btn_lst[0].invoke()

    assert unit_viewer._curr_tick_idx == 0
    assert unit_viewer.start_of_example()

    # << and >>
    if unit_viewer.ev_total > 1:
        unit_viewer._curr_tick_idx = unit_viewer._ticks_per_ev + 1
        unit_viewer.update_textboxes()

        unit_viewer.btn_lst[1].invoke()
        assert unit_viewer.start_of_event()

        unit_viewer.btn_lst[4].invoke()
        assert unit_viewer.end_of_event()

def test_value_menu(unit_viewer):
    group = next(
        g for g in unit_viewer.input_net.groups
        if g.name == "output"
    )

    original_tick = unit_viewer._curr_tick_idx

    # for multi-tick nets, move forward one tick first
    if unit_viewer._ticks_per_ex > 1:
        unit_viewer.btn_lst[3].invoke()  # >

    tick = unit_viewer._curr_tick_idx
    item = unit_viewer.nodes["output"][0]

    # Outputs
    unit_viewer.update_value_choice("Outputs")
    unit_viewer.window.update()

    expected = group.output_history[tick][0]
    if hasattr(expected, "item"):
        expected = expected.item()

    assert float(unit_viewer.tags[item][2]) == expected

    # Targets
    unit_viewer.update_value_choice("Targets")
    unit_viewer.window.update()

    expected = group.target_history[tick][0]
    if hasattr(expected, "item"):
        expected = expected.item()

    assert float(unit_viewer.tags[item][2]) == expected

    # restore menu
    unit_viewer.update_value_choice("Outputs and Targets")

    # restore tick
    if unit_viewer._ticks_per_ex > 1:
        unit_viewer.btn_lst[2].invoke()  # <
        assert unit_viewer._curr_tick_idx == original_tick

def run_unit_viewer_test(sim, net):
    net.visualized = True

    program = main_viewer_tk(input_net=net)
    sim.gui_program = program

    sim.gui_program.unit_viewer = FrameExamplesProgram(
        input_net=net,
        parent=sim.gui_program.window
    )

    test_clicks(sim.gui_program.unit_viewer)
    test_redrawing(sim.gui_program.unit_viewer)
    test_value_menu(sim.gui_program.unit_viewer)

    if net.max_ticks > 1:
        test_time_buttons(sim.gui_program.unit_viewer)

    sim.gui_program.unit_viewer.window.update_idletasks()
    sim.gui_program.unit_viewer.window.update()

    sim.gui_program.unit_viewer.window.destroy()
    program.window.destroy()

def test_different_nets():
    sim, net = build_xor_network()
    run_unit_viewer_test(sim, net)

    sim, net = build_task_net()
    run_unit_viewer_test(sim, net)

    sim, net = build_digits_network()
    run_unit_viewer_test(sim, net)

    sim, net = build_filler_network()
    run_unit_viewer_test(sim, net)

    sim, net = build_filler_srbptt_network()
    run_unit_viewer_test(sim, net)

    sim, net = build_rand_network()
    run_unit_viewer_test(sim, net)

    sim, net = build_boltz_network()
    run_unit_viewer_test(sim, net)

if __name__ == "__main__":
    test_different_nets()
