import math
import os
import tkinter as tk
from dataclasses import dataclass
from collections import defaultdict
from functools import partial
from time import sleep
from tkinter import filedialog
from tkinter import ttk

import pyscreenshot
import webcolors as wc
from ipykernel.eventloops import register_integration

from PyLens.backend.network import Network

# plotRow/layout logic is GUI-independent and lives in src.gui.unit_layout
from PyLens.gui.unit_layout import (
    BlankCell,
    UnitCell,
    apply_plotrow,
    apply_plotrow_from_command,
    reset_plot_state,
)

LIGHT_GREY_HEX = "#d3d3d3"
BLUE_HEX = "#0000ff"
RED_HEX = "#ff0000"
YELLOW_HEX = "#FFD700"
BLACK_HEX = "#000000"
WHITE_HEX = "#ffffff"
PINK_HEX = "#ff69b4"


### NOTE
# UnitCell / BlankCell and plotRow parsing live in src.gui.unit_layout.


# helper
def scalar_value(value):
    return value.item() if hasattr(value, "item") else value


class FrameExamplesProgram():
    """
    Manage running examples through a network and visualizing unit values
    in the unit viewer window.
    """

    def __init__(self, input_net, parent, user_defined_ex_history=None, q=None, stop_event=None, cell_size=9, cell_spacing=3):
        """
        Initialize the FrameExamplesProgram object.

        Attributes:
            curr_ex_history (list): History of group values for the current example.
            curr_ex_output_derivs (list): Output derivatives for each group in the
                current example.
            curr_ex_history_per_tick (list): Output matrices for each tick in the
                current example.
            input_net (Network): The network being visualized.
            input_example_list (list[Example]): The examples available in the
                currently selected example set.
            ev_curr (int): Index of the current event (starting from 0).
            ev_total (int): Total number of events in the current example.
            ex_time_curr (str): Current tick time within the example, formatted as
                ``"interval:tick"``.
            ex_time_total (str): Total time of the current example.
            ev_time_curr (str): Current tick time within the current event.
            ev_time_total (str): Total time of the current event.
            _curr_tick_idx (int): Index of the current tick within the example.
            _ticks_per_ex (int): Number of ticks per example.
            _ticks_per_ev (int): Number of ticks per event.
            example_set_idx (int): Index of the currently selected example set
                (0 = training, 1 = testing, >=2 = extra example sets).
            value_selection (int): Index of the currently selected value type
                in the value menu.
            curr_cell_name (str): Name of the currently hovered or selected cell.
            curr_cell_info (str): Text describing the value of the current cell.
            h_slider_loc (int): Horizontal slider position in range [0, 100].
            v_slider_loc (int): Vertical slider position in range [0, 100].
            color_palette (int): Index of the current color palette.
            frozen_cell (tuple | None): Information about the frozen cell, or
                None if no cell is frozen.
        Args:
            input_net (Network): The input network used for processing.
            parent (tk.Tk): The parent window.
            user_defined_ex_history (list, optional): A list of user-defined example histories. Defaults to None.
            q (optional): A queue for communication between threads. Defaults to None.
            stop_event (Event, optional): An event for stopping the thread. Defaults to None.
        """
        self.initial_values = []
        self.upper_textbox = None
        self.lower_textbox = None
        self.box_0_val = None
        self.box_1_val = None
        self.box_2_val = None
        self.box_3_val = None
        self.box_4_val = None
        self.box_5_val = None
        self.six_btns = []  # btns for event, example time, event time
        self.stop_event = stop_event
        self.user_defined_ex_history = user_defined_ex_history
        self.input_net = input_net
        self.q = q
        self.curr_ex_history = []
        self.curr_ex_input_history = []
        self.curr_ex_output_derivs = []
        self.curr_ex_input_derivs = []
        self.btn_lst = []
        if self.input_net.training_set is not None:
            self.example_set_idx = self.input_net.training_sets.index(self.input_net.training_set)
        else:
            self.example_set_idx = 0
        self.value_selection = 0  # if this is updated to 5 (output deriv) or 6 (input deriv)
        # 0 for "Examples", 1 for "Weight Updates", 2 for "Progress Reports"
        # 3 for Training and Testing, # 4 for "Never"
        # currently only 1 and 2 are implemented
        # it's defaulted to "Never update"
        self.update_display_option = 0
        self.curr_cell_name = ""
        self.curr_cell_info = ""
        self.curr_cell_mean = ""
        self.curr_cell_variance = ""
        self.curr_cell_meanabs = ""
        self.curr_cell_meandist = ""
        self.curr_cell_max = ""
        self.h_slider_loc = 50
        self.v_slider_loc = 50

        self._curr_tick_idx = 0
        self._curr_ex_idx = 0

        self.group_index_dict = {"bias": 0, "input": (1, 0, None), "hidden": (-2, 0, 0), "output": (-1, None, 0),
                                 "target": 3, "elman":(2, None, None)}
        self.plot_width = 0
        self.plot_height = 0

        self.num_colors = 101
        # for different color palette
        self.color_dict = {0: (BLUE_HEX, RED_HEX),  # BLUE_BLACK_RED
                           1: (BLUE_HEX, YELLOW_HEX),  # BLUE_RED_YELLOW
                           2: (BLACK_HEX, WHITE_HEX),  # BLACK_GRAY_WHITE
                           3: (BLACK_HEX, WHITE_HEX)}  # HINTON

        # default to Blue-Black-Red
        self.color_palette = 0
        self.color_low = self.color_dict[self.color_palette][0]
        self.color_high = self.color_dict[self.color_palette][1]

        # initialize all the available colors
        self.color_map = {0: [None] * (self.num_colors), 1: [None] * (self.num_colors),
                          2: [None] * (self.num_colors), 3: [None] * (self.num_colors)}

        # attributes related to color slider
        # the input range is assumed to be 0 ~ 1 for now
        # will be changed in the future
        # these two are always given as a real number
        self.inp_lower_bound = 0
        self.inp_upper_bound = 1

        self.output_lower_bound = 0
        self.output_upper_bound = 1
        self.filled_map_idx = []

        self.frozen_cell = None
        self.window = tk.Toplevel(parent)
        self.window.title("unit viewer")
        self.window.minsize(850, 600)

        self.output_nodes_border = []
        self.output_color_border = {}

        self.nodes = {}
        self.nodes_color = {}

        self.cell_size = cell_size
        self.cell_spacing = cell_spacing

        self.curr_base_oY = 30
        self.prev_base_oY = self.curr_base_oY
        self.curr_base_eY = 60
        self.multi_inp_base = 0
        # training or testing
        self.is_training = False
        self.training_mode_radiobutton_val = tk.IntVar(self.window, value=1) # testing radio button checked by default
        self.left_click = False
        self.right_click = False
        self.right_click_id = -1 # node that was most recently right-clicked
        self.prev_right_click_id= -1 # node that was second most recently right-clicked
        self.selected_node = None  # node that was most recently left-clicked
        self.selected_outline_id = None
        self.selected_outline_color = None
        self.right_click_node_type = ""  # one of input, hidden, output
        self.incoming_val = {}
        self.outgoing_val = {}
        self.output_col_val = None

        self.max_num_col = 0
        for group in self.input_net.groups:
                # Initialize node's ids lists
                self.nodes[group.name] = []
                self.nodes_color[group.name] = []
                compare_val = group.num_cols if group.num_cols is not None else 20
                if compare_val > self.max_num_col:
                    self.max_num_col = compare_val

        # attributes related to screen shot
        # screen_w = self.window.winfo_screenwidth()
        # screen_h = self.window.winfo_screenheight()
        # screen = pyscreenshot.grab()
        # self.pyautogui_w, self.pyautogui_h = screen.size
        # self.horizontal_scale_factor = self.pyautogui_w / screen_w
        # self.vertical_scale_factor = self.pyautogui_h / screen_h

        # the width of horizontal and vertical scroll bar around canvas
        # will be updated in create_widgets
        self.hbar_height = 0
        self.vbar_width = 0


        # attribute of canvas size
        self.current_c_width = 0

        # stores name of units, if any
        self.unit_names_by_group = {}

        for g in self.input_net.groups:
            if len(g.unit_names) != 0:
                self.unit_names_by_group[g.name] = g.unit_names
            else:
                self.unit_names_by_group[g.name] = [""] * g.num_units

        # stores tags of each unit
        self.tags = defaultdict(lambda: [])
        self.drawing = False

        self.output_group_incoming = []
        self.output_group_outgoing = None
        self.inp_hid_group_outgoing = []
        self.inp_hid_group_incoming = []

        self.value_selection_map = {'Outputs and Targets': 0, 'Outputs': 1, 'Targets': 2,
                                    'Inputs': 3, 'External Inputs': 4, 'Output Derivatives': 5,
                                    'Input Derivatives': 6, 'Gains': 7, 'Link Weights': 8,
                                    'Link Derivs': 9, 'Link Deltas': 10}
        self.value_selection_radiobutton_val = tk.IntVar(self.window)
        self.external_input = None
        self.yellow_border_id = None
        self.listbox_exist=False
        self.listbox = None
        self.fill_color_map()
        self.create_widgets()
        self.create_menu()
        # these two attributes must be defined after widgets are created
        c.update()
        self.canvas_w = c.winfo_width()
        self.canvas_h = c.winfo_height()
        c.bind("<Configure>", lambda event: self.center_canvas_items())
        self.window.after_idle(self.center_canvas_items)
        # self.window.mainloop()

    @property
    def curr_ex_output_history_per_tick(self):
        """
        Return the output history per tick.
        """
        return list(map(list, zip(*self.curr_ex_history)))

    @property
    def curr_ex_input_history_per_tick(self):
        """
        Return the input history per tick.
        """
        return list(map(list, zip(*self.curr_ex_input_history)))

    @property
    def input_example_list(self):
        """
        Return the list of examples based on the selected example set.
        """
        example_sets = self.input_net.training_sets + self.input_net.testing_sets
        return example_sets[self.example_set_idx].example

    @property
    def input_example_index_list(self):
        """
        Return the list of example indices based on the selected example set.
        """
        example_sets = self.input_net.training_sets + self.input_net.testing_sets
        return example_sets.example_iterator.index_list

    @property
    def ev_total(self): # Return the total number of events.
        return len(self.input_example_list[0].event)

    @property
    def ev_curr(self): # Return the current event index.
        actual_curr_tick = self._curr_tick_idx
        if type(self.input_net).__name__ == "BoltzmannMachine":
            return self.input_net.event_list[actual_curr_tick]
        if type(self.input_net).__name__ == "ContinuousNetwork" and actual_curr_tick>0:
            actual_curr_tick -= 1
        return int(actual_curr_tick // self._ticks_per_ev)

    @property
    def max_time(self): # Return the maximum time for the current example.
        if self.input_example_list[0].event[0].max_time is not None:
            max_time = self.input_example_list[0].event[0].max_time
        elif self.input_example_list[0].set.max_time is not None:
            max_time = self.input_example_list[0].set.max_time
        else:
            max_time = self.input_net.max_example_time
        return max_time

    @property
    def _ticks_per_ev(self):    # Return the number of ticks per event.
        if type(self.input_net).__name__ == "BoltzmannMachine":
            return self.input_net.ticks_per_event[self.ev_curr]
        else:
            return int(self.max_time * self.input_net.ticks_per_interval)

    @property
    def _ticks_per_ex(self):    # Return the number of ticks per example.
        # Boltzmann Machine ticks depend on settling time
        if type(self.input_net).__name__ == "BoltzmannMachine":
            ticks = self.input_net.ticks_on_example - 1
        else:
            ticks = self._ticks_per_ev * self.ev_total
            if type(self.input_net).__name__ == "ContinuousNetwork":
                ticks += 1
        return ticks

    def tick__to_nodes_id(self):    # Convert tick index to node ids.
        res = {}
        for i in range(self._ticks_per_ex):
            res[i] = {"input_nodes": [], "output_nodes": [],
                      "output_nodes_border": [], "hidden_nodes": [], "bias_nodes": []}
        return res

    @property
    # the third box from the left
    def ex_time_curr(self): # Return the current tick's time relative to the example.
        interval = (self._curr_tick_idx + 1) // self.input_net.ticks_per_interval
        ticks = (self._curr_tick_idx + 1) % self.input_net.ticks_per_interval
        return str(int(interval)) + ":" + str(int(ticks))

    @property
    # the fourth box from the left
    def ex_time_total(self):    # Return the total time spent on the current example.
        t = str(int(self.ev_total * self.max_time))
        # the first tick is for setting up initial output
        # so the first valid tick starts at 0:2
        if type(self.input_net).__name__ == "ContinuousNetwork" or type(self.input_net).__name__ == "BoltzmannMachine":
            return t + ":1"
        return t + ":0"

    @property
    def _curr_tick_idx_relative_to_curr_ev(self):   # Return the index of the current tick relative to the current event.
        offset = 0
        if type(self.input_net).__name__ == "ContinuousNetwork":
            offset = 1
        return (self._curr_tick_idx - offset) % self._ticks_per_ev if self._curr_tick_idx > 0 else 0

    @property
    def ev_time_curr(self): # Return the current tick's time relative to the event it belongs to.
        interval = int((self._curr_tick_idx_relative_to_curr_ev + 1) // self.input_net.ticks_per_interval)
        ticks = int((self._curr_tick_idx_relative_to_curr_ev + 1) % self.input_net.ticks_per_interval)
        return str(interval) + ":" + str(ticks)

    @property
    # the sixth box from the left
    def ev_time_total(self): # Return the total time spent on the current event.
        t = str(int(self.max_time))
        if type(self.input_net).__name__ == "BoltzmannMachine":
            return t + ":1"
        return  t + ":0"

    @property
    def curr_tick_history(self):
        """
        if we select output or target in value menu, output history.
        in input only mode this is using input history.
        """
        if self.value_selection == 3:  # input only
            return self.curr_ex_input_history_per_tick[self._curr_tick_idx]
        elif self._curr_ex_idx is not None and self.curr_ex_output_history_per_tick != []:
            return self.curr_ex_output_history_per_tick[self._curr_tick_idx]
        else:
            return [[]] * 5

    @property
    def curr_tick_bias_lst(self):
        """
        Return the bias list for the current tick.
        """
        return self.curr_tick_history[0]

    @property
    def curr_tick_output_lst(self):
        """
        Return the output list for the current tick.
        """
        if self.value_selection == 3:
            return self.curr_tick_history[-1]
        return self.curr_tick_history[-2]

    @property
    def curr_tick_target_lst(self):
        """
        Return the target list for the current tick.
        """
        if self.value_selection != 3:
            return self.curr_tick_history[-1]

    @property
    def num_input_nodes(self):
        """
        Return the number of input nodes.
        """
        return len(self.curr_tick_inp_lst)

    @property
    def num_output_nodes(self):
        """
        Return the number of output nodes.
        """
        return len(self.curr_tick_output_lst)

    @property
    def num_hidden_nodes(self):
        """
        Return the number of hidden nodes.
        """
        return len(self.curr_tick_hidden_lst)

    @property
    def output_and_target_flag(self):
        return self.value_selection == 0

    @property
    def output_only_flag(self):
        return self.value_selection == 1

    @property
    def target_only_flag(self):
        return self.value_selection == 2

    @property
    def input_only_flag(self):
        return self.value_selection == 3

    @property
    def external_input_only_flag(self):
        return self.value_selection == 4

    @property
    def output_derivs(self):
        return self.value_selection == 5

    @property
    def input_derivs(self):
        return self.value_selection == 6

    @property
    def link_weight_flag(self):
        return self.value_selection == 8

    @property
    def link_derivs_flag(self):
        return self.value_selection == 9

    @property
    def link_delta_flag(self):
        return self.value_selection == 10

    @property
    def link_flag(self):
        return self.link_weight_flag or self.link_delta_flag or self.link_derivs_flag

    @property
    def output_count(self):
        return len(self.input_net.output_groups)

    @property
    def input_count(self):
        return len(self.input_net.input_groups)

    @property
    def bias_count(self):
        return 1 if self.input_net.groups[0].group_type != None else 0

    @property
    def hidden_count(self):
        """
        Return the number of hidden nodes.
        """
        return len(self.input_net.groups) - (self.output_count + self.input_count + self.bias_count)

    @property
    def curr_tick_inp_lst(self):
        """
        Return the input list for the current tick.
        """
        if self.bias_count == 1:
            return self.curr_tick_history[1:1 + self.input_count]
        return self.curr_tick_history[:self.input_count]

    @property
    def curr_tick_external_inp_lst(self):
        """
        Return the external input list for the current tick.
        """
        return self.external_input[self.ev_curr].tolist()

    @property
    def curr_tick_hidden_lst(self):
        """
        Return the hidden list for the current tick.
        """
        hidden_start = self.input_count + self.bias_count
        hidden_end = hidden_start + self.hidden_count
        return self.curr_tick_history[hidden_start:hidden_end]
    
    def check_for_update(self, last_example_trained, s):
        """
        Check if the display should be updated based on the current display option.

        Args:
            last_example_trained (Example): The last example trained.
            s (str): The string to check against the display option.

        """
        update_flag = (self.update_display_option == 0 and s == "example")
        update_flag = update_flag or (self.update_display_option == 1 and s == "weight update")
        update_flag = update_flag or (self.update_display_option == 2 and s == "completion of a batch")
        update_flag = update_flag or (self.update_display_option == 3 and s == "progress report")
        update_flag = update_flag or (self.update_display_option == 4 and s == "training and testing")
        if update_flag:
            if self.update_display_option == 2:
                first_item_idx = self.input_example_index_list[0]
                self.listbox.selection_clear(0, tk.END)
                self.listbox.select_set(first_item_idx)
                self.listbox.see(first_item_idx) 
            if self.update_display_option == 0:
                self._curr_ex_idx = self.input_example_list.index(last_example_trained)
                self.listbox.selection_clear(0, tk.END)
                self.listbox.select_set(self._curr_ex_idx)
                self.draw_tick()
            else:
                self.unit_viewer.update_canvas()
        self.stop_event.set()

    def run_example_wrapper(self, testing_mode=False, example_to_run=None):
        """
        Run the example and update the display.
        """
        self.curr_ex_history = []
        self.curr_ex_input_history = []
        self.curr_ex_input_derivs = []
        self.curr_ex_output_derivs = []
        if self.user_defined_ex_history is not None:
            self.curr_ex_history = self.user_defined_ex_history
            return
        if example_to_run is None:
            real_ex_idx = self._curr_ex_idx
        else:
            real_ex_idx = example_to_run
        event_res = self.input_net.standard_net_train_example(self.input_example_list[real_ex_idx], test=testing_mode)
        if type(self.input_net).__name__ == "ContinuousNetwork" or type(self.input_net).__name__ == "BoltzmannMachine":
            self.parse_example_result_from_history(self.input_example_list[real_ex_idx])
        else:
            self.parse_example_result_from_train_example_result(event_res)
        self.update_derivatives_from_output_history()
        self.update_current_cell_name_and_info()
        self.parse_external_input()
        self.parse_input_history()

    def parse_external_input(self):
        """
        Parse the external input.
        """
        for g in self.input_net.groups:
            if g.group_type == "input":
                self.external_input = g.external_input_history

    def parse_example_result_from_train_example_result(self, event_res):
        """
        Update self.curr_ex_input_derivs, self.curr_ex_output_derivs, self.curr_ex_history.
        
        Args:
            event_res (list): The return value of standard_net_train_example.
        """
        # add the bias group because standard_net_train_example doesn't return it
        for g in self.input_net.groups:
            if g.group_type == "bias":
                bias_event_res = [g.output_history.tolist()]
        num_ticks = len(event_res)
        self.curr_ex_history = list(map(list, zip(*event_res)))
        self.curr_ex_history.insert(0, bias_event_res * num_ticks)

    def update_derivatives_from_output_history(self):
        """
        Update self.curr_ex_input_derivs, self.curr_ex_output_derivs.
        """
        for g in self.input_net.groups:
            # update input derivatives
            curr_input_deriv = g.input_derivs.tolist()
            self.curr_ex_input_derivs.append(curr_input_deriv)

            # update output derivs
            curr_output_deriv = g.output_derivs.tolist()
            self.curr_ex_output_derivs.append(curr_output_deriv)

    def parse_input_history(self):
        """
        Update self.curr_ex_input_history.
        """
        for g in self.input_net.groups:
            if g.group_type == "bias":
                curr_output_history = [g.output_history.tolist()]
                # don't know input of bias so make it
                self.curr_ex_input_history.insert(0, curr_output_history * self._ticks_per_ex)
            else:
                self.curr_ex_input_history.append(g.input_history.tolist()[:self._ticks_per_ex])

    def parse_example_result_from_history(self, example):
        """
        Update self.curr_ex_history, self.curr_ex_input_history.

        Args:
            example (Example): The example to parse.
        """
        for g in self.input_net.groups:
            # update output history for cell values
            curr_output_history = g.output_history.tolist()
            if g.group_type == "bias":
                curr_output_history = [curr_output_history]
            curr_output_history = curr_output_history[:self._ticks_per_ex]
            self.curr_ex_history.append(curr_output_history)

        targets_lst = []
        for i in range(self.ev_total):
            # example is Example remembered from standard_net_train_batch in network.py
            # one target per event,
            targets_np = example.event[i].target_group
            # Events may last different amount of ticks for BMs
            if type(self.input_net).__name__ == "BoltzmannMachine":
                targets_lst += [x.tolist() for x in targets_np] * self.input_net.ticks_per_event[i]
            else:
                targets_lst += [x.tolist() for x in targets_np] * self._ticks_per_ev
        if type(self.input_net).__name__ == "ContinuousNetwork":
            targets_lst.insert(0, [0.5] * len(targets_lst[0]))
        self.curr_ex_history.append(targets_lst)
        self.curr_ex_history[0] = self.curr_ex_history[0] * len(self.curr_ex_history[1])

    def fill_color_map(self):
        """
        Fill the color map based on the selected color palette.
        """
        # if it is already filled then don't need to fill again
        # fill only the ones need for now
        if not (self.color_palette in self.filled_map_idx):
            if self.color_palette == 0:
                blue = 0
                for i in range(self.num_colors):
                    red = i / ((self.num_colors - 1) / 2) - 1
                    if red < 0:
                        blue = -red
                        red = 0
                        self.color_map[0][i] = (0, 0, int(blue * 255))
                    else:
                        self.color_map[0][i] = (int(red * 255), 0, 0)
                        blue = 0

            elif self.color_palette == 1:  # BLUE_RED_YELLOW
                for i in range(self.num_colors):
                    rval = i / (self.num_colors - 1)
                    if rval <= 0.5:
                        green = 0
                        red = rval * 2
                        blue = 1 - red
                        self.color_map[1][i] = (int(red * 255), 0, int(blue * 255))
                    else:
                        red = 1
                        blue = 0
                        green = rval * 2 - 1
                        self.color_map[1][i] = (255, int(green * 255), 0)

            else:  # BLACK_GRAY_WHITE, no HINTON for now
                for i in range(self.num_colors):
                    grey = i / ((self.num_colors - 1) / 2) - 1
                    y = int(127.5 * (grey + 1))
                    self.color_map[2][i] = (y, y, y)

        self.filled_map_idx.append(self.color_palette)

    def inverse_log(self, x):
        """
        Inverse of the sigmoid function.
        
        Args:
            x (float): The input value.
        """
        if x <= 0:
            return -float('inf')
        elif x >= 1:
            return float('inf')
        else:
            return math.log(x / (1 - x))

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def cast_cell_value_from_optional_bound(self, cell_val, lower_bound=None, upper_bound=None):
        """
        transform the raw cell value given optional lower bound and upper bound.

        Args:
            cell_val (float): raw cell value.
            lower_bound (float, optional): the lowest possible cell value allowed to display. Defaults to None.
            upper_bound (float, optional): the highest possible cell value allowed to display. Defaults to None.

        Returns:
            float: casted value
        """
        casted_cell_val = cell_val
        if lower_bound is not None:
            if upper_bound is not None:
                casted_cell_val = (cell_val - lower_bound) / (upper_bound - lower_bound)
                casted_cell_val = self.inverse_log(casted_cell_val)
            else:  # if only lower_bound is given
                casted_cell_val = math.log(cell_val - lower_bound)

        elif upper_bound is not None:  # only max bound, may never occurs
            casted_cell_val = -math.log(upper_bound - cell_val)

        elif upper_bound is not None:  # only max bound, may never occurs
            casted_cell_val = -math.log(upper_bound - cell_val)

        else:  # none of lower bound and upper bound is given
            # maps (-inf, inf) to (0, 1)
            casted_cell_val = self.sigmoid(cell_val)

        return casted_cell_val

    def value_to_color(self, cell_val, type, link_weight=False):
        """
        Calculate the color given cell_val, cell type, and color temperature
        (indicated by slider).

        Args:
            cell_val (str): raw value of unit.
            type (str): one of:
                - "hidden"
                - "output"
                - "target"
                - "input"
                - "bias"
                - "input_derivs"
                - "output_derivs"
                - "link_derivs"
            link_weight (bool, optional): True if the cell is a link weight. Defaults to False.
        """
        temperature = self.v_slider_loc / 4.5
        max_temperature = 20.5
        min_temperature = 1
        casted_cell_val = float('inf')

        if not link_weight:
            self.output_lower_bound = 0
            self.inp_lower_bound = 0

        # in weight mode the link weight is between -1 and 1
        # the negative values are blue, positive values are red
        else:
            self.output_lower_bound = -1
            self.inp_lower_bound = -1

        if cell_val is None:
            return BLUE_HEX

        # add this to handle pytorch tensor
        if hasattr(cell_val, "item"):
            cell_val = cell_val.item()

        if type in ["hidden", "output", "target", "elman"]:

            if cell_val == self.output_upper_bound:
                return self.color_high
            elif cell_val == self.output_lower_bound:
                return self.color_low
            casted_cell_val = self.cast_cell_value_from_optional_bound(cell_val=cell_val,
                                                                       lower_bound=self.output_lower_bound,
                                                                       upper_bound=self.output_upper_bound)


        elif type in ["input", "bias"]:  # raw -> cast to a given range
            if cell_val == self.inp_lower_bound:
                return self.color_low
            elif cell_val == self.inp_upper_bound:
                return self.color_high
            else:  # if they are not given then do the same trick as output
                casted_cell_val = self.cast_cell_value_from_optional_bound(cell_val=cell_val,
                                                                           lower_bound=self.inp_lower_bound,
                                                                           upper_bound=self.inp_upper_bound)

        elif type in ["input_derivs", "output_derivs", "link_derivs"]:  # take  raw value
            casted_cell_val = cell_val

        else:  # more to it later...
            pass

        # color = PINK_HEX
        # if we saturate the node and the node value isn't the middle value (black)
        # then give it the color of either end (pure blue or pure red for example)
        if temperature >= max_temperature:
            if casted_cell_val > 0:
                color = self.color_map[self.color_palette][self.num_colors - 1]
            elif casted_cell_val < 0:
                color = self.color_map[self.color_palette][0]
            else:
                color = self.color_map[self.color_palette][int(self.num_colors / 2)]

        elif temperature <= min_temperature and casted_cell_val != 0:
            color = self.color_map[self.color_palette][int(self.num_colors / 2)]

        else:
            try:
                y = int(100 / (1 + math.exp(-casted_cell_val * temperature)))
            except:
                y = 50
            color = self.color_map[self.color_palette][y]
        return wc.rgb_to_hex(color)

    def reset_plot(self, canvas):
        """
        Reset the plot.

        Args:
            canvas (tk.Canvas): The canvas to reset.
        """
        c.delete("all")
        canvas.delete("all")

    def update_canvas(self):
        """
        Update the canvas. This is the main function that updates the display.
        """
        self.window.update_idletasks()
        if self.drawing:
            return
        elif (not self.input_net) or (self._curr_ex_idx is None):
            self.disable_all_btns()
            self.blank_out_all_textboxes()

        if self._curr_ex_idx is not None:
            s = self.input_example_list[self._curr_ex_idx].set

        self.update_textboxes()

        if self._curr_tick_idx <= 0 or (self._curr_ex_idx is None):
            for i in range(3):
                self.btn_lst[i].configure(state='disabled')
        else:
            for i in range(3):
                self.btn_lst[i].configure(state='normal')

        if self._curr_tick_idx >= self._ticks_per_ex - 1 or (self._curr_ex_idx is None):
            for i in range(3, 8, 1):  # the three -> buttons, and the two lower ones
                self.btn_lst[i].configure(state='disable')
        else:
            for i in range(3, 8, 1):  # the three -> buttons, and the two lower ones
                self.btn_lst[i].configure(state='normal')
        self.window.update_idletasks()
        self.curr_ex_history = []
        self.curr_ex_input_history = []
        self.curr_ex_input_derivs = []
        self.curr_ex_output_derivs = []
        self.draw_tick()

    def create_menu(self):
        """
        Create all menu items and their callbacks.
        """
        menubar = tk.Menu(self.window)
        self.window.configure(menu=menubar)

        viewer_menu = tk.Menu(menubar, tearoff=0)
        example_set_menu = tk.Menu(menubar, tearoff=1)
        procedure_menu = tk.Menu(menubar, tearoff=2)
        value_menu = tk.Menu(menubar, tearoff=3)
        palette_menu = tk.Menu(menubar, tearoff=4)
        menubar.add_cascade(label="Viewer", menu=viewer_menu)

        update_after_menu = tk.Menu(viewer_menu)
        update_option_list = ['Examples', 'Weight Updates', 'Batch', 'Progress Reports',
                              'Training and Testing', 'Never']
        update_choice = tk.IntVar()
        update_choice.set(self.update_display_option)

        def update_after_choice():
            self.update_display_option = update_choice.get()

        menubar.add_cascade(label="Value", menu=value_menu)
        for j in range(len(update_option_list)):
            update_after_menu.add_radiobutton(label=update_option_list[j], state='normal', value=j,
                                              variable=update_choice,
                                              command=update_after_choice)

        self.cell_size_var = tk.IntVar(value=self.cell_size)
        self.cell_spacing_var = tk.IntVar(value=self.cell_spacing)

        cell_size_menu = tk.Menu(viewer_menu)
        for name in range(20):
            cell_size_menu.add_radiobutton(
                label=str(name),
                value=name,
                variable=self.cell_size_var,
                command=partial(self.update_cell_size, name),
            )

        cell_spacing_menu = tk.Menu(viewer_menu)
        for name in range(20):
            cell_spacing_menu.add_radiobutton(
                label=str(name),
                value=name,
                variable=self.cell_spacing_var,
                command=partial(self.update_cell_spacing, name),
            )

        viewer_menu.add_cascade(label="Update After", menu=update_after_menu)
        viewer_menu.add_cascade(label="Cell Size", menu=cell_size_menu)
        viewer_menu.add_cascade(label="Cell Spacing", menu=cell_spacing_menu)
        viewer_menu.add_command(label="Update", command=self.update_canvas)
        viewer_menu.add_command(label="Refresh", command=self.refresh_canvas)
        # viewer_menu.add_command(label="Reset", command=partial(self.reset_plot, c))

        def print_callback():
            """
            Capture the whole Canvas and save it in a user specified location.
            If all units are displayed, postscript, jpeg, png and pdf are supported.
            If some portion of units is not displayed, only postscript format is supported.

            Among all file formats, only postscript is vectorized file, the rest are raster.
            """
            # change if GUI widows size is changed
            # subtract the width of scroll bar in order not to capture the scroll bar
            otf_w = OUTPUT_TARGET_FRAME.winfo_width() - self.vbar_width
            otf_h = OUTPUT_TARGET_FRAME.winfo_height() - self.hbar_height

            left = self.window.winfo_rootx() + OUTPUT_TARGET_FRAME.winfo_x()
            top = self.window.winfo_rooty() + OUTPUT_TARGET_FRAME.winfo_y()

            all_file_type_flag = True

            if all_file_type_flag:
                absolute_path = filedialog.asksaveasfilename(
                        initialdir="/", title="Select file",
                        defaultextension='.ps',
                        filetypes=[("Postscript file", "*.ps"),
                                   ("Jpeg image", "*.jpg"),
                                   ("PNG image", "*.png"), ("PDF file", "*.pdf"),
                                   ("all", "*.*")]
                        )
                # if user didn't close the file dialog
                if absolute_path != "":
                    _, ext = os.path.splitext(absolute_path)
                    if ext[1:] == "ps":
                        c.update()
                        c.postscript(file=absolute_path, colormode='color')
                    else:
                        import pyautogui # lazy import pyautogui so that unit tests doesn't need DISPLAY
                        screen_w = self.window.winfo_screenwidth()
                        screen_h = self.window.winfo_screenheight()

                        self.pyautogui_w, self.pyautogui_h = pyautogui.size()

                        self.horizontal_scale_factor = self.pyautogui_w / screen_w
                        self.vertical_scale_factor = self.pyautogui_h / screen_h
                        # left, top border of frame
                        # right border of frame or right border of the screen, whichever comes first
                        # same applies to bottom border
                        bbox = (int(left * self.horizontal_scale_factor),
                                int(top * self.vertical_scale_factor),
                                min(int(((left + otf_w) * self.horizontal_scale_factor)), self.pyautogui_w),
                                min(int((top + otf_h) * self.vertical_scale_factor), self.pyautogui_h))
                        img = pyscreenshot.grab(bbox)
                        img.save(absolute_path)

        viewer_menu.add_command(label="Print...", command=print_callback)
        viewer_menu.add_command(label="Close", command=self.window.destroy)
        menubar.add_cascade(label="Example Set", menu=example_set_menu)

        procedure_menu.add_radiobutton(label="Training", state='normal', value=0, var=self.training_mode_radiobutton_val,
                                       command=partial(self.change_mode_callback, training=True))
        procedure_menu.add_radiobutton(label="Testing",  state='normal', value=1, var=self.training_mode_radiobutton_val,
                                       command=partial(self.change_mode_callback, training=False))  # default testing mode
        menubar.add_cascade(label="Procedure", menu=procedure_menu)

        def update_current_example_set(i):
            self.example_set_idx = i
            self.create_listbox(ex_set_frame)

        example_set_menu.add_command(label="Training Set", state='disabled')

        for idx, ex_set in enumerate(self.input_net.training_sets):
            active_str = ""
            if ex_set == self.input_net.training_set:
                active_str = " (active)"
            name = ex_set.name
            example_set_menu.add_radiobutton(
                label=f"  {name}"+active_str,  # indent for visual clarity
                command=partial(update_current_example_set, i=idx)
            )

        example_set_menu.add_command(label="Testing Set", state='disabled')

        # Offset the index after training sets
        offset = len(self.input_net.training_sets)

        for idx, ex_set in enumerate(self.input_net.testing_sets):
            active_str = ""
            if ex_set == self.input_net.testing_set:
                active_str = " (active)"
            name = ex_set.name
            example_set_menu.add_radiobutton(
                label=f"  {name}"+active_str,
                command=partial(update_current_example_set, i=offset + idx)
            )
        value_list = [('Outputs and Targets', 'normal'), ('Outputs', 'normal'),
                      ('Targets', 'normal'), ('Inputs', 'normal'),
                      ('External Inputs', 'normal'),
                      ('Output Derivatives', 'normal'),
                      ('Input Derivatives', 'normal'), ('Gains', 'disable'),
                      ('Link Weights', 'normal'), ('Link Derivs', 'normal'),
                      ('Link Deltas', 'normal')]

        for i in range(len(value_list)):
            name, state = value_list[i]
            value_menu.add_radiobutton(label=name, state=state, var=self.value_selection_radiobutton_val,
                                       value=i, command=partial(self.update_value_choice, name))

        menubar.add_cascade(label="Palette", menu=palette_menu)
        color_scheme_lst = ['Blue-Black-Red', 'Blue-Black-Yellow',
                            'Black-Gray-White']

        self.palette_choice = tk.IntVar(
            self.window,
            value=self.color_palette
        )

        for i, name in enumerate(color_scheme_lst):
            palette_menu.add_radiobutton(
                label=name,
                variable=self.palette_choice,
                value=i,
                command=partial(self.update_color_palette, i)
            )

        palette_menu.add_command(label='Hinton Diagram', state='disabled')

    def change_mode_callback(self, training):
        # if currently in training mode but we are changing to testing
        if self.is_training and not training:
            self.is_training = False
            self.run_example_wrapper(testing_mode=True)
        # or the other way around
        if not self.is_training and training:
            self.is_training = True
            self.run_example_wrapper(testing_mode=False, example_to_run=self._curr_ex_idx)
        # If current displayed tick is higher than new mode ticks, clip displayed tick
        if self._curr_tick_idx > self._ticks_per_ex - 1:
            self._curr_tick_idx = self._ticks_per_ex - 1
        self.update_canvas()

    @register_integration('tk')
    def loop_tk(kernel):
        """
        Start a kernel with the Tk event loop.

        Args:
            kernel (Kernel): The kernel to run.
        """
        from tkinter import Tk

        # Tk uses milliseconds
        poll_interval = int(1000 * kernel._poll_interval)

        # For Tkinter, we create a Tk object and call its withdraw method.
        class Timer(object):
            def __init__(self, func):
                self.app = Tk()
                self.app.withdraw()
                self.func = func

            def on_timer(self):
                self.func()
                self.app.after(poll_interval, self.on_timer)

            def start(self):
                self.on_timer()  # Call it once to get things going.
                self.app.mainloop()

        kernel.timer = Timer(kernel.do_one_iteration)
        kernel.timer.start()

    def update_color_palette(self, x):
        """
        Change the current color scheme to x and draw the canvas.

        Args:
            x (int): The new color scheme.
        """
        self.color_palette = x
        self.color_low = self.color_dict[x][0]
        self.color_high = self.color_dict[x][1]
        self.fill_color_map()
        self.draw_tick()

    def refresh_canvas(self):
        self.current_c_width = c.winfo_width()
        self.draw_tick()
        self.window.after_idle(self.center_canvas_items)

    def refresh_and_centering(self):
        self.current_c_width = c.winfo_width()
        self.draw_tick()
        self.window.after_idle(self.center_canvas_items)

    def create_labels(self, parent):
        """
        Create the top bar labels.

        Args:
            parent: The frame within which to create labels.
        """
        # fixed component columns
        for col in [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15]:
            parent.columnconfigure(col, weight=0)

        # four expandable spacers:
        # left, between time/cell, between cell/arrows, right
        for col in [0, 7, 9, 16]:
            parent.columnconfigure(col, weight=1, uniform="spacer")

        textbox_specs = {
            "information": [self.ev_curr, self.ev_total, self.ex_time_curr,
                            self.ex_time_total, self.ev_time_curr,
                            self.ev_time_total],
            "width": [2, 2, 4, 4, 4, 4],
            "row": [1, 1, 1, 1, 1, 1],
            "column": [1, 2, 3, 4, 5, 6],
            "state": ['readonly', 'readonly', 'readonly', 'readonly', 'readonly', 'readonly'],
            "padding": [(10, 0), None, (5, 0), None, (5, 0), None]
        }

        label_specs = {
            "text": ["Event", "Example Time", "Event Time"],
            "column": [1, 3, 5],
            "padx": [2, 2, 2]
        }

        colspan = 2
        for j in range(3):
            lb = ttk.Label(parent, text=label_specs["text"][j])
            lb.grid(row=0, column=label_specs["column"][j],
                    columnspan=colspan, padx=label_specs["padx"][j])

        arrow_widgets = {
            "text": ["<<<", "<<", "<", ">", ">>", ">>>", "|>", "|>>"],
            "column": [10, 11, 12, 13, 14, 15, 14, 15],
            "row": [0, 0, 0, 0, 0, 0, 1, 1],
            # state will be changed as time goes
            "state": ['normal', 'normal', 'normal', 'normal', 'normal',
                      'normal', 'normal', 'normal'],
            "padding": [(0, 0), None, (0, 5), (5, 0), None, (0, 20), None,
                        (0, 20)],
            "command": [partial(self.step_to_either_end_of_example_callback, forward=False),
                        partial(self.step_to_either_end_of_event_callback, forward=False),
                        partial(self.step_tick_callback, forward=False),
                        partial(self.step_tick_callback, forward=True),
                        partial(self.step_to_either_end_of_event_callback, forward=True),
                        partial(self.step_to_either_end_of_example_callback, forward=True),
                        partial(self.step_over_event_callback),
                        partial(self.step_over_example_callback)]
        }

        self.btn_lst = []
        for j in range(8):
            lb = tk.Button(
                parent,
                text=arrow_widgets["text"][j],
                width=2,
                height=1,
                command=arrow_widgets["command"][j],
            )
            self.btn_lst.append(lb)
            lb.grid(
                row=arrow_widgets["row"][j],
                column=arrow_widgets["column"][j],
                padx=2,
            )
            lb.configure(state='normal')

        global UPPER_TEXT, LOWER_TEXT, MEAN_TEXT
        UPPER_TEXT, LOWER_TEXT, MEAN_TEXT = tk.StringVar(), \
                                            tk.StringVar(), tk.StringVar()
        UPPER_TEXT.set("")
        self.upper_textbox = tk.Entry(parent, width=14, textvariable=UPPER_TEXT, justify='center')
        self.upper_textbox.grid(row=0, column=8, padx=2)
        self.upper_textbox.config(state='readonly')

        LOWER_TEXT.set("")
        self.lower_textbox = tk.Entry(parent, width=14, textvariable=LOWER_TEXT, justify='center')
        self.lower_textbox.grid(row=1, column=8, padx=2)
        self.lower_textbox.config(state='readonly')

        # six buttons (event, example time, event time)
        self.box_0_val = tk.Entry(parent, width=2, textvariable=tk.StringVar(), justify='center')  # event
        self.box_1_val = tk.Entry(parent, width=2, textvariable=tk.StringVar(), justify='center')  # event
        self.box_2_val = tk.Entry(parent, width=4, textvariable=tk.StringVar(), justify='center')  # example time
        self.box_3_val = tk.Entry(parent, width=4, textvariable=tk.StringVar(), justify='center')  # example time
        self.box_4_val = tk.Entry(parent, width=4, textvariable=tk.StringVar(), justify='center')  # event time
        self.box_5_val = tk.Entry(parent, width=4, textvariable=tk.StringVar(), justify='center')  # event time

        self.six_btns = [
            self.box_0_val,
            self.box_1_val,
            self.box_2_val,
            self.box_3_val,
            self.box_4_val,
            self.box_5_val,
        ]
        for i in range(len(self.six_btns)):
            self.six_btns[i].grid(
                row=1,
                column=textbox_specs["column"][i],
                padx=2
            )
            if i % 2 == 1:  # second, fourth, and sixth box should be readonly
                self.six_btns[i].config(
                        state='readonly',
                        readonlybackground="white"
                        )

    def disable_all_btns(self):
        """
        Disable all buttons.
        """
        for btn in self.btn_lst:
            btn.configure(state='disabled')

    def blank_out_all_textboxes(self):
        """
        Blank out all text boxes.
        """
        for btn in self.six_btns:
            btn.delete(0, "end")

    def update_current_cell_name_and_info(self):
        """
        Update the value of the cell name and info text boxes.
        """
        # under link mode and no cell is right clicked
        # don't display anything in the two boxes
        if self.link_flag and not self.right_click:
            self.curr_cell_name = ""
            self.curr_cell_info = ""
        elif self.curr_cell_info is None:
            self.curr_cell_info = ""
        elif self.curr_cell_info == 'O:None':
            self.curr_cell_info = "O:0.0"
        elif self.curr_cell_info == 'T:None':
            self.curr_cell_info = "T:0.000000"
        else:
            try:
                prefix, value = self.curr_cell_info.split(":", 1)
                self.curr_cell_info = f"{prefix}: {float(value):.8f}"
            except (ValueError, TypeError):
                pass
        self.upper_textbox.config(state='normal')
        self.upper_textbox.delete(0, "end")
        self.upper_textbox.insert(0, self.curr_cell_name)
        self.upper_textbox.config(
                state='readonly',
                readonlybackground="white"
                )
        self.lower_textbox.config(state='normal')
        self.lower_textbox.delete(0, "end")
        self.lower_textbox.insert(0, self.curr_cell_info)
        self.lower_textbox.config(
                state='readonly',
                readonlybackground="white",
                )

    def center_canvas_items(self):
        bbox = c.bbox("all")
        if bbox:
            c.move(
                "all",
                c.winfo_width() / 2 - (bbox[0] + bbox[2]) / 2,
                0
            )

    def update_cell_spacing(self, cell_info):
        """
        Update the cell spacing.
        """
        self.cell_spacing = int(cell_info)
        c.delete("all")
        self.reset_node_storage()
        self.draw_tick()
        self.window.after_idle(self.center_canvas_items)

    def update_cell_size(self, cell_info):
        """
        Update the cell size.
        """
        self.cell_size = int(cell_info)
        c.delete("all")
        self.reset_node_storage()
        self.draw_tick()
        self.window.after_idle(self.center_canvas_items)

    def update_value_choice(self, new_val):
        """
        Update the value selection.
        """
        self.value_selection = self.value_selection_map[new_val]
        self.value_selection_radiobutton_val.set(self.value_selection)
        self.refresh_canvas()
        OUTPUT_TARGET_FRAME.config(text=new_val)

    def draw_group(self, group):
        """
        The main function that draws the group in the unit viewer.

        Args:
            group (Group): The group to draw. Can be input, hidden, output, bias, or target.
        """
        lst = []
        target_val = [None] * len(group.target)
        if self.output_and_target_flag:
            lst = group.output_history[self._curr_tick_idx] if group.group_type != 'bias' else group.output_history
            target_val = group.target_history[self._curr_tick_idx] if group.group_type != 'bias' else group.target_history
        elif self.output_only_flag or self.link_flag:
            lst = group.output_history[self._curr_tick_idx] if group.group_type != 'bias' else group.output_history
        elif self.target_only_flag:
            lst = group.target_history[self._curr_tick_idx] if group.group_type != 'bias' else group.target_history
        elif self.input_only_flag:
            lst = group.input_history[self._curr_tick_idx] if group.group_type != 'bias' else group.input_history
        elif self.external_input_only_flag:
            lst = group.external_input
        elif self.output_derivs:
            lst = group.output_derivs
        elif self.input_derivs:
            lst = group.input_derivs

        num_node = len(lst)
        if num_node == 0:
            return

        # get the user defined maximum number of columns, else 20
        num_col = 20 if group.num_cols is None else group.num_cols

        # Create nodes the first time draw group is called
        if not self.nodes[group.name]:
            # Draw bias group as single column
            if group.group_type == "bias":
                for k in range(num_node):
                    self.create_node(group=group, col=k, val=lst[k],
                                    target_val=target_val[k], num_col=1, num_node=num_node)
            else:
                for k in range(num_node):
                    self.create_node(group=group, col=k, val=lst[k], target_val=target_val[k],
                                    num_node=num_node, num_col=num_col)

            self.curr_base_oY += 40

        # Only recolor already existing nodes
        else:
            for k in range(num_node):
                self.draw_node(group=group, index=k, val=lst[k], target_val=target_val[k])

    def draw_tick(self):
        """
        Draw the input, hidden, output, bias target layer of the current tick.
        """
        if getattr(self.input_net, "plot_layout", None):
            self.draw_plot_layout()
            return
        self.drawing = True
        self.curr_base_oY = 30
        self.curr_base_eY = 60
        for group in reversed(self.input_net.groups):
            self.draw_group(group)


        # Remove yellow outline if not in link weight mode
        if not self.link_flag:
            self.canvas.itemconfig(self.right_click_id, outline=BLACK_HEX)
        elif self.link_flag and self.right_click_id > -1:
            self.color_weights()
        self.drawing = False
        self.resize_canvas_height()

    def create_node(self, group, col, val, target_val, num_node=None, num_col=None):
        """
        Create a node in the unit viewer.

        Args:
            group (Group): The group to create the node in.
            col (int): The column of the node.
            val (float): The value of the node.
            target_val (float): The target value of the node.
            num_node (int, optional): The number of nodes in the group. Defaults to None.
            num_col (int, optional): The number of columns in the group. Defaults to None.
        """
        # convert to float (including both numpy and pytorch)
        val = scalar_value(val)
        target_val = scalar_value(target_val) if target_val is not None else None

        canvas_width = self.current_c_width if self.current_c_width != 0 else c.winfo_width()
        if not num_col or num_col < 1:
            num_col = int(canvas_width / 17)+1

        output_row = col // num_col
        curr_base_oY = self.curr_base_oY
        
        if group.group_type != "bias":
            if output_row == num_node // num_col:
                num_node_last_row = num_node % num_col
                output_num_start = (num_col - num_node_last_row)/2
                output_num = output_num_start + col%num_col

            else:
                output_num = col if col < num_col else col - (num_col * output_row)

            if group.group_type == 'elman':    # context layer
                output_num += 2
                curr_base_oY = self.curr_base_eY + self.cell_size - 60 + 7

        else:
            output_num = 1

        def _row_offset(output_row):
            """
            Offset on current group's Y coordinate depending on 
            the row being drawn.

            Args:
                output_row (int): The row being drawn.
            """
            return output_row * (13 + ((self.cell_spacing / 5) * 10)) \
                + ((8 + (self.cell_size / 5) * 8) * output_row)

        curr_base_eY = curr_base_oY + 10
        oY = curr_base_oY - self.cell_size + _row_offset(output_row)
        eY = curr_base_eY + self.cell_size + _row_offset(output_row)

        if group.group_type == 'input':
            self.multi_inp_base = eY + 3

        if group.group_type != 'bias':
            if col == num_node - 1:
                self.curr_base_eY = eY + 60
                self.curr_base_oY = oY + 60

        if num_col < self.max_num_col:
            output_num += (self.max_num_col-num_col)/2

        oX = (30 - (self.cell_size / 5) * 5) \
             + (output_num * (8 + ((self.cell_spacing / 5) * 10))) \
             + ((8 + (self.cell_size / 5) * 8) * output_num)
        eX = (40 + self.cell_size) \
             + (output_num * (8 + ((self.cell_spacing / 5) * 10))) \
             + ((8 + (self.cell_size / 5) * 8) * output_num)
        oX = round(oX)
        oY = round(oY)
        eX = round(eX)
        eY = round(eY)

        if self.target_only_flag or self.link_flag or (self.external_input_only_flag and group.group_type != "input"):
            fill_color = LIGHT_GREY_HEX
        elif self.input_derivs:
            fill_color = self.value_to_color(val, "input_derivs")
        elif self.output_derivs:
            fill_color = self.value_to_color(val, "output_derivs")
        else:
            fill_color = self.value_to_color(val, group.group_type)

        # in target only mode, type=output and pass in the target value as the output, target_val=None
        if group.group_type == 'output':
            # if border color is None, no border
            # the change in current unit name
            curr_unit_name = self.unit_names_by_group[group.name][col]
            output_node_color = self.value_to_color(val, "output")

            # if target value isn't given, don't create output border
            # border_color=None under output only, input only or link flag
            # in these three cases border is not created
            border_color = self.value_to_color(target_val, "target") if target_val is not None else None

            if self.link_flag or self.external_input_only_flag:
                output_node_color = LIGHT_GREY_HEX

            if border_color is None:
                # should be proportional to cell size
                t = c.create_rectangle(oX - 2, oY - 2, eX + 2, eY + 2, fill=output_node_color)
                self.nodes[group.name].append(t)
                if self.target_only_flag:
                    self.tags[t] = list(map(str, ("target", col, val, target_val, curr_unit_name)))
                else:
                    self.tags[t] = list(map(str, (group.name, col, val, target_val, curr_unit_name)))

            else:
                t = c.create_rectangle(oX - 2, oY - 2, eX + 2, eY + 2, outline="", fill=output_node_color)
                self.nodes[group.name].append(t)
                # each output node remembers the id of its border
                # TODO display target_val border for any type of group
                s = c.create_rectangle(oX, oY, eX, eY, width=5, outline=border_color, tags='target'+str(col))
                self.output_nodes_border.append(s)
                self.tags[s] = list(map(str, ('target', col, val, target_val, curr_unit_name, oX, oY, eX, eY)))
                self.tags[t] = list(map(str, (group.name, col, val, target_val, curr_unit_name, s, oX, oY, eX, eY)))

            self.output_color_border[col] = border_color

        elif group.group_type == 'input':
            curr_unit_name = self.unit_names_by_group[group.name][col]
            y = c.create_rectangle(oX - 2, oY - 2, eX + 2, eY + 2, fill=fill_color, width=1)
            self.nodes[group.name].append(y)
            self.tags[y] = list(map(str, (group.name, col, val, target_val)))

        elif group.group_type == "hidden" or group.group_type == "elman":
            z = c.create_rectangle(oX - 2, oY - 2, eX + 2, eY + 2, fill=fill_color)
            self.tags[z] = list(map(str, (group.name, col, val, target_val)))
            self.nodes[group.name].append(z)

        else:  # bias
            w = c.create_rectangle(oX - 2, oY - 2, eX + 2, eY + 2, fill=fill_color)
            self.nodes[group.name].append(w)
            self.tags[w] = list(map(str, (group.name, col, val, target_val)))

        self.nodes_color[group.name].append(fill_color)

        if self.frozen_cell == (group.group_type, str(col)):
            c.create_rectangle(oX - 2, oY - 2, eX + 2, eY + 2, fill="", width=1, outline=PINK_HEX)
            self.curr_cell_name = group.name + ":" + str(col) + self.unit_names_by_group[group.name][col]
            self.curr_cell_info = "O" + ":" + str(val)

        self.update_current_cell_name_and_info()

    def draw_node(self, group, index, val, target_val):
        """
        Recolors existing nodes.

        Args:
            group (Group): The group to draw the node in.
            index (int): The index of the node.
            val (float): The value of the node.
            target_val (float): The target value of the node.
        """
        # convert to float (including both numpy and pytorch)
        val = scalar_value(val)
        target_val = scalar_value(target_val) if target_val is not None else None

        if self.link_flag or (self.external_input_only_flag and group.group_type != "input"):
            fill_color = LIGHT_GREY_HEX
        elif self.target_only_flag:
            fill_color = self.value_to_color(val, "target")
        elif self.input_derivs:
            fill_color = self.value_to_color(val, "input_derivs")
        elif self.output_derivs:
            fill_color = self.value_to_color(val, "output_derivs")
        else:
            fill_color = self.value_to_color(val, group.group_type)

        node_id = self.nodes[group.name][index]
        # update tag value
        self.tags[node_id][2] = str(val)
        self.canvas.itemconfig(node_id, fill=fill_color)

        # recolor target border and bring forward/backward in canvas
        # TODO display target_val from any type of group
        if self.output_nodes_border:
            if target_val is not None and group.group_type == "output":
                border_color = self.value_to_color(target_val, "target")
                border_id = self.output_nodes_border[index]
                self.tags[border_id][3] = str(target_val)
                self.canvas.itemconfig(border_id, width=5, outline=border_color)
                self.canvas.tag_raise(border_id)
            elif target_val is None and group.group_type == "output":
                border_color = self.value_to_color(target_val, "target")
                border_id = self.output_nodes_border[index]
                self.canvas.itemconfig(border_id, width=0, outline="", fill="")
                self.canvas.tag_lower(border_id)

    def start_of_event(self):
        """
        Returns True if the current tick is at the start of the current event.
        """

        curr_tick_idx = self._curr_tick_idx
        if type(self.input_net).__name__ == "ContinuousNetwork":
            curr_tick_idx = self._curr_tick_idx - 1
        return curr_tick_idx % self._ticks_per_ev == 0


    def end_of_event(self):
        """
        Returns True if the current tick is at the end of the current event.
       """
        curr_tick_idx = self._curr_tick_idx
        if type(self.input_net).__name__ == "ContinuousNetwork":
            curr_tick_idx = self._curr_tick_idx - 1
        return (curr_tick_idx + 1) % self._ticks_per_ev == 0


    def start_of_example(self):
        """
        Returns True if the current tick is at the start of the current example.
       """
        return self._curr_tick_idx == 0


    def end_of_example(self):
        """
        Returns True if the current tick is at the end of the current example.
       """
        return self._curr_tick_idx == self._ticks_per_ex - 1


    def reset_example_textboxes_callback(self):
        """
       Initialize all six text boxes when loading an example from example set.
       """
        self.update_textboxes()


    def update_textboxes(self):
        """
        Update the three boxes releated to events and ticks, also change the
        status(disable or change to normal) of the buttons according to the
        current tick/event.
        """
        if self.start_of_example():
            for i in range(3):
                self.btn_lst[i].configure(state='disabled')
        else:
            for i in range(3):
                self.btn_lst[i].configure(state='normal')

        if self.end_of_example():
            for i in range(3, 8):
                self.btn_lst[i].configure(state='disabled')
        else:
            for i in range(3, 8):
                self.btn_lst[i].configure(state='normal')

        for btn in self.six_btns:
            btn.config(state='normal')
            btn.delete(0, "end")

        self.box_1_val.insert(0, str(self.ev_total))  # total number of events
        self.box_3_val.insert(0, self.ex_time_total)  # example time total
        self.box_5_val.insert(0, self.ev_time_total)  # event time total

        # if we are setting the initial output value
        if self._curr_tick_idx == 0 and type(self.input_net).__name__ == "ContinuousNetwork":
            self.box_0_val.insert(0, "")  # event
            self.box_2_val.insert(0, "0:1")  # example time
            self.box_4_val.insert(0, "0:1")  # event time

        #  currently at a normal tick
        else:
            self.box_0_val.insert(0, str(self.ev_curr))  # event
            self.box_2_val.insert(0, self.ex_time_curr)  # example time
            self.box_4_val.insert(0, self.ev_time_curr)  # event time

        for i in [1, 3, 5]:  # make 2nd, 4th, and 6th box readonly
            self.six_btns[i].config(
                    state='readonly',
                    readonlybackground="white"
                    )


    def step_to_either_end_of_example_callback(self, forward):
        """
        Jump to the start or end of the current example.

        If ``forward`` is False, jump to the first tick of the example.
        If ``forward`` is True, jump to the last tick of the example.

        Args:
            forward (bool): True if stepping forward, False if stepping back.
        """
        if forward:
            self._curr_tick_idx = self._ticks_per_ex - 1
        else:
            self._curr_tick_idx = 0
            self.reset_example_textboxes_callback()
        self.update_textboxes()
        self.draw_tick()


    def step_to_either_end_of_event_callback(self, forward):
        """
        Jump to the beginning or end of the current event.

        If ``forward`` is False and the current tick is not at the beginning of
        the current event, this jumps to the beginning of the current event.
        Otherwise, it jumps to the beginning of the previous event.

        If ``forward`` is True and the current tick is not at the end of the
        current event, this jumps to the end of the current event. Otherwise,
        it jumps to the end of the next event.

        Note:
            The corresponding button is disabled when the current tick is the
            first or last tick in the current example.

        Args:
            forward (bool): True if stepping forward, False if stepping back.
        """
        if forward:
            if not self.end_of_event():
                self._curr_tick_idx = (self.ev_curr + 1) * self._ticks_per_ev - 1
            else:
                self._curr_tick_idx = min((self.ev_curr + 2) * self._ticks_per_ev - 1, self._ticks_per_ex -1)

            # already reserved the initial tick for continuous network, add it back
            if self._curr_tick_idx != self._ticks_per_ex -1 and type(self.input_net).__name__ == "ContinuousNetwork":
                self._curr_tick_idx += 1

        else:
            if not self.start_of_event():
                self._curr_tick_idx = self.ev_curr * self._ticks_per_ev
            else:
                self._curr_tick_idx = (self.ev_curr - 1) * self._ticks_per_ev
        self.update_textboxes()
        self.draw_tick()


    def step_tick_callback(self, forward):
        """
        The callback of the third and the fourth button on the first row.
        Steps one clock tick and will step across event boundaries.
        If forward is True then step forward, otherwise step back.

        Args:
            forward (bool): True if stepping forward, False if stepping back.
        """
        if forward:
            self._curr_tick_idx += 1
        else:
            self._curr_tick_idx -= 1
        self.update_textboxes()
        self.draw_tick()


    def step_over_example_callback(self):
        """
        Automatically step until the end of the example.
        The rate of stepping is controlled by the slider.
        """
        def step():
            if self.end_of_example():
                return

            self.step_tick_callback(forward=True)

            delay_ms = int((0.0081 * self.h_slider_loc + 0.19) * 1000)
            self.window.after(delay_ms, step)

        step()

    def step_over_event_callback(self):
        """
        Automatically step to the end of the current event, or to the end of the
        next event if already at an event boundary.
        """
        if self.end_of_event() and not self.end_of_example():
            self.step_tick_callback(forward=True)

        def step():
            if self.end_of_event():
                return

            self.step_tick_callback(forward=True)

            delay_ms = int((0.0081 * self.h_slider_loc + 0.19) * 1000)
            self.window.after(delay_ms, step)

        step()


    def reset_node_storage(self):
        """
        Clear the information about three types of nodes.
        """
        for group_name in self.nodes:
            self.nodes[group_name] = []
        self.output_nodes_border = []

        for group_name in self.nodes_color:
            self.nodes_color[group_name] = []
        self.output_color_border = {}

    def create_listbox(self, parent):
        """
        Create the examples listbox.

        Args:
        parent: The frame within which to create the listbox.
        """
        def lb_onselect(evt):
            selection = evt.widget.curselection()
            if not selection:
                return

            self._curr_ex_idx = int(selection[0])
            self.run_example_wrapper(testing_mode=not self.is_training)
            self.reset_example_textboxes_callback()

            # reset node storage references
            self.draw_tick()

        global listbox
        if self.listbox_exist:
            listbox.destroy()
        listbox = tk.Listbox(parent)
        self.listbox_exist=True
        # get the list of examples and show their name if they have one
        example_list = self.input_example_list
        cur_unnamed_idx = 0
        for ex in example_list:
            if ex.name is not None:
                name = ex.name
            else:
                name = "example " + str(cur_unnamed_idx)
                cur_unnamed_idx += 1
            listbox.insert(tk.END, name)

        hbar = tk.Scrollbar(listbox, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        hbar.config(command=c.xview)
        listbox.config(xscrollcommand=hbar.set)
        listbox.pack(side=tk.LEFT, fill="both", expand=True)

        listbox.bind('<<ListboxSelect>>', lb_onselect)
        self.listbox = listbox

        if example_list:
            self._curr_ex_idx = 0
            listbox.select_set(0)
            listbox.see(0)

            self.run_example_wrapper(testing_mode=not self.is_training)
            self.reset_example_textboxes_callback()
            self.draw_tick()
            # Prevent forward navigation until the user selects an example
            # for i in range(3, 8):
            #     self.btn_lst[i].configure(state="disabled")


    def create_scrollbars(self, v_scrollbar_parent, h_scrollbar_parent):
        """
       Creates the vertical slider to change color temperature and the
       horizontal slider to change the speed of automatically stepping
       through clock ticks.

       Args:
            v_scrollbar_parent: The frame within which to create the vertical slider.
            h_scrollbar_parent: The frame within which to create the horizontal slider.

       """

        def v_scaleevent(ev):
            self.v_slider_loc = int(ev)
            self.draw_tick()

        def h_scaleevent(ev):
            self.h_slider_loc = 100 - int(ev)

        vs = tk.Scale(v_scrollbar_parent, from_=100, to=1,
                      command=v_scaleevent)
        vs.set(self.v_slider_loc)
        vs.grid(sticky=tk.W)

        hs = tk.Scale(h_scrollbar_parent, from_=1, to=100, orient='horizontal',
                      command=h_scaleevent)
        hs.set(self.h_slider_loc)
        hs.grid(row=1, column=10, columnspan=4)

    def color_weights(self):
        """
        Colors nodes in link weight display mode.
        """
        tags = self.tags[self.right_click_id]
        node_type = tags[0]
        col_val = int(tags[1])
        self.output_col_val = col_val
        if self.right_click_id > -1:
            if not self.prev_right_click_id:
                self.prev_right_click_id = self.right_click_id
                if len(tags) < 5:
                    self.canvas.itemconfig(self.right_click_id, outline=YELLOW_HEX)
                else:
                    loc = tags[-4:]
                    z = c.create_rectangle(float(loc[0]) - 2, float(loc[1]) - 2, float(loc[2]) + 2,
                                            float(loc[3]) + 2, width=1, outline=YELLOW_HEX)
                    self.yellow_border_id = z
            elif self.right_click_id == self.prev_right_click_id:
                if self.right_click:
                    self.canvas.itemconfig(self.right_click_id, outline=YELLOW_HEX)
                # un-right-click this node
                else:
                    self.canvas.itemconfig(self.right_click_id, outline=BLACK_HEX)
            else:
                # un-right-click this node then yellow outline new node
                self.canvas.itemconfig(self.prev_right_click_id, outline=BLACK_HEX)
                self.canvas.itemconfig(self.right_click_id, outline=YELLOW_HEX)
                self.prev_right_click_id = self.right_click_id

        self.incoming_val.clear()
        self.outgoing_val.clear()

        if self.link_flag:
            connected_groups = [] # groups connected to the current unit's group
            if len(tags) != 0:
                self.right_click_node_type = node_type
                group = self.input_net.get_group_by_name(node_type)
                # For each incoming link in group update incoming_val dictionary, where keys
                # are group names
                for incoming_link in group.incoming_links:
                    if self.link_derivs_flag:
                        values = incoming_link.weight_derivs
                    elif self.link_delta_flag:
                        values = incoming_link.last_weight_delta
                    else:
                        values = incoming_link.weights

                    if values.ndim == 1: # handle 1 to 1 link
                        values_to_display = [None] * len(values)
                        values_to_display[col_val] = values[col_val]
                    else:
                        values_to_display = [row[col_val] for row in values]

                    self.incoming_val[incoming_link.outgoing_group.name] = values_to_display
                    connected_groups.append(incoming_link.outgoing_group.name)
                    
                # For each outgoing link in group update incoming_val dictionary, where keys
                # are group names
                for outgoing_link in group.outgoing_links:
                    if self.link_derivs_flag:
                        values = outgoing_link.weight_derivs
                    elif self.link_delta_flag:
                        values = outgoing_link.last_weight_delta
                    else:
                        values = outgoing_link.weights

                    if values.ndim == 1: # handle 1 to 1 link
                        values_to_display = [None] * len(values)
                        values_to_display[col_val] = values[col_val]
                    else:
                        values_to_display = [row[col_val] for row in values.T]
                    # Prioritize incoming values by not adding outgoing link values when the current
                    # group already has an incoming link from the same group
                    if outgoing_link.incoming_group.name not in self.incoming_val:
                        self.outgoing_val[outgoing_link.incoming_group.name] = values_to_display
                        connected_groups.append(outgoing_link.incoming_group.name)

                # change color of nodes to reflect weight value
                # group_name are the keys of the incoming and outgoing val dictionaries
                # iterate over link values of incoming links and color accordingly
                for group_name in self.incoming_val:
                    for i, value in enumerate(self.incoming_val[group_name]):
                        if value is None: # 1to1 link
                            fill_color = LIGHT_GREY_HEX
                        else:
                            fill_color = self.value_to_color(
                                value, "input", link_weight=True
                            )
                        self.canvas.itemconfig(
                            self.nodes[group_name][i],
                            fill=fill_color
                        )
                # iterate over link values of outgoing links and color accordingly
                for group_name in self.outgoing_val:
                    for i, value in enumerate(self.outgoing_val[group_name]):
                        if value is None:
                            fill_color = LIGHT_GREY_HEX
                        else:
                            fill_color = self.value_to_color(
                                value, "output", link_weight=True
                            )
                        self.canvas.itemconfig(
                            self.nodes[group_name][i],
                            fill=fill_color
                        )
                # color groups without connection to the current group with gray fill
                for group in self.input_net.groups:
                    if group.name not in connected_groups:
                        for node in self.nodes[group.name]:
                            self.canvas.itemconfig(node, fill="gray")

        else:
            if self.yellow_border_id:
                c.delete(self.yellow_border_id)
            # bring back color
            for group_name in self.nodes_color:
                for i, color in enumerate(self.nodes_color[group_name]):
                    self.canvas.itemconfig(self.nodes[group_name][i], fill=color, outline=BLACK_HEX)

    def create_widgets(self):
        """
        Create all the widgets within the windows.
        """
        # Create some room around all the internal frames
        self.window['padx'] = 5
        self.window['pady'] = 5

        self.window.grid_columnconfigure(1, weight=4)
        self.window.grid_rowconfigure(1, weight=1)

        event_time_frame = tk.Frame(
            self.window,
            relief=tk.RIDGE,
            borderwidth=1,
            padx=0,
            pady=6,
        )
        event_time_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.create_labels(event_time_frame)

        global OUTPUT_TARGET_FRAME, c, ex_set_frame

        # make the text a var
        OUTPUT_TARGET_FRAME = ttk.LabelFrame(self.window,
                                             text="Outputs and Targets",
                                             labelanchor="n",
                                             relief=tk.RIDGE)
        OUTPUT_TARGET_FRAME.grid(row=1, column=1,
                                 sticky=tk.N + tk.E + tk.S + tk.W)

        ex_set_frame = ttk.LabelFrame(self.window, text="Example Set",
                                      relief=tk.RIDGE)
        ex_set_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # ----- Basic creation of canvas inside the OUTPUT_TARGET_FRAME to
        # convert buttons to drawings: still in progress -----
        iframe5 = ttk.Frame(OUTPUT_TARGET_FRAME, relief=tk.RAISED)
        iframe5.pack(expand=True, fill="both")

        self.plot_width = 700
        self.plot_height = self.input_net.num_groups * 200 * (self.cell_size + 1)
        c = tk.Canvas(iframe5, bg='white', highlightthickness=0,
                      scrollregion=(0, 0, self.plot_width, self.plot_height))
        c.update()

        # code for creating the horizontal and vertical bars around canvas
        hbar = tk.Scrollbar(iframe5, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        hbar.config(command=c.xview)
        hbar.update()
        self.hbar_height = hbar.winfo_height()

        vbar = tk.Scrollbar(iframe5, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        vbar.config(command=c.yview)
        vbar.update()
        self.vbar_width = vbar.winfo_width()

        c.configure(width=iframe5.winfo_width(), height=iframe5.winfo_height())
        c.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        c.pack(fill="both", expand=True)
        self.canvas = c

        def motion(event):
            # if no node is right-clicked, just display the name and info of the node we are hovering over
            if not self.link_flag:
                if self.frozen_cell is None:
                    information = c.find_withtag(tk.CURRENT)
                    if information:
                        if len(information) > 0:
                            id = information[0]
                            tags = self.tags[id]

                            # if we are hovering over a node
                            if len(tags) > 1:
                                if tags[0] == 'target':
                                    self.curr_cell_info = "T:" + tags[3]
                                else:
                                    self.curr_cell_info = "O:" + tags[2]
                                self.curr_cell_name = tags[0] + ":" + tags[1]

                            # if we are hovering over nothing
                            else:
                                self.curr_cell_info, self.curr_cell_name = "", ""

                            c.update_idletasks()

            # display the link weight of the node we hover over relative to the one being right-clicked
            else:
                information = c.find_withtag(tk.CURRENT)
                if len(information) > 0:
                    id = information[0]  # id is is a random integer
                    tags = self.tags[id]

                    self.curr_cell_info = " "
                    if len(tags) > 1:
                        self.curr_cell_name = tags[0] if tags[0] != "target" else "output"
                        self.curr_cell_name += " : " + tags[1]  # add arrow later

                        # Only access first element in values to display if we right click
                        # bias group node
                        unit_idx = int(tags[1])

                        if tags[0] in self.incoming_val:
                            self.curr_cell_name += " ->"
                            self.curr_cell_info = self.incoming_val[tags[0]][unit_idx]
                        elif tags[0] in self.outgoing_val:
                            self.curr_cell_name = "-> " + self.curr_cell_name
                            self.curr_cell_info = self.outgoing_val[tags[0]][unit_idx]
                        else:
                            self.curr_cell_info, self.curr_cell_name = "", ""

                    # handle link value for numpy pytorch, and 1to1 link
                    if self.curr_cell_info is None:
                        self.curr_cell_info = ""
                    elif hasattr(self.curr_cell_info, "item"):
                        self.curr_cell_info = self.curr_cell_info.item()

                    if self.curr_cell_info != "":
                        self.curr_cell_info = f"{float(self.curr_cell_info):.8f}"

                    c.update_idletasks()
            self.update_current_cell_name_and_info()

        def restore_outline():
            if self.selected_outline_id is not None:
                self.canvas.itemconfig(
                    self.selected_outline_id,
                    outline=self.selected_outline_color
                )

        def on_click(event):  # when user left clicks
            self.left_click = not self.left_click
            c.focus_set()

            x = c.canvasx(event.x)
            y = c.canvasy(event.y)
            information = c.find_overlapping(x, y, x, y)
            id = information[-1] if information else None

            if id != None:
                tags = self.tags[id]
                if id == self.selected_node:
                    # unselect current node
                    restore_outline()
                    self.selected_node = None
                    self.frozen_cell = None
                    self.update_current_cell_name_and_info()
                    return
                elif self.selected_node is not None:
                    # restore previously selected node
                    restore_outline()

                # select newly clicked node
                self.selected_node = id
                # decide between the thin border for non target node, thick border for output node, or the thick boarder of target rim itself
                if tags[0] == "target":
                    outline_id = id
                elif tags[0] == "output" and len(tags) > 5:
                    outline_id = int(tags[5])
                else:
                    outline_id = id

                # save current selected node id and color
                self.selected_outline_id = outline_id
                self.selected_outline_color = self.canvas.itemcget(outline_id, "outline")

                self.canvas.itemconfig(outline_id, outline=PINK_HEX)
                if len(tags) != 0:
                    if tags[0] == 'target':
                        self.curr_cell_info = "O:" + tags[3]
                        self.curr_cell_name = "output" + ":" + tags[1]
                        self.frozen_cell = ("output", tags[1])
                    else:
                        self.curr_cell_info = "O:" + tags[2]
                        self.curr_cell_name = tags[0] + ":" + tags[1]
                        self.frozen_cell = (tags[0], tags[1])
                else:
                    self.frozen_cell = None
            self.update_current_cell_name_and_info()

        def right_click(event):
            x = c.canvasx(event.x)
            y = c.canvasy(event.y)
            information = c.find_overlapping(x, y, x, y)

            if information:
                id = information[-1]

                if self.tags[id][0] != 'target':
                    self.right_click_id = id

                    if self.right_click_id != self.prev_right_click_id:
                        self.right_click = True
                        # If right clicked from unit value mode, change value selection
                        # else only recolor
                        if not self.link_flag:
                            self.update_value_choice('Link Weights')
                        else:
                            self.color_weights()

                    else:
                        if self.link_flag:
                            # Return to output_and_targets mode if right clicking same node twice
                            # in unit value mode and reset right click node id
                            self.right_click = False
                            self.update_value_choice('Outputs and Targets')
                            self.prev_right_click_id = -1
                            self.canvas.itemconfig(self.right_click_id, outline=BLACK_HEX)
                        else:
                            self.update_value_choice('Link Weights')
    
        self.current_c_width = c.winfo_width()

        c.bind('<Motion>', motion)
        c.bind("<Button-1>", on_click)
        c.bind("<Button-2>", right_click)
        c.bind("<Button-3>", right_click)

        slider_frame = ttk.LabelFrame(self.window, relief=tk.RIDGE)
        slider_frame.grid(row=1, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.create_listbox(ex_set_frame)
        self.create_scrollbars(v_scrollbar_parent=slider_frame,
                               h_scrollbar_parent=event_time_frame)
    
    def draw_plot_layout(self):
        """Render ``net.plot_layout`` if present.

        For now we implement only what we need for an initial ``plotRow``:

        - layout is a list[list[UnitCell|BlankCell]]
        - cell geometry uses the viewer's current ``cell_size``/``cell_spacing``
        - we redraw from scratch on each tick (cheap enough; simpler + robust)
        """
        net = self.input_net
        layout = getattr(net, "plot_layout", None)
        if not layout:
            return

        # Always redraw from scratch to avoid having to track move/resize updates.
        self.drawing = True
        self.reset_node_storage()
        self.canvas.delete("all")

        cell_size = int(self.cell_size)
        spacing = int(self.cell_spacing)
        margin_x = 30
        margin_y = 30

        # Build group lookup once
        group_map = {getattr(g, "name"): g for g in getattr(net, "groups", [])}

        # Precompute per-group value arrays for current tick (mirrors draw_group logic)
        group_vals = {}
        group_targets = {}
        for g in getattr(net, "groups", []):
            vals = []
            targets = [None] * (len(getattr(g, "target", [])) or len(getattr(g, "target_history", [None])[-1]) if hasattr(g, "target_history") else 0)

            if self.output_and_target_flag:
                vals = g.output_history[self._curr_tick_idx] if g.group_type != 'bias' else g.output_history
                # Only outputs have targets in your current viewer; keep consistent
                if hasattr(g, "target_history"):
                    targets = g.target_history[self._curr_tick_idx] if g.group_type != 'bias' else g.target_history
            elif self.output_only_flag or self.link_flag:
                vals = g.output_history[self._curr_tick_idx] if g.group_type != 'bias' else g.output_history
                targets = [None] * len(vals)
            elif self.target_only_flag:
                # In target-only mode we display targets as fills (like draw_node does)
                if hasattr(g, "target_history"):
                    vals = g.target_history[self._curr_tick_idx] if g.group_type != 'bias' else g.target_history
                else:
                    vals = [None] * (len(getattr(g, "output_history", [])) or 0)
                targets = [None] * len(vals)
            elif self.input_only_flag:
                vals = g.input_history[self._curr_tick_idx] if g.group_type != 'bias' else g.input_history
                targets = [None] * len(vals)
            elif self.external_input_only_flag:
                vals = g.external_input
                targets = [None] * len(vals)
            elif self.output_derivs:
                vals = g.output_derivs
                targets = [None] * len(vals)
            elif self.input_derivs:
                vals = g.input_derivs
                targets = [None] * len(vals)
            else:
                vals = g.output_history[self._curr_tick_idx] if g.group_type != 'bias' else g.output_history
                targets = [None] * len(vals)

            group_vals[g.name] = vals
            # Some lists may be numpy arrays; keep indexing simple
            group_targets[g.name] = targets if targets is not None else [None] * len(vals)

        max_cols = max((len(row) for row in layout), default=0)
        total_w = margin_x * 2 + max_cols * (cell_size + spacing)
        total_h = margin_y * 2 + len(layout) * (cell_size + spacing)
        self.canvas.configure(scrollregion=(0, 0, max(self.plot_width, total_w), max(self.plot_height, total_h)))

        def _draw_one_cell(r, c, group, uidx):
            x0 = margin_x + c * (cell_size + spacing)
            y0 = margin_y + r * (cell_size + spacing)
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            gtype = group.group_type if getattr(group, "group_type", None) else group.name
            val_list = group_vals.get(group.name, [])
            tgt_list = group_targets.get(group.name, [])
            val = val_list[uidx] if 0 <= uidx < len(val_list) else None
            target_val = tgt_list[uidx] if 0 <= uidx < len(tgt_list) else None

            # Match your existing rules
            if self.link_flag or (self.external_input_only_flag and gtype != "input"):
                fill_color = LIGHT_GREY_HEX
            elif self.target_only_flag:
                fill_color = self.value_to_color(val, "target")
            elif self.input_derivs:
                fill_color = self.value_to_color(val, "input_derivs")
            elif self.output_derivs:
                fill_color = self.value_to_color(val, "output_derivs")
            else:
                fill_color = self.value_to_color(val, gtype)

            # Output: optional target border
            if gtype == "output":
                curr_unit_name = self.unit_names_by_group[group.name][uidx]
                output_node_color = self.value_to_color(val, "output") if not (self.link_flag or self.external_input_only_flag) else LIGHT_GREY_HEX
                border_color = self.value_to_color(target_val, "target") if (target_val is not None and self.output_and_target_flag) else None

                if border_color is None:
                    node_id = self.canvas.create_rectangle(x0, y0, x1, y1, fill=output_node_color, outline=BLACK_HEX, width=1)
                    self.tags[node_id] = list(map(str, (group.name, uidx, val, target_val, curr_unit_name)))
                else:
                    node_id = self.canvas.create_rectangle(x0 + 0.12, y0 + 0.12, x1 - 0.12, y1 - 0.12,
                                                          fill=output_node_color, outline=BLACK_HEX, width=1)
                    border_id = self.canvas.create_rectangle(x0, y0, x1, y1, width=5, outline=border_color,
                                                             tags='target' + str(uidx))
                    self.output_nodes_border.append(border_id)
                    self.tags[border_id] = list(map(str, ('target', uidx, val, target_val, curr_unit_name, x0, y0, x1, y1)))
                    self.tags[node_id] = list(map(str, (group.name, uidx, val, target_val, curr_unit_name, border_id, x0, y0, x1, y1)))

                self.nodes[group.name].append(node_id)
                self.nodes_color[group.name].append(output_node_color)
                return

            # Non-output nodes
            node_id = self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill_color, outline=BLACK_HEX, width=1)
            self.nodes[group.name].append(node_id)
            self.nodes_color[group.name].append(fill_color)
            self.tags[node_id] = list(map(str, (group.name, uidx, val, target_val)))

        for r, row in enumerate(layout):
            for c, cell in enumerate(row):
                if isinstance(cell, BlankCell) or cell is None:
                    continue
                if not isinstance(cell, UnitCell):
                    continue
                group = group_map.get(cell.group_name)
                if group is None:
                    continue
                _draw_one_cell(r, c, group, cell.unit_index)

        # Link mode: restore yellow selection and recolor weights.
        if not self.link_flag:
            if self.right_click_id > -1:
                self.canvas.itemconfig(self.right_click_id, outline=BLACK_HEX)
        elif self.link_flag and self.right_click_id > -1:
            self.color_weights()

        self.drawing = False

    def resize_canvas_height(self):
        self.canvas.update_idletasks()
        bbox = self.canvas.bbox("all")

        if bbox:
            top_padding = 20
            bottom_padding = 40
            content_height = bbox[3] - bbox[1]

            canvas_height = min(
                max(content_height + top_padding + bottom_padding, 150),
                600
            )

            self.canvas.configure(
                height=canvas_height,
                scrollregion=(
                    bbox[0],
                    bbox[1] - top_padding,
                    bbox[2],
                    bbox[3] + bottom_padding
                )
            )

    
    # ---------------------------------------------------------------------
    # plotRow wrappers
    # ---------------------------------------------------------------------

    def reset_plot_layout(self):
        """Reset plotRow state + layout on the network, then redraw."""
        reset_plot_state(self.input_net)
        self.draw_tick()

    def plotRow(self, num_rows=1, *cmd_tokens):
        """Apply plotRow tokens to the network layout, then redraw.

        This is a thin wrapper; the real implementation lives in
        ``src.gui.unit_layout``.
        """
        # Accept either a raw string or token list.
        if len(cmd_tokens) == 1 and isinstance(cmd_tokens[0], str):
            apply_plotrow_from_command(self.input_net, cmd_tokens[0])
        else:
            apply_plotrow(self.input_net, num_rows, [str(x) for x in cmd_tokens])
        self.draw_tick()

    def plotRow_from_command(self, cmd):
        """Convenience wrapper: accept a raw command string."""
        apply_plotrow_from_command(self.input_net, cmd)
        self.draw_tick()
