import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from functools import partial
import webcolors as wc
import math
import random
from ipykernel.eventloops import register_integration
import numpy as np
import threading
# from ..examples.example import Example
# from ..examples.example_set import ExampleSet
# from ..examples.event import Event
# from ..examples.unit_group import UnitGroup

# from ..backend.network import Network
from PyLens.backend.network import Network

LIGHT_GREY_HEX = "#d3d3d3"
BLUE_HEX = "#0000ff"
RED_HEX = "#ff0000"
YELLOW_HEX = "#ffff00"
BLACK_HEX = "#000000"
WHITE_HEX = "#ffffff"
PINK_HEX = "#ff69b4"

class link_viewer():
    """
    The Link Viewer provides a graphical illustration of the links between the layers of the network.
    """

    def __init__(self, parent, all_group_outputs=[], input_net=None, output_example_set=[], training_set_list=[],
                 testing_set_list=[], is_training=True, curr_cell_name="", curr_cell_info="", h_slider_loc=0,
                 v_slider_loc=60, curr_ex_idx=0, color_palette=0, frozen_cell=None, cell_size=0, cell_spacing=0):
        """
        Initialize the link viewer.

        Args:
            parent: parent window
            all_group_outputs (list): the outputs of all the examples.
                each element is an example in the format [input list, hidden list, output list, target list]
                will be abandoned once backend finished implementing returning ExampleSet for run_batch.
                input_net (Network): input network to run.
            output_example_set (list): the result of run_batch.
            ev_curr (str): current event
            ev_total (str): total event
            ex_time_curr (str): current example time
            ex_time_total (str): total example time
            ev_time_curr (str): current event time
            ev_time_total (str): total event time
            training_set_list (list): list of Example objects from training set
            testing_set_list (list): list of Example objects from testing set
            is_training (bool): if the current selection is "training". Default to True.
            curr_cell_name (str): the cell name if cursor is hovering over at some cell.
                bias/input/hidden/output:0/1. None if cursor is not hovering over any block.
            curr_cell_info (str): the cell value if cursor is hovering over some cell.
                O: number / T: number. None if cursor is not hovering over any block.
            h_slider_loc (int): an int between 0 and 100 represents the location of current
                horizontal slider. 100 represents the rightmost position.
            v_slider_loc (int): an int between 0 and 100 represents the location of current
                vertical slider. 100 represents the topmost position.
            curr_ex_idx (int): an int representing the idex of currently selected example.
            color_palette (int): an int between 0 and 3. Default 0. Only deals with 0,1,2 for now.
                0: 'Blue-Black-Red'
                1:'Blue-Black-Yellow'
                2:'Black-Gray-White'
                3:'Hinton Diagram'
            frozen_cell (tuple): a tuple representing the frozen cell.
                frozen_cell = None at initialization and after blank area of canvas is clicked.
                a tuple of the form (type, idx) where type = 'input','hidden' or 'output'
                idx = 0,1,2...
        """

        self.all_group_outputs = all_group_outputs
        self.input_net = input_net
        self.output_example_set = output_example_set

        # ev_total should be populated every time after self.curr_ex_idx changes
        self.ev_curr = None
        self.ev_total = None
        self.ex_time_curr = None
        self.ex_time_total = None
        self.ev_time_curr = None
        self.ev_time_total = None
        self.training_set_list = training_set_list
        self.testing_set_list = testing_set_list
        self.is_training = is_training
        self.variance_cell = ""
        self.mean_abs_cell = ""
        self.mean_dist_cell = ""
        self.max_cell = ""
        self.mean_cell = ""
        self.link_value_cell = ""
        self.input_name_cell = ""
        self.output_name_cell = ""
        self.curr_cell_name = curr_cell_name
        self.curr_cell_info = curr_cell_info
        self.h_slider_loc = h_slider_loc
        self.v_slider_loc = v_slider_loc
        self.curr_ex_idx = curr_ex_idx
        self.node_items = []

        self.frozen_cell = frozen_cell
        self.window = tk.Toplevel(parent)
        self.window.title("link viewer")
        self.window.minsize(650, 0)

        # Stores tk.Int variables when sending/receiving groups commands are generated
        self.active_receiving_groups_checkVars = []
        self.active_sending_groups_checkVars = []
        self.value_radiobutton_var = tk.IntVar(self.window, value=0) # "Link Weights" checked by default in value menu

        self.weights_checkVar = True
        self.derivs_checkVar = False
        self.deltas_checkVar = False

        self.global_link_vals = []

        self.output_count = 0
        self.input_count = 0
        self.bias_count = 0
        self.hidden_count = 0

        self.group_oX = [0,0,0,0]
        self.group_eX = [0,0,0,0]
        self.group_oY = [0,0,0,0]
        self.group_eY = [0,0,0,0]

        self.cell_size = cell_size
        self.cell_spacing = cell_spacing

        self.update_display_option = 2

        # for different color palette

        self.color_dict = {0: (BLUE_HEX, RED_HEX),    # BLUE_BLACK_RED
                           1: (BLUE_HEX, YELLOW_HEX), # BLUE_RED_YELLOW
                           2: (BLACK_HEX, WHITE_HEX), # BLACK_GRAY_WHITE
                           3: (BLACK_HEX, WHITE_HEX)} # HINTON
        self.num_colors = 101

        # default to Blue-Black-Red
        self.color_palette = 0
        self.filled_map_idx = []
        self.color_low = self.color_dict[self.color_palette][0]
        self.color_high = self.color_dict[self.color_palette][1]

        # initialize all the available colors
        self.color_map = {0:[None]*(self.num_colors), 1:[None]*(self.num_colors),
                           2: [None]*(self.num_colors), 3: [None]*(self.num_colors)}


        # fill Blue-Black-Red color map on initializaiton
        self.fill_color_map()

        self.create_menu()
        self.create_widgets()
        c.bind("<Configure>", lambda event: self.center_canvas_items())
        self.window.after_idle(self.center_canvas_items)



    def fill_color_map(self):
        """
        Fill the color map with the color palette.
        """
        # if it is already filled then don't need to fill again
        # fill only the ones need for now
        if not (self.color_palette in self.filled_map_idx):
                if self.color_palette == 0:  # BLUE_BLACK_RED
                    blue = 0
                    for i in range(self.num_colors):

                        red = i / ((self.num_colors-1)/2) - 1
                        if red < 0:
                            blue = -red
                            red = 0
                            self.color_map[0][i] = (0, 0, int(blue*255))
                        else:
                            self.color_map[0][i] = (int(red*255), 0, 0)
                            blue = 0

                elif self.color_palette == 1:  # BLUE_RED_YELLOW
                    for i in range(self.num_colors):
                        rval = i / (self.num_colors-1)
                        if rval <= 0.5:
                            green = 0
                            red = rval * 2
                            blue = 1 - red
                            self.color_map[1][i] = (int(red*255), 0, int(blue*255))
                        else:
                            red = 1
                            blue = 0
                            green = rval * 2 - 1
                            self.color_map[1][i] = (255, int(green*255), 0)

                else:  # BLACK_GRAY_WHITE, no HINTON for now
                    for i in range(self.num_colors):
                        grey = i / ((self.num_colors-1)/2) - 1
                        y = int(127.5*(grey+1))
                        self.color_map[2][i] = (y, y, y)

        self.filled_map_idx.append(self.color_palette)


    def run_batch_wrapper(self):
        """
        Run the network and get the output example set.
        """
        # once the backend complete exampleSet implementation,
        # we can get self.output_example_set from above run_batch() call
        # for now just assume self.input_net.example_sets[0].example is the desired example list
        # i.e. just loading the example set
        self.output_example_set = self.input_net.example_sets[0].example

    def update_color_palette(self, x):
        """
        Change the current color scheme to x and draw the canvas.

        Args:
            x (int): the int representing the new color scheme.
        """

        self.color_palette = x
        self.color_low = self.color_dict[x][0]
        self.color_high = self.color_dict[x][1]
        self.fill_color_map()
        self.draw_nodes(iframe5, c)

    def inverse_log(self, x):
        """
        Inverse log function.

        Args:
            x (float): the input value.
        """
        if x <= 0:
            return -float('inf')
        elif x >= 1:
            return float('inf')
        else:
            return math.log(x/(1-x))

    def value_to_color(self, cell_val, type=None):
        """
        Calculate the color given cell_val, cell type, and color temperature
        (indicated by slider).

        Args:
            cell_val (str): raw value of unit.
            type (str): one of:
                ["link_weight", "link_derivs", "link_deltas"].
        """
        temperature = self.v_slider_loc / 4.5
        max_temperature = 22.22
        min_temperature = 0
        casted_cell_val = float('inf')
        if cell_val is None:
            return LIGHT_GREY_HEX

        # Link viewer
        else:
            if type is None:
                # the default behaves like the output and target
                # bounded by 1 and -1
                casted_cell_val = (cell_val + 1) / 2
                casted_cell_val = self.inverse_log(casted_cell_val)

            if type in ["link_weight", "link_derivs"]:
                casted_cell_val = cell_val

            elif type == "link_deltas":
                pass

            else:  # more to it later...
                pass

        if temperature == max_temperature:
            if casted_cell_val < 0:
                return self.color_low
            else:
                return self.color_high

        elif temperature == min_temperature:
            return self.color_map[self.color_palette][self.num_colors // 2]

        else:
            y = int(100 / (1 + math.exp(-casted_cell_val * temperature)))
            return wc.rgb_to_hex(self.color_map[self.color_palette][y])


    def create_menu(self):
        """
        Create the menu bar.
        """
        menubar = tk.Menu(self.window)
        viewer_menu = tk.Menu(menubar, tearoff=0)
        sending_groups_menu = tk.Menu(menubar, tearoff=1)
        receiving_groups_menu = tk.Menu(menubar, tearoff=2)
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
            print("Selected value:", self.update_display_option)
        
        for j in range(len(update_option_list)):
            update_after_menu.add_radiobutton(label=update_option_list[j], state='normal', value=j,
                                              variable=update_choice,
                                              command=update_after_choice)

        self.cell_size_var = tk.IntVar(value=self.cell_size)
        cell_size_menu = tk.Menu(viewer_menu)
        for name in range(20):
            cell_size_menu.add_radiobutton(
                label=str(name),
                value=name,
                variable=self.cell_size_var,
                command=partial(self.update_cell_size, str(name))
            )


        cell_spacing_menu = tk.Menu(viewer_menu)
        for name in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
            cell_spacing_menu.add_radiobutton(
                label=name,
                command=partial(self.update_cell_spacing, name))

        viewer_menu.add_cascade(label="Update After", menu=update_after_menu)
        viewer_menu.add_cascade(label="Cell Size", menu=cell_size_menu)
        viewer_menu.add_cascade(label="Cell Spacing", menu=cell_spacing_menu, state='disabled')
        viewer_menu.add_command(label="Update", command=self.update_current_cell_name_and_info)
        viewer_menu.add_command(label="Refresh", command=self.update_plots)
        # viewer_menu.add_command(label="Reset", command=None)
        # viewer_menu.add_command(label="Print...", command=None)
        # viewer_menu.add_command(label="Close", command=self.window.destroy)
        menubar.add_cascade(label="Sending Groups", menu=sending_groups_menu)


        menubar.add_cascade(label="Receiving Groups", menu=receiving_groups_menu)

        choice = tk.BooleanVar()
        choice.set(self.is_training)

        self.active_sending_groups = {}
        self.active_receiving_groups = {}
        for group in reversed(self.input_net.groups):
            if group.outgoing_links:
                self.active_sending_groups_checkVars.append(tk.IntVar(self.window, value=1))
                sending_groups_menu.add_checkbutton(label=f"{group.name}", variable=self.active_sending_groups_checkVars[-1],
                                                    command=partial(self.update_sending_groups, group.name))
                self.active_sending_groups.update({group.name: True})
            else:
                self.active_sending_groups.update({group.name: False})
            if group.incoming_links:
                self.active_receiving_groups_checkVars.append(tk.IntVar(self.window, value=1))
                receiving_groups_menu.add_checkbutton(label=f"{group.name}", variable=self.active_receiving_groups_checkVars[-1],
                                                      command=partial(self.update_receiving_groups, group.name))
                self.active_receiving_groups.update({group.name: True})
            else:
                self.active_receiving_groups.update({group.name: False})
                
        menubar.add_cascade(label="Value", menu=value_menu)
        value_menu.add_radiobutton(label="Link Weights", value=0, var=self.value_radiobutton_var,
                                   command=partial(self.update_values, weights=True))
        value_menu.add_radiobutton(label="Link Derivs", value=1, var=self.value_radiobutton_var,
                                   command=partial(self.update_values, derivs=True))
        value_menu.add_radiobutton(label="Link Deltas", value=2, var=self.value_radiobutton_var,
                                   command=partial(self.update_values, deltas=True))

        menubar.add_cascade(label="Palette", menu=palette_menu)

        color_scheme_lst = ['Blue-Black-Red', 'Blue-Black-Yellow', 'Black-Gray-White', 'Hinton Diagram']
        for i in range(len(color_scheme_lst)):
            palette_menu.add_command(label=color_scheme_lst[i], command=partial(self.update_color_palette, i))
        self.window.configure(menu=menubar)


    def update_values(self, weights=False, derivs=False, deltas=False):
        """
        Update the values displayed on the canvas.

        Args:
            weights (bool): if True, display weights.
            derivs (bool): if True, display derivs.
            deltas (bool): if True, display deltas.
        """
        if weights == 1:
            self.weights_checkVar = True
            self.derivs_checkVar = False
            self.deltas_checkVar = False
            link_frame.config(text="Link Weights")

        elif derivs == 1:
            self.weights_checkVar = False
            self.derivs_checkVar = True
            self.deltas_checkVar = False
            link_frame.config(text="Link Derivs")


        elif deltas == 1:
            self.weights_checkVar = False
            self.derivs_checkVar = False
            self.deltas_checkVar = True
            link_frame.config(text="Link Deltas")

        c.delete("all")
        self.draw_nodes(iframe5, c)

    def update_plots(self):
        """
        Update the plots.
        """
        conditions = [self.value_radiobutton_var.get() == 0, self.value_radiobutton_var.get() == 1, self.value_radiobutton_var.get() == 2]
        self.update_values(weights=conditions[0], derivs=conditions[1], deltas=conditions[2])

    def update_sending_groups(self, group_name):
        """
        Update the sending groups.
        """
        self.active_sending_groups[group_name] = not self.active_sending_groups[group_name]
        self.update_groups()

    def update_receiving_groups(self, group_name):
        """
        Update the receiving groups.
        """
        self.active_receiving_groups[group_name] = not self.active_receiving_groups[group_name]
        self.update_groups()

    def update_groups(self):
        """
        Update the groups.
        """
        c.delete("all")
        self.draw_nodes(iframe5, c)

    def update_cell_spacing(self, cell_info):
        """
        Update the cell spacing.
        """
        self.cell_spacing = int(cell_info)
        c.delete("all")

        self.draw_nodes(iframe5, c)


    def update_cell_size(self, cell_info):
        """
        Update the cell size.
        """
        self.cell_size = int(cell_info)
        c.delete("all")

        self.draw_nodes(iframe5, c)


    @register_integration('tk')
    def loop_tk(kernel):
        """
        Start a kernel with the Tk event loop.
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


    def create_labels(self, parent):
        """
        Create the labels for the statistics.

        Args:
            parent: the parent frame.
        """
        # fixed component columns
        for col in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]:
            parent.columnconfigure(col, weight=0)

        # three expandable spacers
        for col in [0, 11, 15]:
            parent.columnconfigure(col, weight=1, uniform="spacer")

        label_specs = {
            "text": ["Mean", "Variance", "Mean Abs.", "Mean Dist.", "Maximum"],
            "column": [1, 3, 5, 7, 9],
            "padx": [2] * 5
        }

        colspan = 1
        for j in range(5):

            lb = ttk.Label(parent, text=label_specs["text"][j])
            lb.grid(row=0, column=label_specs["column"][j], columnspan=colspan, padx=label_specs["padx"][j])



        global mean_text, var_text, m_abs_text, dist_text, max_text, input_name, output_name, link_value
        mean_text, var_text, m_abs_text, dist_text, max_text, input_name, output_name, link_value = \
            tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar()
        mean_text.set("")
        self.mean_textbox = tk.Entry(parent, width=6, textvariable=mean_text, justify='center')
        self.mean_textbox.grid(row=1, column=1, padx=2, sticky=tk.W)
        self.mean_textbox.config(
                state='readonly',
                readonlybackground='white'
                )



        var_text.set("")
        self.var_textbox = tk.Entry(parent, width=8, textvariable=var_text, justify='center')
        self.var_textbox.grid(row=1, column=3, padx=2, sticky=tk.W)
        self.var_textbox.config(
                state='readonly',
                readonlybackground='white'
                )

        m_abs_text.set("")
        self.m_abs_textbox = tk.Entry(parent, width=8, textvariable=m_abs_text, justify='center')
        self.m_abs_textbox.grid(row=1, column=5, padx=2, sticky=tk.W)
        self.m_abs_textbox.config(
                state='readonly',
                readonlybackground='white'
                )

        dist_text.set("")
        self.dist_textbox = tk.Entry(parent, width=8, textvariable=dist_text, justify='center')
        self.dist_textbox.grid(row=1, column=7, padx=2, sticky=tk.W)
        self.dist_textbox.config(
                state='readonly',
                readonlybackground='white'
                )

        max_text.set("")
        self.max_textbox = tk.Entry(parent, width=8, textvariable=max_text, justify='center')
        self.max_textbox.grid(row=1, column=9, padx=2, sticky=tk.W)
        self.max_textbox.config(
                state='readonly',
                readonlybackground='white'
         )

        input_name.set("")
        output_name.set("")
        link_value.set("")
        self.input_name_textbox = tk.Entry(parent, width=12, textvariable=input_name, justify='center')
        self.input_name_textbox.grid(row=0, column=12, padx=2, sticky=tk.W)
        self.input_name_textbox.config(
                state="readonly",
                readonlybackground='white'
                )

        self.output_name_textbox = tk.Entry(parent, width=12, textvariable=output_name, justify='center')
        self.output_name_textbox.grid(row=0, column=14, padx=2, sticky=tk.W)
        self.output_name_textbox.config(
                state="readonly",
                readonlybackground='white'
                )

        self.link_value_textbox = tk.Entry(parent, width=12, textvariable=link_value, justify='center')
        self.link_value_textbox.grid(row=1, column=13, padx=2, sticky=tk.W)
        self.link_value_textbox.config(
                state="readonly",
                readonlybackground='white'
                )

        lb = ttk.Label(parent, text="->")
        lb.grid(row=0, column=13, padx=2)

    def create_scrollbars(self, v_scrollbar_parent,
                          example_canvas):
        """
       Creates the vertical slider to change color temperature and the
       horizontal slider to change the speed of automatically stepping
       through clock ticks.

       Args:
              v_scrollbar_parent: frame within which to create the vertical slider
                example_canvas: canvas to draw
       """

        def v_scaleevent(ev):
            self.v_slider_loc = int(ev)
            c.delete("all")

            self.draw_nodes(iframe5, c)

        vs = tk.Scale(v_scrollbar_parent, from_=100, to=1,
                      command=v_scaleevent)
        vs.set(self.v_slider_loc)
        vs.grid(sticky=tk.W)

    def reset_plot(self, canvas):
        """
        Reset the plot.
        """
        c.delete("all")
        canvas.delete("all")

    def create_node(self, type, col, val, target_val, canvas, outgoing, incoming):
        """
        Draw a cell on the canvas.

        Args:
            type (str): 'input','hidden' or 'output'. If 'output'. must specify target_val.
            col (int): col number as appeared in the frame. Which is the index of the cell within its layer.
            val (float): value of the cell.
            target_val (float): given only when type is output.
            canvas: canvas to draw.
            outgoing: the outgoing group name.
            incoming: the incoming group name.
        """

        if type == 'input':
            oY = 150
            eY = 180
        elif type == 'hidden':
            oY = 90
            eY = 120
        else:
            oY = 30
            eY = 60

        oX = 30 + col * 60
        eX = 60 + col * 60

        fill_color = self.value_to_color(val)

        if type == 'output':
            border_color = self.value_to_color(target_val)
            canvas.create_rectangle(oX+3, oY+3, eX-3, eY-3, fill=fill_color, outline=border_color,
                                    tag=(type, col, val, outgoing, incoming))
            canvas.create_rectangle(oX, oY, eX, eY, fill="", width=5, outline=border_color,
                                    tag=('target', col, target_val, val, outgoing, incoming))

        elif type == 'input':
            border_color = self.value_to_color(target_val)
            canvas.create_rectangle(oX+3, oY+3, eX-3, eY-3, fill=fill_color, stipple="gray25",
                                    tag=(type, col, val, outgoing, incoming))
            # canvas.create_line(oX, oY+30, eX, eY, width=7, tag=(type, col, val))
            canvas.create_rectangle(oX, oY, eX, eY, fill="", width=5,
                                    outline=border_color,
                                    tag=('input-target', col, target_val, val, outgoing, incoming))

        else:
            canvas.create_rectangle(oX, oY, eX, eY, fill=fill_color, tag=(type, col, val, outgoing, incoming))

        if self.frozen_cell == (type, str(col)):
            canvas.create_rectangle(oX, oY, eX, eY, fill="", stipple="gray50", width=1, outline="#FF69B4")
            self.curr_cell_name = type + ":" + str(col)
            self.curr_cell_info = "O" + ":" + str(val)
            
        self.update_current_cell_name_and_info()

    def update_current_cell_name_and_info(self):
        """
        Update the value of the last two boxes.
        """
        def update_textbox(textbox, value):
            textbox.config(state='normal')
            textbox.delete(0, "end")
            textbox.insert(0, value)
            textbox.config(
                    state='readonly',
                    readonlybackground='white'
                    )

        update_textbox(self.max_textbox, self.max_cell)
        update_textbox(self.var_textbox, self.variance_cell)
        update_textbox(self.m_abs_textbox, self.mean_abs_cell)
        update_textbox(self.dist_textbox, self.mean_dist_cell)
        update_textbox(self.mean_textbox, self.mean_cell)
        update_textbox(self.input_name_textbox, self.input_name_cell)
        update_textbox(self.output_name_textbox, self.output_name_cell)
        update_textbox(
            self.link_value_textbox,
            f"{float(self.link_value_cell):.8f}" if self.link_value_cell else ""
        )

    def add_titles(self, frame, canvas):
        """
        Draw the input, hidden, output and target layer of the current example.
        group_outputs is of the form: [[inp0,inp1],[hidden0,hidden1],[output0],[target0]]

        Args:
            frame: the frame where the canvas resides.
            canvas: the canvas to draw.
        """
        outgoing_groups = list(self.Y_anchor_dic.keys())
        incoming_groups = list(self.X_anchor_dic.keys())
        title_font_size = max(9, int(9 + self.cell_size * 0.5))
        index_font_size = max(10, int(10 + self.cell_size * 0.5))

        base_oX = 50
        base_oX_0 = 110
        for group_name in outgoing_groups[:-1]:
            coordinates = []

            for item in self.node_items:
                tags = canvas.gettags(item)
                if len(tags) > 8 and tags[8] == group_name:
                    coordinates.append(canvas.coords(item))

            if not coordinates:
                continue

            y_min = min(coords[1] for coords in coordinates)
            y_max = max(coords[3] for coords in coordinates)

            base_oY = (y_min + y_max) / 2
            first_cell = min(coordinates, key=lambda coords: coords[1])
            base_oY_0 = (first_cell[1] + first_cell[3]) / 2
            canvas.create_text(
                base_oX,
                base_oY,
                fill="black",
                font=("Helvetica 16", title_font_size),
                text=f"from {group_name}"
            )
            canvas.create_text(
                base_oX_0,
                base_oY_0,
                fill="black",
                font=("Helvetica 16", index_font_size),
                text="0"
            )

        base_oY = 40
        base_oY_0 = 75
        for group_name in incoming_groups[:-1]:
            coordinates = []

            for item in self.node_items:
                tags = canvas.gettags(item)

                if len(tags) > 9 and tags[9] == group_name:
                    coordinates.append(canvas.coords(item))

            x_min = min(coords[0] for coords in coordinates)
            x_max = max(coords[2] for coords in coordinates)
            group_width = x_max - x_min

            base_oX = (x_min + x_max) / 2
            first_cell = min(coordinates, key=lambda coords: coords[0])
            base_oX_0 = (first_cell[0] + first_cell[2]) / 2
            canvas.create_text(
                    base_oX,
                    base_oY,
                    fill="black",
                    font=("Helvetica 16",title_font_size), 
                    text=f"to\n{group_name}",
                    width=max(group_width + 20, 18),
                    justify="center",
                    anchor="s",
                    )
            canvas.create_text(
                    base_oX_0, 
                    base_oY_0, 
                    fill="black", 
                    font=("Helvetica 16",index_font_size), 
                    text="0")

    def update_origin_for_plot(self, param, oX, oY, eX, eY):
        """
        Update the origin for the plot.
        """
        self.group_oX[param] = oX
        self.group_oY[param] = oY
        self.group_eX[param] = eX
        self.group_eY[param] = eY

    def draw_nodes(self, frame, canvas):
        """
        Draw the nodes in the canvas.
        Iterates through each group and respective weights in each layer and creates the
        rectangle by passing in appropriately parameters.

        Args:
            frame: the frame where the canvas resides.
            canvas: the canvas to draw.
        """
        # Radio button that regulates which values to display
        # weight or the deriv according to user selection in menu
        self.node_items = []
        weight_check = self.weights_checkVar
        deriv_check = self.derivs_checkVar
        deltas_check = self.deltas_checkVar

        oX = 125 - ((self.cell_size / 5) * 8)/2
        oY = 105 - ((self.cell_size / 5) * 8)/2
        eX = 135 + ((self.cell_size / 5) * 8)/2
        eY = 115 + ((self.cell_size / 5) * 8)/2
        oX_anchor = oX
        eX_anchor = eX
        self.X_anchor_dic = {}
        self.Y_anchor_dic = {}

        self.global_link_vals = []

        # Starts drawing from upper left corner
        for group in reversed(self.input_net.groups):
            if not self.active_sending_groups[group.name]:
                continue

            if group.outgoing_links:
                # Add 20 to leave Y space between groups in canvas
                oY += 30 + ((self.cell_size / 5) * 8)
                eY += 30 + ((self.cell_size / 5) * 8)
                # Save Y coordinates to know where to add titles
                oY_anchor = oY
                eY_anchor = eY
                self.Y_anchor_dic.update({group.name: (oY, eY)})

            for link in reversed(group.outgoing_links):
                if not self.active_receiving_groups[link.incoming_group.name]:
                    continue
                if link.incoming_group.name not in self.X_anchor_dic.keys():
                    oX += 20 + ((self.cell_size / 5) * 8)
                    eX += 20 + ((self.cell_size / 5) * 8)
                    # Save X coordinates for titles and in case groups
                    # have incoming links from multiple sources
                    oX_anchor = oX
                    eX_anchor = eX
                    self.X_anchor_dic.update({link.incoming_group.name: (oX, eX)})
                else:
                    oX = self.X_anchor_dic[link.incoming_group.name][0]
                    eX = self.X_anchor_dic[link.incoming_group.name][1]
                    oX_anchor = oX
                    eX_anchor = eX

                oY = oY_anchor
                eY = eY_anchor

                # handle 1to1 link
                one_to_one = link.weights.ndim == 1
                if one_to_one:
                    rows = cols = len(link.weights)
                else:
                    rows, cols = link.weights.shape

                for i in range(rows):
                    for j in range(cols):
                        oX += 10 + ((self.cell_size / 5) * 8)
                        eX += 10 + ((self.cell_size / 5) * 8)

                        # A 1-D one-to-one link is displayed only on the diagonal
                        if one_to_one and i != j:
                            continue

                        col = i

                        if weight_check:
                            weight_val = link.weights[i] if one_to_one else link.weights[i][j]
                        elif deriv_check:
                            weight_val = link.weight_derivs[i] if one_to_one else link.weight_derivs[i][j]
                        else:
                            weight_val = link.last_weight_delta[i] if one_to_one else link.last_weight_delta[i][j]







                        # handle both numpy and pytorch
                        if hasattr(weight_val, "item"):
                            weight_val = weight_val.item()

                        self.global_link_vals.append(weight_val)
                        fill_color=self.value_to_color(weight_val)

                        # TODO update link stats 
                        mean_val = 0 \
                            #np.mean(outgoing_hidden.weights[i])
                        variance_val = 0 \
                            #np.var(outgoing_hidden.weights)
                        mean_abs_val = 0 \
                            #np.absolute(outgoing_hidden.weights)[i][0]
                        mean_dist = 0 \
                            #np.linalg.norm(outgoing_hidden.weights)
                        max_val = 0 \
                            #np.max(outgoing_hidden.weights[i])

                        node_item = canvas.create_rectangle(oX, oY, eX, eY, fill=fill_color,
                                                tag=(f"{group.name}-{link.incoming_group.name}", col , weight_val,
                                                     mean_val, variance_val, mean_abs_val,
                                                     mean_dist, max_val, f"{group.name}", f"{link.incoming_group.name}", j))
                        self.node_items.append(node_item)
                    if i != rows - 1:
                        oX = oX_anchor
                        eX = eX_anchor
                        if rows > 1:
                            oY += 10 + ((self.cell_size/5)*8)
                            eY += 10 + ((self.cell_size/5)*8)    

        # Add final coordinates to dictionaries to add text  
        self.X_anchor_dic.update({"end_coor_X":(oX, eX)})
        self.Y_anchor_dic.update({"end_coor_Y":(oY, eY)})

        self.global_max = round(float(np.max(self.global_link_vals)), 3) if self.global_link_vals else self.global_max
        self.global_mean = round(float(np.mean(self.global_link_vals)), 3) if self.global_link_vals else self.global_mean
        self.global_variance = round(float(np.var(self.global_link_vals)), 3) if self.global_link_vals else self.global_variance
        self.global_mean_abs = round(float(np.absolute(self.global_link_vals)[0]), 3) if self.global_link_vals else self.global_mean_abs
        self.global_mean_dist = round(float(np.linalg.norm(self.global_link_vals)), 3) if self.global_link_vals else self.global_mean_dist
        self.mean_cell = self.global_mean
        # tags[3] \
        # + ":"+ tags[0]
        self.variance_cell = self.global_variance
        # tags[4] \
        # + ":"+ tags[0]
        self.mean_abs_cell = self.global_mean_abs
        # tags[5]
        # "A:" +tags[1] + ":"+ tags[0]
        self.mean_dist_cell = self.global_mean_dist
        # tags[6]
        # "D:" +tags[1] + ":"+ tags[0]
        self.max_cell = self.global_max
        
        self.add_titles(frame, canvas)
        self.resize_canvas_height()
        self.window.after_idle(self.center_canvas_items)
        self.update_current_cell_name_and_info()

    def center_canvas_items(self):
        bbox = c.bbox("all")
        if bbox:
            c.move(
                "all",
                c.winfo_width() / 2 - (bbox[0] + bbox[2]) / 2,
                0
            )

    def create_listbox(self, parent, frame, canvas):
        """
        Create the examples listbox.

        Args:
            parent: the frame where listbox resides.
            frame: the frame where the visualization of example takes place.
            canvas: the canvas to draw.
        """

        self.draw_nodes(frame, canvas)

    def create_scrollbar(self, scrollbar_parent, example_parent, canvas):
        """
        Currently only creates the vertical slider to change color temperature.

        Args:
            scrollbar_parent: frame within which to create the slider
            example_parent: frame where the example list resides
            canvas: canvas to draw
        """
        def scaleevent(v):
            self.v_slider_loc = int(v)
            self.add_titles(example_parent,canvas)

        w1 = tk.Scale(scrollbar_parent, from_=100, to=1, command=scaleevent)
        w1.set(self.v_slider_loc)
        w1.grid(sticky=tk.W)


    def create_widgets(self):
        """
        Create and configure the widgets for the GUI.
        This method sets up the main window with padding and grid configuration,
        creates various frames and canvases, and binds events to widgets.
        """

        # Create some room around all the internal frames
        self.window['padx'] = 5
        self.window['pady'] = 5

        self.window.grid_columnconfigure(1, weight=4)
        self.window.grid_rowconfigure(1, weight=1)

        eventTimeFrame = tk.Frame(
            self.window,
            relief=tk.RIDGE,
            borderwidth=1,
            padx=0,
            pady=6,
        )
        eventTimeFrame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.create_labels(eventTimeFrame)

        global link_frame, c, iframe5
        link_frame = ttk.LabelFrame(self.window, text="Link Weights ", labelanchor="n", relief=tk.RIDGE)
        link_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        # ex_set_frame = ttk.LabelFrame(self.window, text="Example Set", relief=tk.RIDGE)
        # ex_set_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # ----- Basic creation of canvas inside the link_frame to convert buttons to drawings: still in progress -----

        iframe5 = ttk.Frame(link_frame, relief=tk.RAISED)
        iframe5.pack(expand=True, fill="both")

        c = tk.Canvas(iframe5, bg='white', highlightthickness=0,scrollregion=(0, 0, 5000, 5000))
        c.update()

        hbar = tk.Scrollbar(iframe5, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        hbar.config(command=c.xview)

        vbar = tk.Scrollbar(iframe5, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        vbar.config(command=c.yview)

        c.configure(width=iframe5.winfo_width(), height=iframe5.winfo_height())
        c.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        c.pack(fill="both", expand=True)
        self.canvas = c


        def motion(event):
            """
            Update the current cell name and info.
            """
            if self.frozen_cell is None:
                temp = c.find_withtag(tk.CURRENT)
                tags = c.gettags(temp)

                if len(tags) > 1:
                    if tags[0] != None:
                        self.mean_cell = self.global_mean
                            # tags[3] \
                                         # + ":"+ tags[0]
                        self.variance_cell = self.global_variance
                            # tags[4] \
                                             # + ":"+ tags[0]
                        self.mean_abs_cell = self.global_mean_abs
                            #tags[5]
                            #"A:" +tags[1] + ":"+ tags[0]
                        self.mean_dist_cell = self.global_mean_dist
                            #tags[6]
                            # "D:" +tags[1] + ":"+ tags[0]
                        self.max_cell = self.global_max
                            #tags[7]
                            # "MAX:" + tags[1] + ":"+tags[0]
                        self.input_name_cell = tags[8] + ":" + tags[1]
                        self.output_name_cell = tags[9] + ":" + tags[10]
                        self.link_value_cell = tags[2]
                    else:
                        self.curr_cell_info = "O:" + tags[0]
                    self.curr_cell_name = tags[0] + ":" + tags[1]
                else:
                    self.curr_cell_info, self.curr_cell_name = "", ""

                c.update_idletasks()
            self.update_current_cell_name_and_info()


        c.bind('<Motion>', motion)


        slider_frame = ttk.LabelFrame(self.window, relief=tk.RIDGE)
        slider_frame.grid(row=1, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.create_scrollbars(v_scrollbar_parent=slider_frame,
                               example_canvas=c)
        self.draw_nodes(iframe5, c)
        self.update_current_cell_name_and_info()

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
                scrollregion=(bbox[0], bbox[1] - top_padding, bbox[2], bbox[3] + bottom_padding)
            )

# can only run simulator.py now!
if __name__ == "__main__":
    lesion_rate = 0
    dropout_rate = 0
    perma_lesion_rate = 0.1
    rand10_net = Network(name="rand10x40")
    rand10_net.add_group(name="first", num_units=10, group_type="input",
                         input_transforms=[], output_transforms=[])
    rand10_net.add_group(name="second", num_units=70, group_type="hidden",
                         input_transforms=["dot"],
                         output_transforms=["sigmoid"])
    rand10_net.add_group(name="third", num_units=10, group_type="output",
                         input_transforms=["dot"],
                         output_transforms=["sigmoid"])

    # rand10_net.connect_groups(outgoing_group="first", incoming_group="second", initialization="uniform", proj_type="one-to-one")
    rand10_net.connect_groups(outgoing_group="first", incoming_group="second",
                              initialization="uniform", proj_type="random",
                              lesion_rate=lesion_rate,
                              dropout_rate=dropout_rate,
                              perma_lesion_rate=perma_lesion_rate)
    # rand10_net.connect_groups(outgoing_group="first", incoming_group="second", initialization="uniform", proj_type="full")
    rand10_net.connect_groups(outgoing_group="second", incoming_group="third",
                              initialization="uniform", proj_type="full")

    # load example set
    # path = r"examples\lens_example_input\rand10x40.ex"
    path = os.getcwd()
    src_index = path.index(r"\src")
    path = path[:src_index]
    path += r"\examples\lens_example_input\rand10x40.ex"
    print(path)
    # allows anyone to open \examples\lens_example_input\rand10x40.ex
    rand10_net.load_example_set(
        proc=False,
        file_name=path, testing=True)
        # file_name=r"C:\Users\yesinn.kwak\PycharmProjects\PYLens\src\examples\example_files\rand10x40.ex", testing=True)

    link_program = link_viewer(input_net=rand10_net)
    link_program.window.mainloop()
