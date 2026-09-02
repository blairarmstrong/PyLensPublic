import logging
import os
import regex as re
import runpy
import sys
import threading
from collections import defaultdict
from functools import partial
from multiprocessing import Process
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread
import sympy as sp

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import *

from PyLens.gui.graph_viewer_tk import GraphViewer
from PyLens.gui.unit_viewer_tk import FrameExamplesProgram
from PyLens.gui.link_viewer_tk import link_viewer
from PyLens.backend.network import Network
from PyLens.backend.parameters import NetworkParameters,OptimizerParameters

def raise_frame(frame):
    frame.tkraise()

class main_viewer_tk():
    # TODO: add capability to use gui without having an input_net pre-specified

    """
    The Main Viewer contains information about: network, display, file loading, training algorithms, training hyperparameters, training options, training controls, and an exit button.
    """
    def __init__(self, input_net=None,stop_event=None, master=None):
        """
        Initialization of the mainviewer.
        Setting up the basic structure and variables of the GUI.
        List of variables that are modifiable from the GUI of the input network.

        Args:
            input_net (Network): The network object that is being displayed.
            stop_event (threading.Event): Event to stop the training process.
            master (tk.Tk): The main window of the GUI.
        """
        self.window = tk.Tk()
        self.master = master
        self.stop_event = stop_event
        self.btn_lst = []
        self.input_net = input_net
        self.window.title("main viewer")
        self.window.geometry("600x810")
        self.default_num_cores = 2

        self.train_batch_size = tk.IntVar()
        self.train_weight_updates = tk.IntVar()
        self.train_error_crit = tk.IntVar()
        self.train_report_int = tk.IntVar()
        self.num_cores = tk.IntVar()
        self.is_closing = False  # True once GUI shutdown begins
        self.training_check_callback_id = None

        self.training = False
        self.buttons = [] # GUI buttons will be stored here to activate/disable at the same time
        self.create_widgets()

        if input_net:
            self.input_net=input_net
            self.set_initial_optimizer(self.input_net.update_method)
            self.set_example_sets(self.input_net)
            self.activate_buttons()
            self.update_params(self.input_net)

        # information about loading example
        self.ex_filename = None

        self.training_set_name = ""
        self.testing_set_name = ""

        self.network_defaults = None
        self.optimizer_defaults = None

        # this holds the viewers so we make sure only 1 link viewer
        # and one unit viewer is running at the same time
        global thread_dictionary
        thread_dictionary = {}
        thread_dictionary["unit"] = None
        thread_dictionary["link"] = None
        thread_dictionary["graph"] = None

        self.window.protocol("WM_DELETE_WINDOW", self.exit_program) # Make sure the program exits when clicking the 'x' button

        self.unit_viewer = None
        self.link_viewer_gui = None
        # self.window.mainloop()
    
    def show_warning_message(self, message):
        """
        Shows a pop-up warning box with a custom message.

        Args:
            message (str): The message to display in the warning box.
        """
        messagebox.showwarning("Warning", f"{message}")

    def get_plot_variable(self):
        """
        Returns variable to graph in graph viewer.
        """
        custom_entry = self.custom_plot_variable_entry.get()
        if custom_entry == "":
            print("Warning, no value has been selecting for plotting. Error will be plotted as default.")
            custom_entry = "error"
        self.input_net.stats_plotted[0] = custom_entry
        print("Setting plot variable to: {0}".format(custom_entry))
        return custom_entry

    def start_graph_viewer(self):
        """
        Starts a new graph viewer gui window.
        """
        # Get plot variable from text entry
        update_events = ["example", "weight update", "completion of a batch", "progress report", "training and testing"]
        user_choice = self.update_after_var.get()
        update_after = update_events.index(user_choice)
        plot_variable = self.get_plot_variable()
        window_name = plot_variable

        def idx_variable_filter(plot_variable):
            """
            Helper function to replace special vairable with regular name.

            Args:
                plot_variable (str): The variable to be plotted.
            """
            pattern = r'\b\w+(?:\[[^\]]+\])+(?:\.[\w]+(?:\[[^\]]+\])*)*'
            variable_pair = {}
            special_variables = re.findall(pattern, plot_variable)
            for idx, special_variable in enumerate(special_variables):
                new_name = f"idx_{idx}"
                variable_pair[sp.symbols(new_name)] = special_variable
                plot_variable = plot_variable.replace(special_variable, new_name)
            return variable_pair, plot_variable

        variable_pair, plot_variable = idx_variable_filter(plot_variable)
        for new_name, special_variable in variable_pair.items():
            try:
                exec("self.input_net." + special_variable)
            except:
                raise ValueError(f"Invalid variable: {special_variable} cannot be processed.")
        symbols = sp.sympify(plot_variable).free_symbols
        variables = {name: sp.symbols(name) for name in self.input_net.stats_plotter.progress_stats.keys()}

        input_net_attrs = {
        attr: sp.symbols(attr)
        for attr in dir(self.input_net)
        if not attr.startswith("_") and not callable(getattr(self.input_net, attr))
        }

        variables.update(input_net_attrs)
        for key, _ in variable_pair.items():
            variables[str(key)] = key
        for symbol in symbols:
            if str(symbol) not in variables:
                raise ValueError(f"Invalid variable: {symbol} not found in available variables.")
        try:
            expr = sp.sympify(plot_variable, locals=variables)
        except sp.SympifyError as e:
            print(f"Error in parsing input as symbolic math: {e}")
            expr = None

        # Create new graph viewer if variable exists
        graphviewer = GraphViewer(parent=self.window, network=self.input_net, 
                                  plot_variable=expr, special_variables=variable_pair, 
                                  window_name=window_name, update_after=update_after)
        self.input_net.graphs.append(graphviewer)
        
        print("starting graph viewer")
        graphviewer.start()
        graphviewer.window.mainloop()

    def start_graph_viewer_unit_test(self):
        """
        Starts a new graph viewer gui window.
        """
        # Get plot variable from text entry
        update_events = ["example", "weight update", "completion of a batch", "progress report", "training and testing"]
        user_choice = self.update_after_var.get()
        update_after = update_events.index(user_choice)
        plot_variable = self.get_plot_variable()
        window_name = plot_variable

        def idx_variable_filter(plot_variable):
            """
            Helper function to replace special vairable with regular name.
            """
            pattern = r'\b\w+(?:\[[^\]]+\])+(?:\.[\w]+(?:\[[^\]]+\])*)*'
            variable_pair = {}
            special_variables = re.findall(pattern, plot_variable)
            for idx, special_variable in enumerate(special_variables):
                new_name = f"idx_{idx}"
                variable_pair[sp.symbols(new_name)] = special_variable
                plot_variable = plot_variable.replace(special_variable, new_name)
            return variable_pair, plot_variable

        variable_pair, plot_variable = idx_variable_filter(plot_variable)
        for new_name, special_variable in variable_pair.items():
            try:
                exec("self.input_net." + special_variable)
            except:
                raise ValueError(f"Invalid variable: {special_variable} cannot be processed.")
        symbols = sp.sympify(plot_variable).free_symbols
        variables = {name: sp.symbols(name) for name in self.input_net.stats_plotter.progress_stats.keys()}

        input_net_attrs = {
        attr: sp.symbols(attr)
        for attr in dir(self.input_net)
        if not attr.startswith("_") and not callable(getattr(self.input_net, attr))
        }

        variables.update(input_net_attrs)
        for key, _ in variable_pair.items():
            variables[str(key)] = key
        for symbol in symbols:
            if str(symbol) not in variables:
                raise ValueError(f"Invalid variable: {symbol} not found in available variables.")
        try:
            expr = sp.sympify(plot_variable, locals=variables)
        except sp.SympifyError as e:
            print(f"Error in parsing input as symbolic math: {e}")
            expr = None
            
        # Create new graph viewer if variable exists
        graphviewer = GraphViewer(parent=self.window, network=self.input_net, 
                                  plot_variable=expr, special_variables=variable_pair, 
                                  window_name=window_name, update_after=update_after)
        self.input_net.graphs.append(graphviewer)

    def create_widgets(self):
        """
        Create all the widgets, frames and buttons of the main viewer.
        """

        # Create some room around all the internal frames
        self.window['padx'] = 5
        self.window['pady'] = 5

        self.create_frames()
        self.create_inside()


    def create_frames(self):
        """
        Configure the rows of each frame and create each row section.
        Each row is given a specific weight to mediate the size of the buttons.
        """
        self.window.rowconfigure(0,weight=1)
        self.window.rowconfigure(1,weight=1)
        self.window.rowconfigure(2,weight=1)
        self.window.rowconfigure(3,weight=1)
        self.window.rowconfigure(4,weight=1)
        self.window.rowconfigure(5,weight=1)
        self.window.rowconfigure(6,weight=1)
        self.window.rowconfigure(7,weight=1)

        self.window.columnconfigure(0,weight=1)

        self.network_info_frame = tk.Frame(self.window,borderwidth=1, relief=SUNKEN)
        self.network_info_frame.grid(row=0,column=0,sticky="NSEW")

        self.file_commands_frame = tk.Frame(self.window)
        self.file_commands_frame.grid(row=1, column=0, sticky="NSEW")

        self.display_frame = tk.Frame(self.window, borderwidth=1)
        self.display_frame.grid(row=2, column=0,sticky="NSEW", pady=(8, 0))

        self.algorithms_frame = tk.Frame(self.window,borderwidth=1, relief=SUNKEN)
        self.algorithms_frame.grid(row=3, column=0, sticky="NSEW")

        self.algorithms_param_frame = tk.Frame(self.window,borderwidth=1, relief=SUNKEN)
        self.algorithms_param_frame.grid(row=4, column=0, sticky="NSEW")

        self.training_param_frame = tk.Frame(self.window,borderwidth=1, relief=SUNKEN)
        self.training_param_frame.grid(row=5, column=0, sticky="NSEW")

        self.training_control_frame = tk.Frame(self.window)
        self.training_control_frame.grid(row=6, column=0, sticky="NSEW")

        self.task_control_frame = tk.Frame(self.window)
        self.task_control_frame.grid(row=7, column=0, sticky="NSEW")


    def create_inside(self):
        """
        Call the methods for creating the buttons inside each frame.
        As each row and frames contain different methods, multiple functions
        are called appropriately.
        """
        self.network_info_frames()
        self.display_buttons()
        self.file_commands_buttons()
        self.algo_buttons()
        self.algo_param_frames()
        self.training_param_frames()
        self.training_control_buttons()
        self.task_control_buttons()

    def network_info_frames(self):
        """
        The network info frame contains three drop down menus that allows users
        to choose the loaded network, training set and testing set.
        Each column of the frame is a menu of radio buttons.
        """
        self.network_info_frame.columnconfigure(0, weight=1, uniform="group1")
        self.network_info_frame.columnconfigure(1, weight=1, uniform="group1")
        self.network_info_frame.columnconfigure(2, weight=1, uniform="group1")

        # initialize radio button variable
        global train_set_radio, test_set_radio
        train_set_radio = tk.StringVar()
        test_set_radio = tk.StringVar()

       # create labels
        label1 = tk.Label(self.network_info_frame, text="Network")
        label2 = tk.Label(self.network_info_frame, text="Training Set")
        label3 = tk.Label(self.network_info_frame, text="Testing Set")

        # set variables for the network's pre-loaded training and testing sets
        if self.input_net:
            self.loaded_training = self.input_net.training_sets
            self.loaded_testing = self.input_net.testing_sets
            name = self.input_net.name
        else:
            self.loaded_training = []
            self.loaded_testing = []
            name = ''

        # a global dictionary of training and testing sets for easy access
        global train_values, test_values
        train_values = defaultdict()
        test_values = defaultdict()
        for training_examples in self.loaded_training:
            train_values[training_examples.name] = training_examples
        for testing_examples in self.loaded_testing:
            test_values[testing_examples.name] = testing_examples

        label1.grid(row=0,column=0)
        label2.grid(row=0,column=1)
        label3.grid(row=0,column=2)

        # setting the network name for the label - needs to be a dictionary
        # to facilitate inputs of multiple networks
        global network_name_menu
        network_name_menu = tk.Entry(self.network_info_frame, justify="center")
        network_name_menu.insert(END, name)
        network_name_menu.config(state='readonly')
        network_name_menu.grid(row=1,column=0,sticky="news")

        global training_set_menu_button, testing_set_menu_button
        # create dropdown menu for selecting training set
        training_set_menu_button = tk.Menubutton(self.network_info_frame, textvariable=train_set_radio, relief=GROOVE)
        training_set_menu_button.menu = tk.Menu(training_set_menu_button, tearoff=0)
        training_set_menu_button["menu"] = training_set_menu_button.menu

        # create dropdown menu for selecting testing set
        testing_set_menu_button = tk.Menubutton(self.network_info_frame, textvariable=test_set_radio, relief=GROOVE)
        testing_set_menu_button.menu = tk.Menu(testing_set_menu_button, tearoff=0)
        testing_set_menu_button["menu"] = testing_set_menu_button.menu

        # convert dictionary into respective radio buttons
        # for training and testing set
        for (text, value) in train_values.items():
            training_set_menu_button.menu.add_radiobutton(label=text, variable=train_set_radio,
                                                          command=lambda: self.update_training_set())

        for (text, value) in test_values.items():
            testing_set_menu_button.menu.add_radiobutton(label=text, variable=test_set_radio,
                                                         command=lambda: self.update_testing_set())

        training_set_menu_button.grid(row=1, column=1, sticky="news")
        testing_set_menu_button.grid(row=1, column=2, sticky="news")

        # set the default initial example sets to the current input net
        if self.input_net:
            self.set_example_sets(self.input_net)

        # If there are no loaded examples, disable training/testing set buttons
        if len(self.loaded_training) < 1:
            training_set_menu_button.configure(state="disabled")

        if len(self.loaded_testing) < 1:
            testing_set_menu_button.configure(state="disabled")

    def set_example_sets(self, input_net):
        """
        Helper function to activate the training and testings sets buttons from
        the input network.
        """
        self.loaded_training = input_net.training_sets
        self.loaded_testing = input_net.testing_sets

        for idx, training_set in enumerate(self.loaded_training):
            if training_set.name == self.input_net.training_set.name:
                train_set_radio.set(training_set.name)
                break
        for idx, testing_set in enumerate(self.loaded_testing):
            if testing_set.name == self.input_net.testing_set.name:
                test_set_radio.set(testing_set.name)
                break

        if len(self.loaded_training) >=1:
            training_set_menu_button.configure(state="active")
            training_set_menu_button.config(text=train_set_radio.get())

        if len(self.loaded_testing) >= 1:
            testing_set_menu_button.configure(state="active")
            testing_set_menu_button.config(text=test_set_radio.get())

    def update_training_set(self):
        """
        Helper function to facilitate the change of training set radio button.
        """
        print("Training set changed to: {0}".format(train_set_radio.get()))
        training_set_menu_button.config(text=train_set_radio.get())
        # print(train_values) # dictionary of example set
        self.input_net.training_set = train_values[train_set_radio.get()]

    def update_testing_set(self):
        """
        Helper function to facilitate the change of testing set radio button.
        """
        testing_set_menu_button.config(text=test_set_radio.get())
        self.input_net.testing_set = test_values[test_set_radio.get()]

    def display_buttons(self):
        """
        Construct buttons to create new displays/GUI such as the unit viewer,
        link viewer and an object viewer (currently disabled).
        """
        global amount_unit
        global amount_link
        amount_link = False
        amount_unit = False

        # create a frame to hold the buttons
        for i in range(3):
            self.display_frame.rowconfigure(i,weight=0)
        for j in range(2):
            self.display_frame.columnconfigure(j,weight=1,uniform="group1")

        # unit viewer button
        button1 = tk.Button(self.display_frame, text='Unit Viewer',padx=0,pady=0, relief=GROOVE, height=1,
                            command= lambda: self.start_unit_viewer(), state="disabled")
        button1.grid(row=0, column=0 ,sticky="ew")
        self.buttons.append(button1)

        # link viewer button
        button2 = tk.Button(self.display_frame, text='Link Viewer',padx=0,pady=0, relief=GROOVE, height=1,
                            command=lambda: Thread(target=self.start_link_viewer()).start(),
                            state="disabled")
        button2.grid(row=0, column=1,sticky="ew")
        self.buttons.append(button2)
        
        # Graph viewer button
        button3 = tk.Button(self.display_frame, relief=GROOVE, text='Graph Viewer', padx=0, pady=0, height=1,
                            command=lambda: self.start_graph_viewer(), state="disabled")
        button3.grid(row=1, column=0, sticky="ew")
        self.buttons.append(button3)

        # Graph viewer text entry box
        self.custom_plot_variable_entry = tk.Entry(self.display_frame, width=20, fg="gray")  
        self.custom_plot_variable_entry.grid(
                    row=1, column=1, sticky="ew"
                    )
        self.buttons.append(self.custom_plot_variable_entry)

        # Insert default text "Error"
        self.custom_plot_variable_entry.insert(0, "error")

        # Define a function to clear the error when user types
        def clear_error(event):
            if self.custom_plot_variable_entry.get() == "error":
                self.custom_plot_variable_entry.delete(0, tk.END)  # Clear the entry
                self.custom_plot_variable_entry.config(fg="black")  # Reset text color

        # Bind the entry box to detect when the user clicks inside
        self.custom_plot_variable_entry.bind("<FocusIn>", clear_error)

        # Disable input initially (optional, until enabled later)
        self.custom_plot_variable_entry.config(state="disabled") 

        # Create the dropdown variable
        self.update_after_var = tk.StringVar(self.display_frame)
        self.update_after_var.set("progress report")  # default

        # The list of options
        update_after_options = ["example", "weight update", "completion of a batch", "progress report", "training and testing"]

        # Create the dropdown
        self.update_after_dropdown = tk.OptionMenu(
            self.display_frame,
            self.update_after_var,
            *update_after_options
        )
        self.update_after_dropdown.grid(row=2, column=1, sticky="news")
        self.update_after_dropdown.config(state="disabled")  # or "normal" to enable
        self.buttons.append(self.update_after_dropdown)

        # Create a label that shows "Update after: ___"
        self.update_after_label = tk.Label(
            self.display_frame,
            text="(X axis) Graph update after:"
        )
        self.update_after_label.grid(row=2, column=0, sticky="news")

        # Whenever the user picks a new dropdown item, update the label's text
        def on_update_after_changed(*args):
            current_value = self.update_after_var.get()
            self.update_after_label.config(text=f"(X axis) Graph update after: {current_value}")

        # Trigger the callback whenever the StringVar changes
        self.update_after_var.trace_add("write", on_update_after_changed)

    def start_unit_viewer(self, cell_size=9, cell_spacing=3):
        """
        Helper function that starts the unit viewer.
        """

        global thread_dictionary
        # check if there is a unit viewer open close it and open another one
        if thread_dictionary["unit"] is not None:
            try:
                thread_dictionary["unit"].window.destroy()
            except:
                thread_dictionary["unit"] = None
        self.unit_viewer = FrameExamplesProgram(input_net=self.input_net, parent=self.window, cell_size=cell_size, cell_spacing=cell_spacing)
        thread_dictionary["unit"] = self.unit_viewer

    def start_link_viewer(self, cell_size=0, cell_spacing=0):
        """
        Helper function for the button that starts the link viewer.
        """
        global thread_dictionary
        # checkc if there is a link viewer open close it and open another one
        if thread_dictionary["link"] is not None:
            try:
                thread_dictionary["link"].window.destroy()
            except:
                thread_dictionary["link"] = None
        self.link_viewer_gui = link_viewer(self.window, input_net=self.input_net, cell_size=cell_size, cell_spacing=cell_spacing)
        thread_dictionary["link"] = self.link_viewer_gui

        self.link_viewer_gui.window.mainloop()

    def file_commands_buttons(self):
        """
        Create buttons for running scripts, saving weights and loading weights
        and examples
        Each button calls respective helper functions
        """
        self.file_commands_frame.columnconfigure(0,weight=1,uniform="group1")
        self.file_commands_frame.columnconfigure(1,weight=1,uniform="group1")

        # Run Script button (always active)
        button1 = tk.Button(self.file_commands_frame, text='Run Script', relief=GROOVE, height=1,
                            command=partial(self.load_file_callback, "py"))
        button1.grid(row=0, column=0, pady=0, padx=0,sticky="ew")

        # Save Weights button
        button2 = tk.Button(self.file_commands_frame, relief=GROOVE, text='Save Weights', height=1,
                            command=lambda: self.save_weights(), state="disabled")
        button2.grid(row=1, column=0,sticky="ew")
        self.buttons.append(button2)

        # Load Examples button (always active)
        button3 = tk.Button(self.file_commands_frame, text='Load Examples', height=1,
                            command=partial(self.load_file_callback, "ex"), relief=GROOVE)
        button3.grid(row=0, column=1,sticky="ew")

        # Load Weights button
        button4 = tk.Button(self.file_commands_frame, relief=GROOVE, text='Load Weights', height=1,
                            command=lambda: self.load_weights(), state="disabled")
        button4.grid(row=1, column=1,sticky="ew")
        self.buttons.append(button4)

    def load_weights(self):
        """
        Helper function for the load weights button.
        Opens a window for user to select the weight file and calls the
        load_weight function of the network - needs testing.
        """
        s = filedialog.askopenfilename(initialdir="../../", title="title",
                                       filetypes=(("PYLens, CLens weight file", ("*.pickle", "*.wt")),
                                       ("PYLens weight file","*.pickle"),
                                       ("CLens weight file", "*.wt"),
                                       ("all files", "*.*")))
        if s != "":
            _, extension = os.path.splitext(s)
            self.input_net.load_weight(fn=s)

    def save_weights(self):
        """
        Helper function for the save weighs button.
        Prompts the user to select the location to save the file by opening
        the a file dialog and calls the store_weight function of the network.
        TODO: Currently saves an additional empty file.
        """
        s = filedialog.asksaveasfile(initialdir="../../", mode="w",
                                       filetypes=(
                                       ("weight file","*.pickle"), ("all files", "*.*")))
        if s!="":
            self.input_net.store_weight(fn=s.name, format='pickle', weight_only=True, storage_objects=["weights"])

    def update_warning_label(self):
        for entry in self.param_entries:
            fg = entry.cget("fg")
            if fg == "red" or fg == "blue":
                self.warning_label.config(text="Warning: Some parameters not saved or contain errors.")
                return
        self.warning_label.config(text="")

    def algo_param_frames(self):
        """
        Create Labels + Entry widgets for algorithm parameters.
        Now the user can type, and the network updates without pressing Enter.
        """

        # Configure columns so the grid layout expands nicely
        for i in range(4):
            self.algorithms_param_frame.columnconfigure(i, weight=1, uniform="group1")

        for row in range(5):
            self.algorithms_param_frame.rowconfigure(row, weight=1)

        # Text for the labels
        texts = [
            "Learning Rate","Momentum","Weight Decay","Cost Strength",
            "Init Rand Range","Target Radius","Zero Error Radius",
            "Test Group Crit", "Clamp Strength", "Gain"
        ]

        col = 0
        text_counter = 0

        # Create labels in two columns: column=0 and column=2
        for i in range(10):
            if i == 5:
                col = 2
            row = i if i < 5 else i - 5

            label = tk.Label(self.algorithms_param_frame, text=texts[text_counter])
            label.grid(row=row, column=col, sticky="W")
            text_counter += 1

        # ---- Define all your "evaluate" functions for each parameter ----
        # NOTE: Now they do NOT need an 'event' parameter (because we're using trace_add, not <Return>).

        def out_focus_learning_rate(*_):
            current = learning_rate_menu.get()
            if str(current) != str(self.input_net.optimizer.learning_rate):
                if learning_rate_menu.cget("fg") != "red":
                    learning_rate_menu.config(fg="blue")
            else:
                learning_rate_menu.config(fg="black")
            self.update_warning_label()

        def evaluate_learning_rate(*_):
            val = learning_rate_menu.get()
            if val:
                try:
                    self.input_net.update_learning_rate(float(val))
                    print(f"Learning rate updated to: {val}")
                    learning_rate_menu.config(fg="black")
                except ValueError:
                    print("Error in updating learning rate")
                    print(f"Old value is used: {self.input_net.learning_rate}")
                    learning_rate_menu.config(fg="red")
            self.update_warning_label()
        
        def out_focus_momentum(*_):
            current = momentum_menu.get()
            if str(current) != str(self.input_net.optimizer.momentum):
                if momentum_menu.cget("fg") != "red":
                    momentum_menu.config(fg="blue")
            else:
                momentum_menu.config(fg="black")
            self.update_warning_label()

        def evaluate_momentum(*_):
            val = momentum_menu.get()
            if val:
                try:
                    self.input_net.update_optimizer_momentum(float(val))
                    print(f"Momentum updated to: {val}")
                    momentum_menu.config(fg="black")
                except ValueError:
                    print("Error in updating momentum")
                    print(f"Old value is used: {self.input_net.optimizer.momentum}")
                    momentum_menu.config(fg="red")
            self.update_warning_label()

        def out_focus_weight_decay(*_):
            current = weight_decay_menu.get()
            if str(current) != str(self.input_net.optimizer.weight_decay):
                if weight_decay_menu.cget("fg") != "red":
                    weight_decay_menu.config(fg="blue")
            else:
                weight_decay_menu.config(fg="black")
            self.update_warning_label()

        def evaluate_weight_decay(*_):
            val = weight_decay_menu.get()
            if val:
                try:
                    self.input_net.update_optimizer_weight_decay(float(val))
                    print(f"Weight decay updated to: {val}")
                    weight_decay_menu.config(fg="black")
                except ValueError:
                    print("Error in updating weight decay")
                    print(f"Old value is used: {self.input_net.optimizer.weight_decay}")
                    weight_decay_menu.config(fg="red")
            self.update_warning_label()
        
        def out_focus_cost_strength(*_):
            current = cost_strength_menu.get()
            if str(current) != str(self.input_net.output_cost_strength): 
                if cost_strength_menu.cget("fg") != "red":
                    cost_strength_menu.config(fg="blue")
            else:
                cost_strength_menu.config(fg="black")
            self.update_warning_label()

        def evaluate_cost_strength(*_):
            val = cost_strength_menu.get()
            if val:
                try: 
                    self.input_net.output_cost_strength = float(val)
                    print(f"Cost strength updated to: {val}")
                    cost_strength_menu.config(fg="black")
                except ValueError:
                    print("Error in updating cost strength")
                    print(f"Old value is used: {self.input_net.output_cost_strength}")
                    cost_strength_menu.config(fg="red")
            self.update_warning_label()

        def out_focus_init_rand_range(*_):
            current = init_rand_range_menu.get()
            if str(current) != str(self.input_net.rand_range): 
                if init_rand_range_menu.cget("fg") != "red":
                    init_rand_range_menu.config(fg="blue")
            else:
                init_rand_range_menu.config(fg="black")
            self.update_warning_label()

        def evaluate_init_rand_range(*_):
            val = init_rand_range_menu.get()
            if val:
                try:
                    self.input_net.rand_range = float(val)
                    print(f"Init rand range updated to: {val}")
                    init_rand_range_menu.config(fg="black")
                except ValueError:
                    print("Error in updating init rand range")
                    print(f"Old value is used: {self.input_net.rand_range}")
                    init_rand_range_menu.config(fg="red")
            self.update_warning_label()

        def out_focus_target_radius(*_):
            current = target_radius_menu.get()
            if str(current) != str(self.input_net.target_radius): 
                if target_radius_menu.cget("fg") != "red":
                    target_radius_menu.config(fg="blue")
            else:
                target_radius_menu.config(fg="black")
            self.update_warning_label()

        def evaluate_target_radius(*_):
            val = target_radius_menu.get()
            if val:
                try:
                    self.input_net.update_target_radius(float(val))
                    print(f"Target radius updated to: {val}")
                    target_radius_menu.config(fg="black")
                except ValueError:
                    print("Error in updating target radius")
                    print(f"Old value is used: {self.input_net.target_radius}")
                    target_radius_menu.config(fg="red")
            self.update_warning_label()

        def out_focus_zero_err_rad(*_):
            current = zero_err_rad_menu.get()
            if str(current) != str(self.input_net.zero_error_radius): 
                if zero_err_rad_menu.cget("fg") != "red":
                    zero_err_rad_menu.config(fg="blue")
            else:
                zero_err_rad_menu.config(fg="black")
            self.update_warning_label()

        def evaluate_zero_err_rad(*_):
            val = zero_err_rad_menu.get()
            if val:
                try:
                    self.input_net.update_zero_error_radius(float(val))
                    print(f"Zero error radius updated to: {val}")
                    zero_err_rad_menu.config(fg="black")
                except ValueError:
                    print("Error in updating zero error radius")
                    print(f"Old value is used: {self.input_net.zero_error_radius}")
                    zero_err_rad_menu.config(fg="red")
            self.update_warning_label()

        def out_focus_test_group_crit(*_):
            current = test_group_crit_menu.get()
            if str(current) != str(self.input_net.test_group_criterion_threshold): 
                if test_group_crit_menu.cget("fg") != "red":
                    test_group_crit_menu.config(fg="blue")
            else:
                test_group_crit_menu.config(fg="black")
            self.update_warning_label()

        def evaluate_test_group_crit(*_):
            val = test_group_crit_menu.get()
            if val:
                try:
                    self.input_net.test_group_criterion_threshold = float(val)
                    print(f"Test group criterion updated to: {val}")
                    test_group_crit_menu.config(fg="black")
                except ValueError:
                    print("Error in updating test group criterion")
                    print(f"Old value is used: {self.input_net.test_group_criterion_threshold}")
                    test_group_crit_menu.config(fg="red")
            self.update_warning_label()

        def out_focus_clamp_str(*_):
            current = clamp_str_menu.get()
            if str(current) != str(self.input_net.clamp_strength): 
                if clamp_str_menu.cget("fg") != "red":
                    clamp_str_menu.config(fg="blue")
            else:
                clamp_str_menu.config(fg="black")
            self.update_warning_label()

        def evaluate_clamp_str(*_):
            val = clamp_str_menu.get()
            if val:
                try:
                    self.input_net.clamp_strength = float(val)
                    print(f"Clamp strength updated to: {val}")
                    clamp_str_menu.config(fg="black")
                except ValueError:
                    print("Error in updating clamp strength")
                    print(f"Old value is used: {self.input_net.clamp_strength}")
                    clamp_str_menu.config(fg="red")
            self.update_warning_label()

        def out_focus_gains(*_):
            current = gains_menu.get()
            if str(current) != str(self.input_net.gain): 
                if gains_menu.cget("fg") != "red":
                    gains_menu.config(fg="blue")
            else:
                gains_menu.config(fg="black")
            self.update_warning_label()

        def evaluate_gains(*_):
            val = gains_menu.get()
            if val:
                try:
                    self.input_net.gain = float(val)
                    print(f"Gains updated to: {val}")
                    gains_menu.config(fg="black")
                except ValueError:
                    print("Error in updating gains")
                    print(f"Old value is used: {self.input_net.gain}")
                    gains_menu.config(fg="red")
            self.update_warning_label()

        global learning_rate_menu, momentum_menu, weight_decay_menu, cost_strength_menu
        global init_rand_range_menu, target_radius_menu, zero_err_rad_menu, test_group_crit_menu
        global clamp_str_menu, gains_menu

        self.param_entries = []

        learning_rate_menu = tk.Entry(self.algorithms_param_frame)
        learning_rate_menu.grid(row=0, column=1, sticky="news")
        learning_rate_menu.bind("<Return>", evaluate_learning_rate)
        learning_rate_menu.bind("<FocusOut>", out_focus_learning_rate)
        self.buttons.append(learning_rate_menu)
        self.param_entries.append(learning_rate_menu)

        momentum_menu = tk.Entry(self.algorithms_param_frame)
        momentum_menu.grid(row=1, column=1, sticky="news")
        momentum_menu.bind("<Return>", evaluate_momentum)
        momentum_menu.bind("<FocusOut>", out_focus_momentum)
        self.buttons.append(momentum_menu)
        self.param_entries.append(momentum_menu)

        weight_decay_menu = tk.Entry(self.algorithms_param_frame)
        weight_decay_menu.grid(row=2, column=1, sticky="news")
        weight_decay_menu.bind("<Return>", evaluate_weight_decay)
        weight_decay_menu.bind("<FocusOut>", out_focus_weight_decay)
        self.buttons.append(weight_decay_menu)
        self.param_entries.append(weight_decay_menu)

        cost_strength_menu = tk.Entry(self.algorithms_param_frame)
        cost_strength_menu.grid(row=3, column=1, sticky="news")
        cost_strength_menu.bind("<Return>", evaluate_cost_strength)
        cost_strength_menu.bind("<FocusOut>", out_focus_cost_strength)
        self.buttons.append(cost_strength_menu)
        self.param_entries.append(cost_strength_menu)

        init_rand_range_menu = tk.Entry(self.algorithms_param_frame)
        init_rand_range_menu.grid(row=4, column=1, sticky="news")
        init_rand_range_menu.bind("<Return>", evaluate_init_rand_range)
        init_rand_range_menu.bind("<FocusOut>", out_focus_init_rand_range)
        self.buttons.append(init_rand_range_menu)
        self.param_entries.append(init_rand_range_menu)


        target_radius_menu = tk.Entry(self.algorithms_param_frame)
        target_radius_menu.grid(row=0, column=3, sticky="news")
        target_radius_menu.bind("<Return>", evaluate_target_radius)
        target_radius_menu.bind("<FocusOut>", out_focus_target_radius)
        self.buttons.append(target_radius_menu)
        self.param_entries.append(target_radius_menu)

        zero_err_rad_menu = tk.Entry(self.algorithms_param_frame)
        zero_err_rad_menu.grid(row=1, column=3, sticky="news")
        zero_err_rad_menu.bind("<Return>", evaluate_zero_err_rad)
        zero_err_rad_menu.bind("<FocusOut>", out_focus_zero_err_rad)
        self.buttons.append(zero_err_rad_menu)
        self.param_entries.append(zero_err_rad_menu)

        test_group_crit_menu = tk.Entry(self.algorithms_param_frame)
        test_group_crit_menu.grid(row=2, column=3, sticky="news")
        test_group_crit_menu.bind("<Return>", evaluate_test_group_crit)
        test_group_crit_menu.bind("<FocusOut>", out_focus_test_group_crit)
        self.buttons.append(test_group_crit_menu)
        self.param_entries.append(test_group_crit_menu)

        clamp_str_menu = tk.Entry(self.algorithms_param_frame)
        clamp_str_menu.grid(row=3, column=3, sticky="news")
        clamp_str_menu.bind("<Return>", evaluate_clamp_str)
        clamp_str_menu.bind("<FocusOut>", out_focus_clamp_str)
        self.buttons.append(clamp_str_menu)
        self.param_entries.append(clamp_str_menu)


        gains_menu = tk.Entry(self.algorithms_param_frame)
        gains_menu.grid(row=4, column=3, sticky="news")
        gains_menu.bind("<Return>", evaluate_gains)
        gains_menu.bind("<FocusOut>", out_focus_gains)
        self.buttons.append(gains_menu)
        self.param_entries.append(gains_menu)

        self.warning_label = tk.Label(self.algorithms_param_frame, text="", fg="red")
        self.warning_label.grid(row=5, column=0, columnspan=4, sticky="w")

        self.update_params(self.input_net)

    def update_params(self, input_net=None):
        """
        Helper function that fills in the value of the entry widgets in the
        algorithm frame according to the default values specified in
        NetworkParameters and Optimizer Parameters.

        Args:
            input_net (Network)
        """
        self.network_defaults = NetworkParameters()
        self.optimizer_defaults = OptimizerParameters()
        if input_net != None:
            learning_rate_menu.delete(0, len(learning_rate_menu.get()))
            learning_rate_menu.insert(10, self.input_net.optimizer.learning_rate)
            learning_rate_menu.config(fg="black")

            momentum_menu.delete(0,len(momentum_menu.get()))
            momentum_menu.insert(10, self.input_net.optimizer.momentum)
            momentum_menu.config(fg="black")

            weight_decay_menu.delete(0,len(weight_decay_menu.get()))
            weight_decay_menu.insert(10, self.input_net.optimizer.weight_decay)
            weight_decay_menu.config(fg="black")
            
            cost_strength_menu.delete(0,len(cost_strength_menu.get()))
            cost_strength_menu.insert(10, self.input_net.output_cost_strength)
            cost_strength_menu.config(fg="black")

            init_rand_range_menu.delete(0,len(init_rand_range_menu.get()))
            init_rand_range_menu.insert(10, self.input_net.rand_range)
            init_rand_range_menu.config(fg="black")

            target_radius_menu.delete(0,len(target_radius_menu.get()))
            target_radius_menu.insert(10, self.input_net.target_radius)
            target_radius_menu.config(fg="black")

            zero_err_rad_menu.delete(0,len(zero_err_rad_menu.get()))
            zero_err_rad_menu.insert(10, self.input_net.zero_error_radius)
            zero_err_rad_menu.config(fg="black")

            test_group_crit_menu.delete(0, len(test_group_crit_menu.get()))
            test_group_crit_menu.insert(10, self.input_net.test_group_criterion_threshold)
            test_group_crit_menu.config(fg="black")

            clamp_str_menu.delete(0,len(clamp_str_menu.get()))
            clamp_str_menu.insert(10, self.input_net.clamp_strength)
            clamp_str_menu.config(fg="black")

            gains_menu.delete(0,len(gains_menu.get()))
            gains_menu.insert(10, self.input_net.gain)
            gains_menu.config(fg="black")

    def set_initial_optimizer(self, net_optimizer):
        """
        Helper function that defaults the optimizer radio button to the
        originally selected optimizer of the network.

        Args:
            net_optimizer (str): The optimizer of the network
        """
        if net_optimizer == "steepest":
            self.optimizer_radio.set(101)
        elif net_optimizer == "momentum":
            self.optimizer_radio.set(102)
        elif net_optimizer == "dougs momentum":
            self.optimizer_radio.set(103)
        elif net_optimizer == "delta bar delta":
            self.optimizer_radio.set(104)
        elif net_optimizer == "adam":
            self.optimizer_radio.set(105)
        else:
            self.optimizer_radio.set(101)

    def algo_buttons(self):
        """
        Create the radio buttons for optimizer selection of the network.
        The selection of a radio button calls the function set_optimizer
        to update the network's optimizer appropriately.
        """
        self.algorithms_frame.columnconfigure(0, weight=1, uniform="group1")
        self.algorithms_frame.columnconfigure(1, weight=1, uniform="group1")
        self.optimizer_radio = tk.IntVar()
        global texts
        texts = [("Steepest",101),("Momentum",102),
                 ("Dougs momentum",103), ("Delta bar delta",104),
                 ("Adam",105)]

        text_counter = 0
        col = 0
        rad_val = 0
        # for loop to populate the radio buttons accordingly into two columns
        for i in range(len(texts)):
            if i == 3:
                col = 1
            if i >= 3:
                i -= 3
            radio_val = texts[rad_val][1]
            input_text = texts[text_counter][0]
            button = tk.Radiobutton(self.algorithms_frame,
                                    text=input_text,
                                    variable=self.optimizer_radio,
                                    command=lambda: self.set_optimizer(),
                                    value=texts[rad_val][1],
                                    state="disabled")
            button.grid(row=i, column=col, sticky="nw")
            self.buttons.append(button)
            rad_val += 1
            text_counter += 1


    def set_optimizer(self):
        """
        Helper function that facilitates the selection of the optimizer
        for the network through radio button selection.
        Calls the set_update_method helper function to mediate the selection.
        """
        cur_choice = self.optimizer_radio.get()
        if cur_choice == 101:
            self.input_net.set_update_method(self.input_net.learning_rate,"steepest")
        elif cur_choice == 102:
            self.input_net.set_update_method(self.input_net.learning_rate, "momentum")
        elif cur_choice == 103:
            self.input_net.set_update_method(self.input_net.learning_rate, "dougs momentum")
        elif cur_choice == 104:
            self.input_net.set_update_method(self.input_net.learning_rate, "delta bar delta")
        elif cur_choice == 105:
            self.input_net.set_update_method(self.input_net.learning_rate, "adam")


    def update_train_params(self, user_input, train_param):
        """
        Helper function to allow the updating of the training parameters.

        Args:
            user_input (str): The user input value.
            train_param (int): The index of the training parameter to update.
        """
        net_train_params = [self.train_batch_size,self.train_weight_updates,
                            self.train_error_crit, self.train_report_int, self.num_cores]
        try:
            net_train_params[train_param].set(int(user_input))
            if train_param == 2:
                self.input_net.batch_error_threshold = net_train_params[2].get()

        except Exception as e:
            logging.info(repr(e))

    def varify_parallel_training(self):
        """
        Helper function that allows the user to enable parallel training.
        """
        if self.parallel_training.get() == 1:
            self.num_cores_menu.config(state="normal")
        else:
            self.num_cores_menu.config(state="normal")
            self.num_cores.set(self.default_num_cores)  # Reset value
            self.num_cores_menu.delete(0, tk.END)  # Clear existing text
            self.num_cores_menu.insert(0, str(self.default_num_cores))  # Insert default value
            self.num_cores_menu.config(state="disabled")
            
    def training_param_frames(self):
        """
        Create entry and label widgets that allows user to change the training
        parameters of the network
        Default parameters are originally set for the network and trace is
        used so that helper function can access the user input
        """

        self.parallel_training = tk.IntVar(
            value=int(self.input_net.parallel_mode) if self.input_net is not None else 0
        )
        self.num_cores_menu = None
        for i in range(4):
            self.training_param_frame.columnconfigure(i, weight=1,
                                                        uniform="group1")
        # initialize trace variables
        num_cores_trace = tk.StringVar()
        batch_size_trace = tk.StringVar()
        weight_updates_trace = tk.StringVar()
        error_crit_trace = tk.StringVar()
        report_int_trace = tk.StringVar()

        texts = ["Batch Size","Error Criterion","Weight Updates","Report Interval", "Number of Cores"]
        col = 0
        text_counter = 0
        for i in range(4):
            if i == 2:
                col = 2
            if i >= 2:
                i -= 2
            label = tk.Label(self.training_param_frame, text=texts[text_counter])
            label.grid(row=i,column=col,sticky="W")
            text_counter += 1
        
        label = tk.Label(self.training_param_frame, text=texts[4])
        label.grid(row=2,column=2,sticky="W")

        batch_size_menu = tk.Entry(self.training_param_frame, textvariable=batch_size_trace)
        batch_size_menu.insert(10, self.network_defaults.PAR_N_batchSize)
        batch_size_menu.config(state="disabled")
        self.train_batch_size.set(self.network_defaults.PAR_N_batchSize)
        self.buttons.append(batch_size_menu)

        weight_updates_menu = tk.Entry(self.training_param_frame, textvariable=weight_updates_trace)
        weight_updates_menu.insert(10, self.network_defaults.PAR_N_numUpdates)
        weight_updates_menu.config(state="disabled")
        self.train_weight_updates.set(self.network_defaults.PAR_N_numUpdates)
        self.buttons.append(weight_updates_menu)

        error_crit_menu = tk.Entry(self.training_param_frame, textvariable=error_crit_trace)
        error_crit_menu.insert(10, self.network_defaults.PAR_N_criterion)
        error_crit_menu.config(state="disabled")
        self.train_error_crit.set(self.network_defaults.PAR_N_criterion)
        self.buttons.append(error_crit_menu)
        
        report_int_menu = tk.Entry(self.training_param_frame, textvariable=report_int_trace)
        report_int_menu.insert(10,self.network_defaults.PAR_N_reportInterval)
        report_int_menu.config(state="disabled")
        self.train_report_int.set(self.network_defaults.PAR_N_reportInterval)
        self.buttons.append(report_int_menu)

        num_cores_menu = tk.Entry(self.training_param_frame, textvariable=num_cores_trace)
        num_cores_menu.insert(10, self.default_num_cores)
        num_cores_menu.config(state="disabled")
        self.num_cores.set(self.default_num_cores)
        self.num_cores_menu = num_cores_menu

        paralle_training_button = tk.Checkbutton(self.training_param_frame,
                                         text="Parallel Training",
                                         variable=self.parallel_training,
                                         command=lambda: self.varify_parallel_training(),
                                         state="disabled")

        paralle_training_button.grid(row=2, column=0, sticky="nw")
        self.buttons.append(paralle_training_button)

        batch_size_menu.grid(row=0,column=1,sticky="news")
        weight_updates_menu.grid(row=0,column=3,sticky="news")
        error_crit_menu.grid(row=1, column=1, sticky="news")
        report_int_menu.grid(row=1,column=3,sticky="news")
        num_cores_menu.grid(row=2,column=3,sticky="news")

        # for each change in the entry of the menu, update_train_params is
        # called to reflect the changes that the user input
        batch_size_trace.trace("w", lambda name, index, mode,
           train__size=batch_size_trace: self.update_train_params(batch_size_trace.get(),0))
        weight_updates_trace.trace("w", lambda name, index, mode,
           train_batch_size=batch_size_trace: self.update_train_params(weight_updates_trace.get(),1))
        error_crit_trace.trace("w", lambda name, index, mode,
           train_batch_size=batch_size_trace: self.update_train_params(error_crit_trace.get(),2))
        report_int_trace.trace("w", lambda name, index, mode,
           train_batch_size=batch_size_trace: self.update_train_params(report_int_trace.get(),3))
        num_cores_trace.trace("w", lambda name, index, mode,
           train_batch_size=batch_size_trace: self.update_train_params(num_cores_trace.get(),4))

    def training_control_buttons(self):
        """
        Create buttons that control the training of the network.
        Currently the feature of 3 of 4 buttons are unavailable.
        Radio button is used to modulate the training such that only one can
        be pressed/available at a time.
        """
        global test_button
        self.training_control_frame.columnconfigure(0,weight=1,uniform="group1")
        self.training_control_frame.columnconfigure(1,weight=1,uniform="group1")
        self.training_control_frame.rowconfigure(2,weight=1)

        # Reset Network button
        button1 = tk.Button(self.training_control_frame, relief=GROOVE,
                            text='Reset Network',padx=0,pady=0, command=lambda: self.reset_network(),
                            state="disabled")
        button1.grid(row=0, column=0, pady=0, padx=0,sticky="news")
        self.buttons.append(button1)
        
        # Test Network button
        test_button = tk.Button(self.training_control_frame, relief=GROOVE,
                                text='Test Network',padx=0,pady=0,
                                command=lambda: self.test_network())
        test_button.grid(row=1, column=0,sticky="news", pady=0, padx=0)

        button3 = tk.Button(self.training_control_frame, relief=GROOVE,
                            text='Reset Training Set',padx=0,pady=0)
        button3.grid(row=0, column=1, pady=0, padx=0,sticky="news")
        button3["state"] = DISABLED

        button4 = tk.Button(self.training_control_frame, relief=GROOVE,
                            text='Record Outputs',padx=0,pady=0)
        button4.grid(row=1, column=1,sticky="news", pady=0, padx=0)
        button4["state"] = DISABLED

        # A radio button that calls helper function to start or stop training
        # of the network. The default selection is "stop" as "start" should
        # be available at first
        self.training_radio = tk.IntVar()
        global start_radio, stop_radio
        start_radio = tk.Radiobutton(self.training_control_frame,
                                     text="Start Training", relief=GROOVE, 
                                     indicatoron = 0,
                                     height=4,
                                     variable=self.training_radio,
                                     command=lambda: self.start_training(),
                                     value=1001)

        stop_radio = tk.Radiobutton(self.training_control_frame,
                                     text="Stop Training", indicatoron=0, relief=GROOVE,
                                     height=4,
                                     variable=self.training_radio,
                                     command=lambda: self.stop_training(),
                                     value=1002)

        start_radio.grid(row=2, column=0,sticky="ew")
        stop_radio.grid(row=2, column=1,sticky="ew")
        self.update_training_control_buttons()

    def update_training_control_buttons(self):
        """
        Helper function that updates the radio button status and the training
        control buttons.
        """
        if self.training:
            self.training_radio.set(1001) 
            start_radio.configure(state="disabled")
            stop_radio.configure(state="active")
        else:
            self.training_radio.set(1002)
            start_radio.configure(state="active")
            stop_radio.configure(state="disabled")

        # Disable training if there are no training examples loaded
        if len(self.loaded_training) < 1:
            start_radio.configure(state="disabled")
            stop_radio.configure(state="disabled")
        else:
            start_radio.configure(state="active")
            stop_radio.configure(state="active")

        # Disable testing if there are not testing examples loaded
        if len(self.loaded_testing) < 1:
            test_button.configure(state="disabled")
        else:
            test_button.configure(state="active")

    def test_network(self):
        """
        Function that gets called when the test network button is pressed.
        A helper function that uses a hook function in the backend.
        """
        print(self.train_batch_size.get())
        self.input_net.test(self.train_batch_size.get(), reset_error=True)

    def start_training_thread(self):
        """
        Helper function that allows the training to be executed on a new thread.
        """
        # global training_thread
        # training_thread = threading.Thread(target=self.start_training)
        # training_thread.start()
        self.start_training()

    def start_training(self):
        """
        Helper function that calls a hook in the network to start training.
        Sets the radio button status accordingly.
        """
        # Resets the stats for the plotting
        # However upon pressing START TRAINING a second time, a dimension
        # error is thrown from the stats plotter - not fully reset
        self.input_net.stats_plotter.reset_stats()
        self.input_net.user_interrupt = False
        self.input_net.res = ""

        if self.training:
            return

        self.training = True
        self.deactivate_buttons()
        self.update_training_control_buttons()

        # Can't change training / test set while training
        training_set_menu_button.configure(state="disabled")
        testing_set_menu_button.configure(state="disabled")

        print("Training with batch size:{0} | weight updates:{1}  | err crit: {2}  | report int: {3}"
              .format(self.train_batch_size.get(),self.train_weight_updates.get(),
                       self.train_error_crit.get(), self.train_report_int.get()))

        self.training_thread = Thread(
            target=self.input_net.train,
            args=(
                self.train_weight_updates.get(),
                self.train_batch_size.get(),
            ),
            kwargs={
                "report_interval": self.train_report_int.get(),
                "stop_event": self.stop_event,
                "parallel_mode": self.parallel_training.get(),
                "num_worker": self.num_cores.get(),
            },
            daemon=True,
        )

        self.training_thread.start()
        self.training_check_callback_id = self.window.after(
            100,
            self.check_training_finished,
        )


    def check_training_finished(self):
        self.training_check_callback_id = None
        if self.is_closing:
            return

        if self.training_thread.is_alive():
            self.training_check_callback_id = self.window.after(
                100,
                self.check_training_finished,
            )
            return

        self.training = False
        self.activate_buttons()
        self.update_training_control_buttons()
        training_set_menu_button.configure(state="active")
        testing_set_menu_button.configure(state="active")

    def stop_training(self):
        """
        Helper function that calls a hook in the network to stop training.
        Sets the radio button status accordingly.
        """
        #  to prevent flag going off during non-training
        if self.training == True:
            self.training = False
            stop_radio.config(text="Stopping...")
            self.training_radio.set(1002)
            print("stopping .... ")
            self.input_net.user_interrupt = True
            start_radio.config(state="disabled")
        
    def update_stop_radio(self):
        """
        Helper function that updates the radio button status after
        the training has stopped.
        """
        if self.is_closing:
            return

        def update_gui():
            if self.is_closing:
                return

            stop_radio.config(text="Stop Training")
            start_radio.config(state="active")

        self.window.after(0, update_gui)

    def task_control_buttons(self):
        """
        Create buttons for waiting and exiting the mainviewer.
        """
        self.task_control_frame.columnconfigure(0,weight=1)
        self.task_control_frame.columnconfigure(1,weight=4)
        self.task_control_frame.columnconfigure(2,weight=1)
        self.task_control_frame.rowconfigure(0,weight=1)

        button1 = tk.Button(self.task_control_frame, relief=GROOVE, text='Wait',padx=0,pady=0, height=2)
        button1.grid(row=0, column=0, pady=0, padx=0,sticky="ew")
        button1["state"] = DISABLED

        button2 = tk.Button(self.task_control_frame, relief=GROOVE, text='Exit',
                            padx=0,pady=0, height=2, command=self.exit_program)
        button2.grid(row=0, column=1,sticky="ew", pady=0, padx=0)

    def load_example_popup(self, path):
        """
        Helper function for loading examples.

        Args:
            path (Path): The path of the example file.
        """
        self.loading_ex_setting_win = tk.Toplevel()
        self.loading_ex_setting_win.wm_title("Load Examples")

        global example_load_set, number_load_examples

        label1 = tk.Label(self.loading_ex_setting_win, text="Set in which to load examples:")
        label1.grid(row=0, column=0)
        example_load_set = tk.Entry(self.loading_ex_setting_win)
        example_load_set.grid(row=1, column=0)
        label2 = tk.Label(self.loading_ex_setting_win, text="Number of examples:")
        label2.grid(row=2, column=0)
        number_load_examples = tk.Entry(self.loading_ex_setting_win)
        number_load_examples.grid(row=3, column=0)

        self.load_mode = tk.StringVar(value="replace")  # default

        self.mode_var = tk.StringVar(value="replace")

        mode_frame = tk.Frame(self.loading_ex_setting_win)
        mode_frame.grid(row=4, column=0, columnspan=2, pady=5)

        replace_radio = tk.Radiobutton(mode_frame, text="Replace", variable=self.mode_var, value="replace")
        replace_radio.pack(side="left", padx=10)

        append_radio = tk.Radiobutton(mode_frame, text="Append", variable=self.mode_var, value="append")
        append_radio.pack(side="left", padx=10)

        cancel = ttk.Button(self.loading_ex_setting_win, text="Cancel", command=self.loading_ex_setting_win.destroy)
        cancel.grid(row=5, column=1)
        okay = ttk.Button(self.loading_ex_setting_win, text="Okay", command=partial(self.set_loading_example_param,
                                                                                    path))
        okay.grid(row=5, column=0)

    def set_loading_example_param(self, path: Path) -> None:
        print("setting loading ex parameters...")
        load_set = example_load_set.get().lower()
        no_examples = number_load_examples.get()

        try:
            no_examples = int(no_examples)
        except:
            no_examples = None

        name = path.stem
        filepath_str = path.as_posix()
        mode = self.mode_var.get()  # <-- add this line

        if "train" not in load_set and "test" not in load_set:
            if len(train_values) == 0:
                print("No valid set specified. Loading to training set")
                load_set = 'training'
            elif len(test_values) == 0:
                print("No valid set specified. Loading to testing set")
                load_set = 'testing'
            else:
                print("No valid set specified. Defaulting to training set")
                load_set = 'training'

        if 'train' in load_set:
            if mode == "replace":
                # Clear current training sets
                self.input_net.training_sets.clear()
                self.input_net.training_set = None
                train_values.clear()
                training_set_menu_button.menu.delete(0, 'end')  # remove all menu items
                train_set_radio.set(name)

            new_example_set = self.input_net.load_example_set(
                filepath_str, num_examples_loaded=no_examples, name=name,
                training=True, testing=False
            )
            training_set_menu_button.menu.add_radiobutton(
                label=name, variable=train_set_radio,
                command=lambda: self.update_training_set()
            )
            train_values[new_example_set.name] = new_example_set

        elif 'test' in load_set:
            if mode == "replace":
                # Clear current testing sets
                self.input_net.testing_sets.clear()
                test_values.clear()
                testing_set_menu_button.menu.delete(0, 'end')
                self.input_net.testing_set = None
                test_set_radio.set(name)

            new_example_set = self.input_net.load_example_set(
                filepath_str, num_examples_loaded=no_examples, name=name,
                training=False, testing=True
            )
            testing_set_menu_button.menu.add_radiobutton(
                label=name, variable=test_set_radio,
                command=lambda: self.update_testing_set()
            )
            test_values[new_example_set.name] = new_example_set

        if no_examples is None:
            no_examples = "all"
        print(f"Loaded {no_examples} examples from {path} into {load_set} set.")

        self.loading_ex_setting_win.destroy()
        self.set_example_sets(self.input_net)
        self.update_training_control_buttons()


    def update_network_name(self, name):
        """
        Helper function to update the network name in the GUI.
        """
        curr_name = network_name_menu.get()
        network_name_menu.config(state=NORMAL)
        network_name_menu.delete(0,len(curr_name))
        network_name_menu.insert(0, self.input_net.name)
        network_name_menu.config(state='readonly')

    def load_file_callback(self, type):
        """
        Open a file dialog the facilitate the functions of run script and
        load example. Run script is currently hardcoded to the xor script
        as opening the script starts the training instantaneously.
        TODO: dynamic pathing of the script file and fix instantaneously running behavior

        Args:
            type (str): The type of file to load.
        """
        if type == "ex":
            title = "Load Examples"
            filetype = ("example files","*.ex")
        else: # type = "py"
            title = "Run Scripts"
            filetype = ("python script","*.py")
        filename = filedialog.askopenfilename(initialdir="../../examples", title=title,
                                                   filetypes = (filetype,("all files","*.*")))

        if filename != "":
            filepath = Path(filename)
            extension = filepath.suffix

            if extension[1:] == type:
                if type == "ex":
                    self.load_example_popup(filepath)
                else: # Run script from selected file
                    from PyLens.simulator import Simulator
                    original_use_gui = Simulator.use_gui

                    # intercept the redundant gui call in script by user
                    try:
                        Simulator.use_gui = lambda self, *args, **kwargs: None
                        script = runpy.run_path(
                            filepath.as_posix(),
                            run_name="pylens_loaded_script"
                        )
                    finally:
                        Simulator.use_gui = original_use_gui
                    
                    for value in script.values():
                        if isinstance(value, Network):
                            self.input_net = value
                    print(f"{self.input_net.name} network loaded.")
                    self.update_network_name(self.input_net.name)
                    self.set_initial_optimizer(self.input_net.update_method)
                    self.set_example_sets(self.input_net)
                    self.activate_buttons()
                    self.update_params(self.input_net)
                    self.update_training_control_buttons()

                    if len(self.input_net.training_sets) > 0:
                        training_set_menu_button.menu.add_radiobutton(label=self.input_net.training_sets[0].name, variable=train_set_radio,
                                                            command=lambda: self.update_training_set())
                    if len(self.input_net.testing_sets) > 0:
                        testing_set_menu_button.menu.add_radiobutton(label=self.input_net.testing_sets[0].name, variable=test_set_radio,
                                                            command=lambda: self.update_testing_set())
            else:
                messagebox.showerror("File Format", "Need a "+type+" file")

    def reset_network(self):
        """
        Reset the network to the initial state.
        """
        print("Attempting to reset network.")
        self.input_net.reset_network()
        self.update_params(self.input_net)
        print("Network reset.")
    
    def activate_buttons(self):
        """
        Sets the state of buttons to active.
        """
        if self.input_net:
            for button in self.buttons:
                try:
                    button.config(state="active")
                except:
                    button.config(state="normal")
            self.varify_parallel_training()

    def deactivate_buttons(self):
        for button in self.buttons:
            button.config(state="disabled")
        self.num_cores_menu.config(state="disabled")

    def check_for_update(self, last_example_trained, s):
        """
        Check if the viewers need to be updated.
        """
        update_events = {
            0: "example",
            1: "weight update",
            2: "completion of a batch",
            3: "progress report",
            4: "training and testing"
        }

        def needs_update(viewer, s):
            """
            Helper function to determine if update is needed
            
            Args:
                viewer (Viewer): The viewer to check for updates.
                s (str): The event to check for.
            """
            if viewer is None:
                return False
            return update_events.get(viewer.update_display_option) == s

        if needs_update(self.link_viewer_gui, s):
            self.link_viewer_gui.update_plots()

        # Update unit viewer
        if needs_update(self.unit_viewer, s):
            uv = self.unit_viewer
            if uv.update_display_option == 2:
                first_item_idx = uv.input_example_index_list[0]
                uv.listbox.selection_clear(0, tk.END)
                uv.listbox.select_set(first_item_idx)
                uv.listbox.see(first_item_idx)
            if uv.update_display_option == 0:
                uv._curr_ex_idx = uv.input_example_list.index(last_example_trained)
                uv.listbox.selection_clear(0, tk.END)
                uv.listbox.select_set(uv._curr_ex_idx)
                uv.draw_tick()
            else:
                uv.update_canvas()
        if self.link_viewer_gui is not None:
            self.link_viewer_gui.stop_event.set()
        if self.unit_viewer is not None:
            self.unit_viewer.stop_event.set()

    def exit_program(self):
        self.is_closing = True
        if self.training_check_callback_id is not None:
            self.window.after_cancel(self.training_check_callback_id)
        self.window.destroy()

#
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
    path = os.getcwd()
    src_index = path.index(r"\src")
    path = path[:src_index]
    path += r"\examples\lens_example_input\rand10x40.ex"
    # this code allows anyone to run main_viewer_tk.py
    # make sure env variable BASETYPE=numpy

    rand10_net.load_example_set(
         proc=False, name="example 1",
         file_name=path)
    rand10_net.load_example_set(
         proc=False, name="example 2",
         file_name=path)
    rand10_net.load_example_set(
         proc=False, name="example 3",
         file_name=path)
    rand10_net.load_example_set(
         proc=False, name="example 4",
         file_name=path)
    rand10_net.load_example_set(
         proc=False, name="example 5",
         file_name=path)
    app = main_viewer_tk(input_net=rand10_net)
#
