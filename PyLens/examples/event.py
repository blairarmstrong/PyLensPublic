from typing import List
import re
from .unit_group import UnitGroup
import numpy as np

# EVENT FIELDS
DEF_S_MAX_TIME = None
DEF_S_MIN_TIME = None
DEF_S_GRACE_TIME = None


class Event:
    """It consist of information related to one event of the Example object
    and has UnitGroup object as inputs and targets

    """
    shared_inputs: bool  # flag
    shared_targets: bool  # flag
    max_time: float
    min_time: float
    grace_time: float
    default_input: float
    active_input: float
    default_target: float
    active_target: float
    input_group = []  # List[np]
    target_group = []  # List[np]
    example = None
    pre_proc_name = None
    post_proc_name = None

    def __init__(self, ex):
        self.set = ex.set
        self.example = ex
        self.max_time = DEF_S_MAX_TIME
        self.min_time = DEF_S_MIN_TIME
        self.grace_time = DEF_S_GRACE_TIME
        self.default_input = self.set.default_input
        self.active_input = self.set.active_input
        self.default_target = self.set.default_target
        self.active_target = self.set.active_target
        self.input_group = []
        self.target_group = []
        self.pre_proc_name = None
        self.post_proc_name = None
        self.input_group_len = []
        self.input_group_name = []
        self.target_group_len = []
        self.target_group_name = []
        self.network = self.set.network

    def pre_proc(self):
        """initialize and execute the code associated with this Event"""
        if self.pre_proc_name in self.set.procs_dict:
            proc_code = self.example.set.procs_dict[self.pre_proc_name]
            exec(proc_code)
        else:
            exec("print('error: no pre proc defined for this event')")

    def post_proc(self):
        """initialize and execute the code associated with this Event"""
        if self.post_proc_name in self.set.procs_dict:
            proc_code = self.example.set.procs_dict[self.post_proc_name]
            exec(proc_code)
        else:
            exec("print('error: no post proc defined for this event')")

    def parse_event_header_string(self, event_header: str) -> bool:
        """ Parse through event_header substring and assign the values to event using lookup_list

        :param event_header: substring of the example set file representing an event header
        :type event_header: str
        :return:
        """
        delimiters = re.findall(r'[^:]\s', event_header)
        event_header_list = re.split(r'[^:]\s', event_header)
        for i in range(len(delimiters)):
            event_header_list[i] += delimiters[i].strip()

        lookup_list = ["pre_proc:", "post_proc", "min:", "max:", "grace:", "defI:", "defT:", "actT:", "actI:"]
        for lookup_string in lookup_list:
            for element in event_header_list:
                if lookup_string in element:
                    colon_index = element.find(":")
                    if " " in element:
                        value = element[colon_index + 2:].strip()
                    else:
                        value = element[colon_index + 1:].strip()
                    if self.assign_field_values(lookup_string, value) is False:
                        return False
                    event_header_list.remove(element)
                    break
        return True

    def parse_event_list(self, event_list: str, sparse_mode: bool):
        """ Parse through event_list and populates attributes in event Event object
        in the same way as LENS. Return if an error is found.
        :param event_list: a substring of the .ex file containing information about the event
        :type event_list: str
        :param sparse_mode: true if reading sparse mode file
        :type sparse_mode: bool
        :return: false if an error is found
        "rtype: optional, false
        """
        for group in self.example.set.input_group:
            self.input_group_len.append(group.num_units)
            self.input_group_name.append(group.name)
        for group in self.example.set.target_group:
            self.target_group_len.append(group.num_units)
            self.target_group_name.append(group.name)
        event_string = event_list.strip()

        if sparse_mode:
            if not self.parse_sparse_format(event_string):
                return False

        else:
            if not self.parse_dense_format(event_string):
                return False
        return True
    def parse_dense_format(self, event_string: str) -> bool:
        """ Assuming dense format for event_string, reads the string and sets values for
        the unit groups of this event accordingly. Return true if successful, else false.

        :param event_string: string representation of an event in dense format
        :type event_string: str
        :return: bool
        """
        inp_tar_lst = re.split("[ITB]:", event_string)
        inp_tar_lst.pop(0)
        # separates by letter (dense only) and removes the first value
        # because it's the description and does not contain data
        # event_dict is a set of key-value pairs with letter keys and list of numbers value
        event_dict = {}
        i = 0
        for unit_type in ["I", "T", "B"]:
            if unit_type in event_string:
                event_dict[unit_type] = inp_tar_lst[i]
                i += 1
        for unit_type in event_dict:
            # event_dict is the dictionary where keys are associated with their list of values
            if re.search(r'{(.*?)}[^*]', event_dict[unit_type]) is not None:
                if unit_type in ("I", "B"):
                    res = self.add_specific_unit_group(True, event_dict[unit_type], self.input_group_name,
                                                       self.input_group_len)
                    if res is False:
                        return False

                    for name in res:
                        index = self.input_group_name.index(name)
                        self.input_group_name.pop(index)
                        self.input_group_len.pop(index)

                    if unit_type == "I":
                        event_dict["I"] = []

                if unit_type in ("T", "B"):
                    res = self.add_specific_unit_group(False, event_dict[unit_type], self.target_group_name,
                                                       self.target_group_len)
                    if res is False:
                        return False
                    else:
                        for name in res:
                            index = self.target_group_name.index(name)
                            self.target_group_name.pop(index)
                            self.target_group_len.pop(index)
                    event_dict[unit_type] = []

            else:
                if unit_type in ("I", "B"):
                    if type(event_dict[unit_type]) is str:
                        event_dict[unit_type] = event_dict[unit_type].split()
                    if self.add_unit_groups(True, self.input_group_len, event_dict[unit_type],
                                            self.input_group_name) is False:
                        return False

                if unit_type in ("T", "B"):
                    if type(event_dict[unit_type]) is str:
                        event_dict[unit_type] = event_dict[unit_type].split()
                    if self.add_unit_groups(False, self.target_group_len, event_dict[unit_type],
                                            self.target_group_name) is False:
                        return False

        if "B" not in event_dict:
            if "I" not in event_dict:
                if self.add_unit_groups(True, self.input_group_len, [], self.input_group_name) is False:
                    return False
            if "T" not in event_dict:
                if self.add_unit_groups(False, self.target_group_len, [], self.target_group_name) is False:
                    return False
        return True

    def parse_sparse_format(self, event_string: str) -> bool:
        """ Assuming dense format for event_string, reads the string and sets values for
        the unit groups of this event accordingly. Return true if successful, else false.

        :param event_string: string representation of an event in sparse format
        :type event_string: str
        :return: bool
        """
        if event_string == "":
            if self.add_unit_groups(True, self.input_group_len, [], self.input_group_name) is False:
                return False
            if self.add_unit_groups(False, self.target_group_len, [], self.target_group_name) is False:
                return False
        else:
            event_dict = {}
            inp_tar_lst = re.split("[itb]:", event_string)
            inp_tar_lst.pop(0)
            i = 0
            for unit_type in ["i", "t", "b"]:
                if unit_type in event_string:
                    event_dict[unit_type] = inp_tar_lst[i]
                    i += 1
            for unit_type in event_dict:
                if re.search(r'{(.*?)}', event_dict[unit_type]) is not None:
                    external_inputs = re.findall(r'{(.*?)}', event_dict[unit_type])
                    unit_indexes = re.split(r'{(.*?)}', event_dict[unit_type])
                    unit_indexes.pop(0)
                    for inpt in external_inputs:
                        while inpt in unit_indexes:
                            unit_indexes.remove(inpt)
                    if unit_type in ("i", "b"):
                        unit_lst = [str(self.default_input) for _ in range(sum(self.input_group_len))]
                        for i, value in enumerate(external_inputs):
                            if self.get_sparse_units_list(True, unit_lst, unit_indexes[i],
                                                          external_inputs[i]) is False:
                                return False
                        if self.add_unit_groups(True, self.input_group_len, unit_lst,
                                                self.input_group_name) is False:
                            return False
                    elif unit_type in ("t", "b"):
                        unit_lst = [str(self.default_target) for _ in range(sum(self.target_group_len))]
                        for i, value in enumerate(external_inputs):
                            if self.get_sparse_units_list(False, unit_lst, unit_indexes[i],
                                                          external_inputs[i]) is False:
                                return False
                        if self.add_unit_groups(False, self.target_group_len, unit_lst,
                                                self.target_group_name) is False:
                            return False
                else:
                    if unit_type in ("i", "b"):
                        unit_lst = [str(self.default_input) for _ in range(sum(self.input_group_len))]
                        if self.get_sparse_units_list(True, unit_lst, event_dict[unit_type]) is False:
                            return False
                        if self.add_unit_groups(True, self.input_group_len, unit_lst,
                                                self.input_group_name) is False:
                            return False

                    if unit_type in ("t", "b"):
                        unit_lst = [str(self.default_target) for _ in range(sum(self.target_group_len))]
                        if self.get_sparse_units_list(False, unit_lst, event_dict[unit_type]) is False:
                            return False
                        if self.add_unit_groups(False, self.target_group_len, unit_lst,
                                                self.target_group_name) is False:
                            return False
            if "b" not in event_dict:
                if "i" not in event_dict:
                    if self.add_unit_groups(True, self.input_group_len, [],
                                            self.input_group_name) is False:
                        return False
                if "t" not in event_dict:
                    if self.add_unit_groups(False, self.target_group_len, [],
                                            self.target_group_name) is False:
                        return False
        return True

    def get_sparse_units_list(self, doing_inputs: bool, units: List[int],
                              unit_indexes: str, external_input=None) -> bool:
        """ Fills units argument with list of spare units using the external_input. Affects inputs if
        doing_inputs is true otherwise affects targets. If no externa_input is given, active ones will be used.

        :param doing_inputs: true if add units to input group
        :type doing_inputs: bool
        :param units: list of units to add
        :param unit_indexes: string of the list of indexes for the event
        :param external_input:
        :return:
        """
        if external_input is None:
            if doing_inputs:
                external_input = self.active_input
            else:
                external_input = self.active_target
        index_range = unit_indexes.split()
        for index in index_range:
            reg = re.compile("[0-9]+-[0-9]+")
            if reg.match(index) is not None:
                hyphen = index.find("-")
                start = int(index[:hyphen])
                end = int(index[hyphen + 1:])
                if start >= end:
                    return self.example.set.parse_error(
                        "wrong index range passed in sparse formatting at event layer " + str(
                            self.example.event.index(self)) + " of example " + str(
                            self.example.set.example.index(self.example)))
                for i in range(start, end + 1):
                    units[i] = str(external_input)
            elif index == "*":
                if doing_inputs:
                    for i in range(len(units)):
                        units[i] = str(self.active_input)
                else:
                    for i in range(len(units)):
                        units[i] = str(self.active_target)
                return True
            elif index.isdigit():

                units[int(index)] = str(external_input)
            else:
                return self.example.set.parse_error(
                    "incorrect type of value passed in sparse formatting at event layer "
                    + str(self.example.event.index(self)) + " of example "
                    + str(self.example.set.example.index(self.example)))
        return True

    def add_specific_unit_group(self, doing_inputs: bool, unit_lst: str, group_names: List[str], group_lens: List[int]):
        """ Add the unit groups which are specified in unit_lst, group_names and group_lens.
        Each new index in these lists is information corresponding to a unit group.

        :param doing_inputs: 
        :param unit_lst: list of units in group
        :param group_names: list of names in group
        :param group_lens: list of lengths groups in event
        :return:
        """
        unit_names = re.findall(r'{(.*?)}[^*]', unit_lst)
        unit_values = re.split(r'{(.*?)}[^*]', unit_lst)
        unit_values.pop(0)

        for name in unit_names:
            while name in unit_values:
                unit_values.remove(name)
        given_group_len = []
        units = []
        for i, value in enumerate(unit_names):
            units.extend(unit_values[i].split())
            group_length = group_lens[group_names.index(unit_names[i])]
            given_group_len.append(group_length)

        if self.add_unit_groups(doing_inputs, given_group_len, units, unit_names) is False:
            return False
        return unit_names

    def add_unit_groups(self, doing_inputs: bool, group_len: List[int],
                        units: List[str], unit_names: List[str]) -> bool:
        """ Parses through information in group_len, units and unitNames
        to create new unit groups accordingly, and add them to this event

        :param doing_inputs:
        :param group_len: list of lengths of groups being processed
        :param units: list of units in the unit group being processed
        :param unit_names: list of names of the units being processed

        :return:
        """
        counter = 0
        group_counter = 0
        reg = re.compile(r"{(.+)}\*")
        if len(units) == 1 and reg.match(units[0]):
            open_brac = units[0].find("{")
            close_brac = units[0].find("}")
            value = units[0][open_brac + 1: close_brac]
            units = [value for _ in range(sum(group_len))]
        while counter < len(units) and group_counter < len(group_len):
            unit_group = UnitGroup(self, group_len[group_counter], unit_names[group_counter])
            for _ in range(group_len[group_counter]):
                if counter < len(units):
                    if unit_group.add_units(doing_inputs, units[counter]) is False:
                        return self.example.set.parse_error("Invalid type if unit passed at Event layer " + str(
                            self.example.event.index(self)) + " of example " + str(
                            self.example.set.example.index(self.example)))
                    counter += 1
                else:
                    break
            if unit_group.check_units_size(doing_inputs) is False:
                return self.example.set.parse_error(
                    "Too many input units in event " + str(
                        self.example.event.index(self)) + " of example " + str(
                        self.example.set.example.index(self.example)))
            group_counter += 1
        if group_counter < len(group_len):
            while group_counter < len(group_len):
                unit_group = UnitGroup(self, group_len[group_counter], unit_names[group_counter])
                if unit_group.check_units_size(doing_inputs) is False:
                    return self.example.set.parse_error("Too many target units" + str(
                        self.example.event.index(self)) + " of example " + str(
                        self.example.set.example.index(self.example)))
                group_counter += 1
        return True

    def write_event(self) -> str:
        """ Return dense format string representation of this event
        """
        event_string = ""
        event_string += self.write_event_header()
        input_list = "I: "
        for i_group in self.input_group:
            for value in np.nditer(i_group, order='K'):
                if value == float("NaN"):
                    input_list += "- "
                else:
                    input_list += str(int(value)) + " "
        target = "T: "
        for t_group in self.target_group:
            for value in np.nditer(t_group, order='K'):
                if value == float("NaN"):
                    target += "- "
                else:
                    target += str(int(value)) + " "
        event_string += input_list
        event_string += target
        event_string += "\n"
        return event_string

    def write_event_header(self) -> str:
        """
        write the event header to the file
        """
        event_header_string = "[ "
        if self.pre_proc_name is None:
            event_header_string += "pre_proc: " + "- "
        else:
            event_header_string += "pre_proc: " + self.pre_proc_name + " "

        if self.post_proc_name is None:
            event_header_string += "post_proc: " + "- "
        else:
            event_header_string += "post_proc: " + self.post_proc_name + " "

        if self.min_time is None:
            event_header_string += "min: " + "- "
        else:
            event_header_string += "min: " + str(self.min_time) + " "

        if self.max_time is None:
            event_header_string += "max: " + "- "
        else:
            event_header_string += "max: " + str(self.max_time) + " "

        if self.grace_time is None:
            event_header_string += "grace: " + "- "
        else:
            event_header_string += "grace: " + str(self.grace_time) + " "

        if self.default_input is None:
            event_header_string += "defI: " + "- "
        else:
            event_header_string += "defI: " + str(self.default_input) + " "
        if self.default_target is None:
            event_header_string += "defT: " + "- "
        else:
            event_header_string += "defT: " + str(self.default_target) + " "
        if self.active_input is None:
            event_header_string += "actI: " + "- "
        else:
            event_header_string += "actI: " + str(self.active_input) + " "
        if self.active_target is None:
            event_header_string += "actT: " + "- "
        else:
            event_header_string += "actT: " + str(self.active_target)
        event_header_string += "]" + "\n"
        return event_header_string

    def print_out_event(self, printing=True, tabs=0) -> str:
        """ Prints out the instance variables of an Event and of its input and target groups
        This function is a work in progress for testing purposes.
        Each new layer of composition is indicated by indent.

        If calling this function directly, please leave
        printing=True and tabs=0 to their default values.

        :param tabs: int
        :param printing: bool
        """
        variables_list = [("Obj", "Event"), ("example.name", self.example.name), ("post_proc", self.post_proc_name),
                          ("pre_proc", self.pre_proc_name), ("maxTime", self.max_time),
                          ("minTime", self.min_time), ("graceTime", self.grace_time)]
        variables_list.extend([("defI", self.default_input), ("actI", self.active_input), ("defT", self.default_target),
                               ("actT", self.active_target)])

        event_string = ""
        event_string += format_object_line(variables_list, tabs)
        for input_num, value in enumerate(self.input_group):
            input_g = self.input_group[input_num]
            variables_list = [("input group", input_g)]
            event_string += format_object_line(variables_list, tabs + 1)
        for target_num, value in enumerate(self.target_group):
            target = self.target_group[target_num]
            variables_list = [("target group", target)]
            event_string += format_object_line(variables_list, tabs + 1)
        if printing:
            print(event_string)
        return event_string

    def print_out(self):
        """Print out this event"""
        self.print_out_event()

    def assign_field_values(self, lookup_string: str, value: str) -> bool:
        """ Set the field value of type proc, min, max, grace, defI, defT, actI, actT to their
        respective instance attributes in Event S; if Event is None then assign to ExampleSet S

        :param lookup_string: could be proc, min, max, grace, defI, defT, actI, actT
        :type lookup_string: str
        :type value: str
        :return: Optional error
        """
        value_format = re.compile("^[0-9]+([,.][0-9]+)?$")
        if lookup_string == "pre_proc:":
            self.pre_proc_name = value.strip()

        elif lookup_string == "post_proc:":
            self.post_proc_name = value.strip()

        elif lookup_string == "min:":
            if value_format.match(value):
                if self.min_time <= float(value) <= self.max_time:
                    self.min_time = float(value)
                # remove the last piece, which is space or "]"
            elif value == "-":
                self.min_time = None
            else:
                return self.example.set.parse_error(
                    "missing value after \"min:\" in header of event {0} of example {1}".format(str(
                        self.example.event.index(self)), str(self.example.set.example.index(self.example))))
        elif lookup_string == "max:":
            if value_format.match(value):
                if self.max_time >= float(value) >= self.min_time:
                    self.max_time = float(value)
            elif value == "-":
                self.max_time = None
            else:
                return self.example.set.parse_error(
                    "missing value after \"max:\" in header of event {0} of example {1}".format(str(
                        self.example.event.index(self)), str(self.example.set.example.index(self.example))))
        elif lookup_string == "grace:":
            if value_format.match(value):
                if float(value) <= self.grace_time:
                    self.grace_time = float(value)
            elif value == "-":
                self.grace_time = None
            else:
                return self.example.set.parse_error("missing value after \"grace:\" in header of event " + str(
                    self.example.event.index(self)) + " of example " + str(
                    self.example.set.example.index(self.example)))
        elif lookup_string == "defI:":
            if value_format.match(value):
                self.default_input = float(value)
            elif value == "-":
                self.default_input = None
            else:
                return self.example.set.parse_error("missing value after \"defI:\" in header of event " + str(
                    self.example.event.index(self)) + " of example " + str(
                    self.example.set.example.index(self.example)))
        elif lookup_string == "defT:":
            if value_format.match(value):
                self.default_target = float(value)
            elif value == "-":
                self.default_target = None
            else:
                return self.example.set.parse_error("missing value after \"defT:\" in header of event " + str(
                    self.example.event.index(self)) + " of example " + str(
                    self.example.set.example.index(self.example)))
        elif lookup_string == "actI:":
            if value_format.match(value):
                self.active_input = float(value)
            elif value == "-":
                self.active_input = None
            else:
                return self.example.set.parse_error("missing value after \"actI:\" in header of event " + str(
                    self.example.event.index(self)) + " of example " + str(
                    self.example.set.example.index(self.example)))
        elif lookup_string == "actT:":
            if value_format.match(value):
                self.active_target = float(value)
            elif value == "-":
                self.active_target = None
            else:
                return self.example.set.parse_error("missing value after \"actT:\" in header of event " + str(
                    self.example.event.index(self)) + " of example " + str(
                    self.example.set.example.index(self.example)))
        return True


# These are helper functions for the printing functions.
def tab(char=1):
    """
    returns string of n spaces
    :param char:
    :return:
    """
    return "     " * char


def format_object_line(variables_list, num_tabs=0):
    """
    Gets string print out of all the instance variables of this Event
    :param variables_list:
    :param num_tabs:
    :return:
    """
    s_output = tab(num_tabs)
    for item in variables_list:
        s_output += item[0] + " = " + str(item[1]) + ", "
    return s_output + "\n"
