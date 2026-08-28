Example Files
========================

Text Format
------------------

Example files are written in text format. The text format is rather complex, yet flexible. It is customary to give text example files the extension ".ex". When saving files, you might as well always add .gz or .bz2 to the name so the file will automatically be compressed. Decompression adds negligible time to the loading process. When loading, leave off the .gz or .bz2 and the file will be loaded correctly whether or not it is compressed.

Before reading further, you may want to familiarize yourself with concepts of time and example events in Lens, although you may not need to concern yourself too much with this if you are just writing simple example files for feed-forward networks with one set of inputs and one set of targets per example. If the training set selects examples in ORDERED mode (which means that examples are presented to the network during training in the order in which they were read from the example file), having more than one event per example or making each event a different example may be basically equivalent for most networks. However, multiple events per example should be used if that is the logical organization of the example set. Additionally, if you are using any example selection strategy other than ORDERED, you must use multiple events per example if those events need to occur in sequence.

The text example file format was substantially altered for PyLens, although most earlier formats can still be loaded. Note that it is no longer necessary to specify the number of input and output units in the header. Those values will be ignored if they do appear. In fact, it is no longer necessary to even have a set header.

The file format is shown below. As in C code, whitespace only matters when leaving it out would cause ambiguity or cause two numbers or strings to run together. Extra whitespace generally does not matter (although it is a good idea to follow each example with a line break). Any line beginning with a hash mark ("#") will be treated as a comment and ignored. In the format specification, elements in angled brackets denote values of specific types. <I > denotes an integer, <R > denotes a real number, <B > denotes a boolean (0 or 1) value, and <S > denotes a string. The string should be surrounded by quotes, brackets, parentheses, or curly braces if it contains any whitespace. A dash ("-") can be used in place of a real number to indicate a NaN, which is often used for parameter values or to inactivate inputs or targets.

All fields are optional unless they are in red. Words and punctuation in bold should appear in the file as written. Text in blue is just explanatory and does not appear. If two or more things are in blue square brackets separated by pipes ("|"), then one or the other must appear. If something is in parentheses, it can be repeated 0 or more times. If two or more things are in parentheses and are separated by pipes ("|"), then you can repeatedly choose from among the alternatives. For example, (a | b | c) would match a sequence of zero or more characters, where each character is either a, b, or c. There can't be any space between a field identifier and the colon.

::

    proc:  <S set-proc>

    max:   <R set-maxTime>

    min:   <R set-minTime>

    grace: <R set-graceTime>

    defI:  <R set-defaultInput>

    actI:  <R set-activeInput>

    defT:  <R set-defaultTarget>
    actT:  <R set-activeTarget>
    ;

    for each example:
      name:   <S example-name>

      proc:   <S example-proc>

      freq:   <R example-frequency>

      <I example-numEvents>   #this can be left out if it is 1

      for each list of events:
        [(<I event> | <I event>-<I event> | \*)
            proc:  <S event-proc>

            max:   <R event-maxTime>

            min:   <R event-minTime>

            grace: <R event-graceTime>

            defI:  <R event-defaultInput>

            actI:  <R event-activeInput>

            defT:  <R event-defaultTarget>

            actT:  <R event-activeTarget>

        ]

        (I:\|i:\|T:\|t:\|B:\|b:\|) (
          dense range:  (<S group-name> <I first-unit>) (<R input-value>) |
          sparse range: {<S group-name> <R input-value>} [* | (<I unit> | <I unit>-<I unit>)]

        )

    ;

Example Set Header
------------------

The example set header consists of a number of optional fields, which may appear in any order. The proc: is a procedure that is executed before the rest of the set is loaded. Typically, this will define some procedures (or source a script file defining procedures) that will be used by procs belonging to examples or events in the set.

The header may also specify the maxTime, minTime, graceTime, defaultInput, activeInput, defaultTarget, and activeTarget fields of the example set. The ...Time values are used for any event that does not specify its own non-NaN values. The defaultInput and defaultTarget values are used for any unit that does not have a value specified on a particular event. The activeInput and activeTarget are the default values used when a unit is named in a sparse representation, which is explained below.

The set header need not end with a semicolon. In fact, there could be no set header at all, which will typically be the case. However, a semicolon can be used to avoid ambiguity. For example, if the first example has a proc: specification at the start of its header, you must do:

::

    ;

    proc: ...

    rather than:

    proc: ...

Otherwise, the proc: will be interpreted as an example set procedure rather than the first example's procedure.


Procs
-----

A proc is a snippet of code that is executed before the rest of the object is loaded. Procs can also be manually called anytime using the .proc() function; this executes the snippet of code associated with the object. Procs may be defined for each example set, example or event. All procs are associated with field values and cannot have duplicate names. The procs field values are stored in a separate file, usually called “<nameofexampleset>_procs.ex”. It has its own special format (below).

Format of proc files:
A proc file may or may not begin with global variables defined to be used by any of the functions. All global variables must be defined in the procs file before the first proc. At the example sets level, global variables and their values are stored in the instance variable global_dict, which is a dictionary of key-value pairs. They can be accessed anytime from the example set. For example, if “a=0” at the beginning of the proc file defines a global variable “a”, it can be accessed as follows: ExampleSet.global_dict[“a”].

The global variables are followed by procs with their special format. All procs start with a special key word  “startofproc” followed by the name of the proc. Then, in a new line lies an executable code snippet enclosed in curly brackets “{code}”. For each function defined in the proc, the function needs to pass in the global variables as parameters if they are being used in the function body. You can reference the following example of a procs file.

To define a proc called testing_proc at example_set, we may have the following text inside at the example_set header inside the example_set.ex file:

proc: example_set_0;

And inside the example_set_procs.ex file, we may have:

::

    a = 0 # global variables

    b = 9  # global variables

    startofproc example_set_0 {

    def funct(self.network,a, b):

    print(a*b)

    }

    startofproc example_0 {

    def ex_func(self.network, a, b):

    print(self.name)

    print('testing :)) this is an example!!')

    }


Example Set Mode
----------------

The mode is a parameter which is passed in the constructor of the example Set to set the order of the examples in the example_list of example_set. There are 5 different modes:
In ORDERED mode, which is the default, examples will be presented in the order in which they were found in the example file.

In RANDOMIZED mode, examples will be selected at random with replacement, each having the same probability of selection. Note that this differs from PERMUTED because it uses replacement. It differs from PROBABILISTIC because it ignores the example frequency.

In PERMUTED mode, examples will be selected at random without replacement, each having the same probability of selection. A different order will be computed for each pass through the set.

In PROBABILISTIC mode, examples are selected based on their given frequency. Specified frequency values will be normalized over all examples and this distribution used for selection. If example sets are concatenated, the distribution will be recalculated based on the specified frequencies. An example with no frequency specified is given a value of 1.0.

#TODO custom mode

CUSTOM mode allows you to write a procedure that generates the index of the next example. When it's time to choose the next example, the example set's chooseExample procedure will be called. This should return an integer between 0 and one less than the number of examples, inclusive.

Example Iterator
----------------

Example Iiterator is a class implemented to iterate over the examples in  the examples_list of the exampleSet object. The Iterator keeps track of the current example over which the iterator is currently, the previous example, and the next example. Calling iterator_example() method of the exampleSet object allows us to iterate over the example while calling next_example(), prev_example() and curr_example() allows us to access the current example without moving the iterator. If the iterator reaches the last example in the list where the list has the examples arranged in the order mentioned by the mode instance variable, the iterator resets itself by first reordering the example_list based on the mode and then resetting the value of its next, curr and prev parameters so that the curr points to the first example in the ordered example_list.

Example Header
--------------

Each example begins with a header consisting of a few optional specifications, which can appear in any order. The most important one is the number of events in the example. If this does not appear, it is assumed to be 1. Otherwise, it is mandatory.

name: defines the label that will be used for the example. If no name is given, the example will be labeled by its index, starting with 0. Remember that strings containing whitespace should be surrounded by curly braces.

proc: defines a command that will be run when the example is first loaded. You might use this to cause the network to do something special before this example.

freq: sets the frequency of example presentation that will be used when selecting examples in PROBABILISTIC mode or when using pseudo-example frequencies, which simply scale the error and output derivatives by the example frequency. The default frequency is 1.0. Frequencies need not be normalized to a sum of 1.0. That will be done automatically if PROBABILISTIC mode is used. You can turn on pseudo-frequencies by setting the network's pseudoExampleFreq field to 1.

Event Lists

Following the example header are the event specifications. Because they are quite flexible, these can be a bit confusing. There are three main sections that may appear in an event specification: event lists, inputs, and targets.

The event list is used to set parameters for the events and activates certain events so that the next inputs and targets will be applied to those events. Because multiple events can appear in the event list, it is possible to cause events to share input or target representations without duplicating the representation in the example file or in memory.

The event list is enclosed in square brackets. The list is either a series of event numbers and ranges of event numbers or an asterisk, which stands for all events. There should just be whitespace between elements of the list. If the list is empty, it is equivalent to having an asterisk. Following the event list can be a number of optional parameter settings, which can appear in any order. These settings will only affect the events in the list. For example, this sets the maxTime and minTime of events 0, 3, 4, 5, 6, and 9:

::

    [0 3-6 9 max:3.5 min:1]

Either of these set the default sparse inputs for all events to -1.0:

::

    [* defI:-1]
    [defI:-1]

The first set of inputs and the first set of targets following an event list will be applied to those events. You may not specify more than one set of inputs or more than one set of targets for an event. If a set of inputs is specified that is not immediately following an event list, it will be applied to the event following the highest numbered event that already has inputs. A similar rule applies for targets. Consider this example:

::

    6

    [0-2 4]

    I: 0 1 0

    I: 1 0 1

    T: 1 0

    ;

The inputs 0 1 0 will be applied to events 0, 1, 2, and 4, because these inputs immediately follow the event list. The next inputs, 1 0 1, cannot also be applied to those events so they will be applied to event 5. The targets, however, will apply to the active events, 0-2 and 4, because targets have not yet been specified for those events. In this example, event 3 won't have any inputs or targets and event 5 won't have targets.

You can give inputs and targets to each event in order by simply listing them either in blocks or intermingled:

::

    I: 1 0 0

    I: 0 1 0

    I: 0 0 1

    T: 0 1

    T: 1 0

    T: 1 1

    or:

    I: 1 0 0

    T: 0 1

    I: 0 1 0

    T: 1 0

    I: 0 0 1

    T: 1 1

Note, though, that if you have activated a list of events to set some parameters, you will need to active event 0 before you can list the inputs and targets for each event in order. Otherwise, the first ones will apply to the active events and the others may cause errors. For example, you may need to do this:

::

    [max: 3]

    [0]

    I: 1 0 0

    I: 0 1 0

    T: 0 1

    T: 1 1

Inputs and Targets
------------------------

A set of inputs is introduced by either I: or i:. Following this is a series of one or more ranges. A range can either use a dense encoding, which specifies values for a consecutive block of units, or a sparse encoding, which specifies a single value and then a list of unit numbers. When an event is loaded into the network, first the externalInput of each INPUT unit is set to the event's defaultInput and the target of each OUTPUT unit is set to the event's defaultTarget. Then each of the ranges is applied to change the inputs or targets for some or all of the units. Typically the default values will be 0.0, although the user may wish to set them to NaN (-). A unit with a NaN target will be ignored when error is computed. A HARD_CLAMP unit with a NaN externalInput will attempt to calculate its input using the standard input combining and transfer methods. If the unit has no input combining or transfer method or no inputs and the externalInput is NaN, it will probably have an output of 0.

Dense Encoding
--------------

The dense encoding is preceded by an optional set of {}. Within the curly brackets may be an optional group name and/or an optional first unit index, in either order. If the group name is specified, only units in that group will be affected and the unit indices will be relative to the start of that group. Note that any network that uses the example must have a group with that exact name. If the group name must contain whitespace, it may be surrounded by quotation marks, brackets or parentheses. If a group is not specified, then unit indices will be relative to the entire network, counting from the first unit in the first INPUT group to the last unit in the last INPUT group. The first unit index is the index (relative to either the group or the network) of the first unit for which an externalInput value will be specified.

Following the curly brackets is a list of real values, which will be the inputs to the consecutive block of input units including and following the starting unit. There is no need to specify in advance how many values will be in this list and there is no need to terminate it with any punctuation (unless it is the last thing in the example). This defaults to 0. For example:

::

    I: {input2 3} 0.1 0.2 0.3 {2} 0.4

will set the externalInputs to units input2:3, input2:4 and input2:5 to 0.1, 0.2, and 0.3, respectively. The next range then sets the externalInput to the third input unit in the network to 0.4. If you do not wish to put a group name or first unit in the parentheses, you may leave them off if they are immediately following an I:. For example, these two lines are equivalent:

::


    I: {} 2 3

    I: 2 3

The dense encoding for targets is just like that for inputs, except that a T: introduces it. If a B: is used, it means both inputs and targets. It is equivalent to using I: and then repeating the whole thing again with a T:. The B: can be very useful for auto-encoder and prediction tasks. It will save space both in the example file and in memory when the set is loaded.

Sparse Encoding
---------------

The sparse range is introduced by a set of curly braces, {}. Within the curly braces may be an optional group name, as for the dense encoding, and/or an externalInput value. If the input value is not specified it will default to the event's activeInput. However, if two or more events are sharing inputs (because they were listed in an event list), the activeInput value of the first event will apply to all of them, regardless of the other events' activeInput values.

Following the curly braces is either a * or a list of unit numbers and unit ranges. The * will cause all units in the group (or in the network if a group is not given) to have the specified externalInput value. Otherwise, only those units whose indices are listed will receive the input. This example will set the externalInput of units 0, 4, 5, and 6 to 1.0, and the externalInput for units 1, 2 and 3 to -1.0:

::


    I: {1.0} 0 2 4-6 {-1.0} 1-3

If you do not wish to put a group name or externalInput value in the curly braces for the first range, the braces may be eliminated and i: used rather than I:. These are equivalent:

::


    I: {} 0 3

    i: {} 0 3

    i: 0 3

One more example:

::


    [defI:- actI:1]

    i: 0-3 5 8 {2.0} 4 9-11

The above case has two ranges. Units 0, 1, 2, 3, 5, and 8 will have externalInput 1.0. Units 4, 9, 10, and 11 will have externalInput 2.0, and units 6, 7, and anything above 12 will have the default input, which is NaN. Therefore, those units will compute their own inputs and targets using any input combining and transfer function they may have.

Sparse targets are similar to sparse inputs, except T: or t: are used. Here is one way to make sure an event has no targets:

::


    T:{-}*

A b: indicates both inputs and targets. It is equivalent to using i: and then repeating the whole thing again with a t:.

Examples
--------

Here is a very simple example file for an XOR problem:

::


    I:0 0 T:0;

    I:0 1 T:1;

    I:1 0 T:1;

    I:1 1 T:0;

Here is an equivalent file that is ever so slightly smaller because it uses sparse inputs:


::


    ;;

    i:1 t:0;

    i:0 t:0;

    i:\*;

This is a poorly done auto-encoder example file:

::


    I:1 0 0 0

    T:1 0 0 0;

    I:0 1 0 0

    T:0 1 0 0;

    I:0 0 1 0

    T:0 0 1 0;

    I:0 0 0 1

    T:0 0 0 1;

Here it is improved using sparse encodings:

::


    i:0 t:0;

    i:1 t:1;

    i:2 t:2;

    i:3 t:3;

It can be compressed further using the "both" symbol:

::


    b:0; b:1; b:2; b:3;

Here is an XOR example file with many bells and whistles:

::

    # Here is the first example.  It has two events:

    name:{0 0}

    freq:2.7

    proc: {example_proc0}

    2

    [max:2 min: 1]

    [0] I: 0 0

    [max:2.5 proc:{puts "starting the second event"}]

    # Only specifying inputs for the first event and targets for the second

    [1] T: 0;

    # Here is the second example.  It has one event:

    freq: 4.5

    name: "0 1"

    proc:{example_proc2}

    # This means all (one) events have maxTime of 3.5:

    [max:3.5]

    I:1 Tt:1;

    # Here is the third example.  It has two events with the default headers:

    name:1-0

    2

    # Both events use inputs "1 0", but the first has no targets.

    [0] i: 1 0

    [1] t:\*;


ExampleSet Object Structure
___________________________

ExampleSet - Example - Event

ExampleSet is the top level of the training data. In its initialization (see examples below), ExampleSet takes in a Network object, a filename
and other parameters.

The ExampleSet.example attribute is a list of Example Objects that are in the ExampleSet.

The Example.event attribute is a list of Event Objects that are in the Example.

Loading ExampleSets
___________________

When the load_example_set() function is called, the following procedure is used in the back end to read the example set file into an ExampleSet object:

1. filter out commented lines (lines starting with #).

2. check whether the file is using sparse or dense format, and remember the setting.

3. split the file content by example, which is separated by '\n'. then for each example do the following:

4. create Example object. parse the example header, looking for keywords ending with ':', and setting their respective values.

5. parse event header, looking for keywords ending with ':', and setting their respective values.

6. set the values for input, output and target groups according to the format above.

Writing Example Files
---------------------

PyLens includes an ExampleSet level function: ExampleSet.write_example_set_to_file(file_name). This function writes the dense format text representation of this example set to file_name. At any point after a series of manipulations to the initial example set, you can write the example set to a file to save its current state.

You can also manually construct your own example set and generate a text representation of it.

Usage example:

::

    # creates an example set

    set = ExampleSet(network, proc, name, file_name, 1, 1, 1, 1, 0, 0)

    ex = Example(set)

    set.example.append(ex)

    ev = Event(ex)

    ex.event.append(ev)

    # populate the values of event as necessary

    ev.input_group = []

    ev.target_group = []

    # write out the file

    set.write_example_set_to_file(“testing.txt”)

