Pre-Commit
==========

Pre-Commit is a library that can do checked prior to committing.

You can implement exiting pre-commit hooks and you can also create your custom hooks.

Installation
-------------------

The two commands that are needed:

::
    pip3 install pre-commit

This command uses pip3 to install the pre-commit library

::
    pre-commit install

This command installs the git hook scripts

In PyLens, these two command lines are added into the requirements.txt and MakeFile.

To install pre-commit in PyLens, simply run 'make' or 'make requirements' and 'make pre-commit'

Creating Pre-Commit Configurations
-----------------------------------

Currently, the pre-Commit Configurations are within the file ".pre-commit-config.yaml"

The "repos:" on line 1 allows us to add multiple git hook repositories to include in our test.

    -   repo: local

Allows us to run test scripts within the local repository.

Then, we will add hooks inside of "hooks:"

Each hook contains a name and ID, which will be displayed on command line.

    entry:

is the command that you will be running for the test.

    venv/bin/python3 pre-commit_hooks/xor_hook.py

The line above runs python3 in the virtual environment, and it runs the xor_hook.py in pre-commit_hooks directory.

There are other keys that you can add to the hooks.

verbose - will print output even the test passes

always_run - will always run this hook even if there are no matching files

More keys are listed here:
https://pre-commit.com/#plugins

Creating Unit Tests
----------------------

It is fairly simple to create unit test.

You will only need to create a script that exits with code 0 if test passes and code 1 if test fails.


