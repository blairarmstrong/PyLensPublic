Github Action Unit Tests
=========================

This is the CI/CD Tool that PYLens uses on its repository.

Any unit tests that a developer wishes to perform on each push and pull request will follow these steps

Github action performs whenever there is a "push" or "pull request" event on the master branch. Other branches will not perform the tests.

Installation
-------------

You should not need to install anything locally.

All scripts are performed on Github.

Creating Unit Tests
---------------------

**Set Up**

There is one crucial step before pushing your unit tests onto github.

You will need to add all required libraries that needs to be installed in "requirements.txt"

You can either add the libraries manually or run

::
    pip freeze > requirements.txt

However, make sure you are running in a virtual environment when running the command above, or else pip adds all libraries installed in your local machine to requirements.txt

Also, please make sure non of the existing libraries are deleted when you run the command above.

**Unit Test file**

You can place the unit test anywhere in the repository. However, it would be best to place it in "pre-commit_hooks" directory as all current unit tests are within that directory.

You can run any sort of tests in the unit test file and create any helper functions.

The tests runs like a regular python file execution; it will run "main" first.

To indicate the test has passed, simple exit with exit code 0 or let the program terminate without any error raised.

To indicate the test has failed, simply raise an error or exit with an exit code that is non-zero.

A simple test would be:

::
    if __name__ == "__main__":
        a = 1
        b = 1
        assert a == b

The program above passes the test if the assertion is true and fails if the assertion fails.

**Adding unit test as an action**

After you have created an unit test file, you will need to add it to the "unit_test.yml" file.

This indicates to Github that this is also a file to run.

Add your file by simply following previous examples and adding a "-name" attribute and "run" command.

You do not need to add any building scripts as previous commands already creates an environment and installs all dependencies using pip.

For example, if I want to add a file called "unit_test_one.py" that has the path "pre-commit_hooks/unit_test_one.py"

I would add the following to "unit_tests.yml"

\- name: Example One

run: python3 pre-commit_hooks/unit_test_one.py

Please make sure the indentation matches previous lines.

After you make these changes, push the updated unit_tests.yml and your test file to Github.

You should see the actions running under "Actions" tab. It will tell you the progress of the test and shows a green check mark if test has passes and red "x" if failed.

Upon failure, github also sends an email to notify you.

**Important**

Please make sure your unit tests does not have any keyboard or plots interaction as Github does not support these features in Github Actions.

Also ensure that your test file does not wait on a user input or the tkinter "mainloop" as these commands runs indefinitely and may cause a timeout.

There is a timeout-minute attribute to the github action, this is to prevent the action from running for too long.

As we create more tests, this attribute might need to be increased to ensure all unit tests are performed.

Currently it has been set to 5 minutes as the average test takes 2.5 minutes.

