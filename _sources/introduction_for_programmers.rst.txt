Introduction
==============

PyLens was designed with the realization that it is not possible to write a simulator that can satisfy the needs of all users right out of the box. Users can access Python data structures within scripts and assemble neural networks from components such as units, groups (layers), activation functions, and links. This makes it possible to implement many customizations without altering the code directly. Nevertheless, more low-level additions are often necessary, and one of the main goals of Lens was to make it relatively easy for more advanced users to get in and change the code to fit their needs.

The main activity of neural network computation is handled in the backend code. This backend is accompanied by a Tkinter-based GUI interface that is responsible for neural network visualization, graph plotting, and commands that link to the backend.

Because PyLens is still a work in progress, it is likely that modifications to the main code will be made frequently. The philosophy at this point in development is that nothing is guaranteed to remain the same, though any changes will be backward compatible as much as possible.

This is bound to create some conflict with users who have their own versions of the code. Therefore, when changing the source code, it is important to restrict any changes to as few modules as possible. That way, if the generic versions of those modules are changed, reimplementing the additions is as painless as possible.
