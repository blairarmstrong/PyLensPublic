Command Reference
==========================


**Network Commands**

- :doc:`create_net`: |create_net|
- :doc:`add_net`: |add_net|
- :doc:`use_net`: |use_net|
- :doc:`delete_net`: |delete_net|
- :doc:`store_network`: |store_network|
- :doc:`load_network`: |load_network|
- :doc:`set_time`: |set_time|

**Group Commands**

- :doc:`add_group`: |add_group|
- :doc:`delete_group`: |delete_group|
- :doc:`order_groups`: |order_groups|
- :doc:`get_group_type`: |get_group_type|
- :doc:`change_group_type`: |change_group_type|
- :doc:`copy_unit_values`: |copy_unit_values|
- :doc:`print_unit_values`: |print_unit_values|
- :doc:`reset_unit_values`: |reset_unit_values|
- :doc:`polarity`: |polarity|

**Connection Commands**

- :doc:`connect_groups`: |connect_groups|
- :doc:`connect_group_to_unit`: |connect_group_to_unit|
- :doc:`connect_units`: |connect_units|
- :doc:`elman_connect`: |elman_connect|
- :doc:`copy_connect`: |copy_connect|
.. - :doc:`add_link_type`: TBC
.. - :doc:`delete_link_type`: TBC
.. - :doc:`store_weight`: TBC
.. - :doc:`load_weight`: TBC

**Disconnection Commands**

- :doc:`thaw_group_inputs`: Unfreezes (thaws) specific or all incoming links to a group
- :doc:`disconnect_groups`: |disconnect_groups|
- :doc:`disconnect_group_units`: |disconnect_group_units|
- :doc:`disconnect_units`: |disconnect_group_units|
- :doc:`delete_links`: |delete_links|
- :doc:`delete_group_inputs`: |delete_group_inputs|
- :doc:`delete_group_outputs`: |delete_group_inputs|
- :doc:`delete_unit_inputs`: |delete_unit_inputs|
- :doc:`delete_unit_outputs`: |delete_unit_outputs|

**Link Commands**

- :doc:`freeze`: Freezes the weight updates for the link
- :doc:`unfreeze`: Unfreezes the weight updates for the link

**Lesioning and Healing Commands**

- :doc:`lesion_all_groups`: Applies a lesion to all groups in the network
- :doc:`lesion_specific_group`: Applies a lesion to all groups in the network
- :doc:`lesion_all_links`: Removes or modifies incoming links to all groups in the network
- :doc:`link_specific_lesion`: Applies a lesion to specific links between two groups in the network
- :doc:`lesion_bias_links`: Applies a lesion to the bias links in the network
- :doc:`heal_all_groups`: Restores lesioned units in all groups within the network
- :doc:`heal_specific_group`: Restores lesioned units in a specific group
- :doc:`heal_all_links`: Restores lesioned links across all groups in the network

**Noise Commands**

- :doc:`add_all_links_noise`: Applies noise to all links in the network based on a specified distribution
- :doc:`add_links_specific_noise`: Applies noise to specific links between two groups in the network
- :doc:`remove_all_links_noise`: Removes noise from all links in the network
- :doc:`remove_links_specific_noise`: Removes noise from specific links between two groups in the network

**Example Commands**

- :doc:`load_example_set`: Reads an example file and loads it into an example set

**Training Commands**

- :doc:`train`: |train|
- :doc:`set_update_method`: |train|
- :doc:`update_weights`: |update_weights|
- :doc:`test`: |test|
- :doc:`reset_network`: |reset_network|
- :doc:`reset_derivs`: |reset_derivs|
- :doc:`save_stats`: |save_stats|

**Parallel Training Commands**

- :doc:`set_properties`: Set the properties of the network as required for parallel training mode


