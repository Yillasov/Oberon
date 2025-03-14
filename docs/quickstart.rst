Quick Start Guide
===============

This guide will help you get started with the Neuromorphic Biomimetic UCAV SDK.

Basic Usage
----------

Here's a simple example to create a basic UCAV configuration:

.. code-block:: python

   from oberon import System, components
   
   # Create a new UCAV system
   ucav = System("my_ucav")
   
   # Add components
   ucav.add_component(components.Airframe("biomimetic_wing"))
   ucav.add_component(components.FlightController())
   ucav.add_component(components.SensorArray())
   
   # Initialize the system
   ucav.initialize()
   
   # Run a simulation
   ucav.simulate(duration=10.0)  # Simulate for 10 seconds

Component Configuration
---------------------

Components can be configured using configuration files or programmatically:

.. code-block:: python

   from oberon import Configuration
   
   # Load configuration from file
   config = Configuration()
   config.load_file("ucav_config.yaml")
   
   # Create components with configuration
   controller = components.FlightController(config.get("flight_controller"))
   
   # Or configure programmatically
   controller.set_config(
       control_rate=100,  # Hz
       stability_mode="adaptive"
   )

Creating Custom Components
------------------------

You can create custom components by inheriting from the base Component class:

.. code-block:: python

   from oberon.core import Component
   
   class CustomSensor(Component):
       def __init__(self, component_id=None, **kwargs):
           super().__init__(component_id, **kwargs)
           # Initialize your sensor
       
       def _initialize(self):
           # Initialization logic
           pass
       
       def _update(self, delta_time):
           # Update logic - called every simulation step
           pass
       
       def get_reading(self):
           # Custom method
           return sensor_value

Next Steps
---------

- Explore the :doc:`API Reference <api/index>` for detailed information on all components
- Check out the :doc:`Examples <examples/index>` for more advanced usage scenarios
- Learn about the :doc:`Architecture <architecture>` of the SDK