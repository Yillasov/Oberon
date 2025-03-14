Installation
===========

This guide will help you install the Neuromorphic Biomimetic UCAV SDK and its dependencies.

Requirements
-----------

- Python 3.8 or higher
- pip package manager
- Optional: CUDA-compatible GPU for accelerated neuromorphic simulations

Basic Installation
-----------------

You can install the SDK using pip:

.. code-block:: bash

   pip install neuromorphic-ucav-sdk

Development Installation
-----------------------

For development purposes, you can install the SDK from source:

.. code-block:: bash

   git clone https://github.com/username/neuromorphic-ucav-sdk.git
   cd neuromorphic-ucav-sdk
   pip install -e .

This will install the SDK in development mode, allowing you to modify the code and see the changes immediately.

Hardware-Specific Installation
-----------------------------

Depending on your target hardware, you may need to install additional dependencies:

CUDA Support
~~~~~~~~~~~

For CUDA support (recommended for neuromorphic simulations):

.. code-block:: bash

   pip install neuromorphic-ucav-sdk[cuda]

Neuromorphic Hardware
~~~~~~~~~~~~~~~~~~~

For specific neuromorphic hardware support:

.. code-block:: bash

   pip install neuromorphic-ucav-sdk[loihi]  # For Intel Loihi
   pip install neuromorphic-ucav-sdk[spinnaker]  # For SpiNNaker