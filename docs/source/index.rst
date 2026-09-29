Experience and Reasoning Agent
========================

.. code-block:: text

   belief/  reasoning/  environment/    three leaves; none may import another
   agents/                              may import all three
   driver                               may import agents + environment

The point is that a single period can be reasoned about one layer at a time: the planner
cannot quietly start reading the belief, and the environment cannot start asking what the
agent believes.

:doc:`api/belief`
   The posterior over :math:`Q`, its kernels, and its prior means.

:doc:`api/reasoning`
   Pricing the menu's contrasts (the planner) and answering them (the oracle).

:doc:`api/environment`
   The entrepreneur's problem, the menu of actions, and the building blocks they compose from.

:doc:`api/agents`
   The experience-and-reasoning learner, and the value-function-iteration benchmark.

:doc:`api/driver`
   The period loop that runs an episode or a panel.

:doc:`api/visualization`
   Figure helpers and house style.

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
