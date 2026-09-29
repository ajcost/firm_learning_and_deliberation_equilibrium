Experience and Reasoning
========================

Documentation for the simulation code in ``src/``: an entrepreneur who learns about the
action-value function :math:`Q` through two channels at once.

It **experiences** — every period it lives through is filed as a GPTD observation against a
Gaussian-process belief. It **reasons** — a planner prices the contrasts on the menu it faces,
and an oracle answers the ones worth buying. One belief absorbs both, which makes the two
channels substitutes for the same posterior. Choice is made under an entropy floor, so how
sharply the agent can act is tied to how much it knows.

These pages are generated from the docstrings in the source. Each module's own documentation
is the authoritative description of what it does and of the mathematics it implements; this
site is a rendering of it, not a second account.

How the code is laid out
------------------------

The package is organised as a strict dependency graph, checked by the test suite rather than
left to convention:

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
