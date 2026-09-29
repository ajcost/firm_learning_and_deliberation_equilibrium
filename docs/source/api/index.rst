API Reference
=============

Every module under ``src/``, ordered by the dependency layering the package enforces:
``belief``, ``reasoning`` and ``environment`` are leaves that never import each other,
``agents`` composes all three, and ``driver`` sits on top.

.. toctree::
   :maxdepth: 2

   belief
   reasoning
   environment
   agents
   driver
   visualization
