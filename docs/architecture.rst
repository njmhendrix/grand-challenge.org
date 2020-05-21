============
Architecture
============

This document uses the `C4 Model`_ for visualising the architecture of grand-challenge.org.
This covers `System Context`_, `Containers`_ (Applications/Data Stores), and `Components`_ (Modules/Libraries).

Ubiquitous Language
-------------------

Users
~~~~~

Many different users and roles are supported on grand-challenge.org.

Researcher
    A Researcher is a user who wants to manage medical imaging data and reports,
    collect annotations on medical imaging data,
    create a challenge for the data science community to generate solutions to clinical problems,
    and objectively evaluate the submitted algorithms to challenges.

Data Scientist
    A Data Scientist is a user who participates in a challenge,
    downloads the training medical imaging data & annotations for a challenge,
    uploads algorithms for a challenge,
    and makes those algorithms available for clinicians to execute.

Clinician
    A Clinician is a user who uses a workstation to learn how to annotate or read medical imaging data,
    uses a workstation to make annotations to medical imaging data,
    and uploads new medical imaging data for execution by algorithms.




System Context
--------------

An overview of the grand-challenge.org system, its users, and its system dependencies.

.. uml:: diagrams/system_context.puml

Containers
----------

The overall shape of the architecture and technology choices.
*Note: a container is a separately deployable application or data store, not necessarily a Docker container.*

.. uml:: diagrams/container.puml

Components
----------

Logical modules and their interactions with containers.
*Note: a component is a grouping of related functionality behind a well defined interface,
not components in the sense of pipelines.*

.. .. uml:: diagrams/samples/component.puml


.. _`C4 Model`: https://c4model.com/