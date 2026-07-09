"""Ultra compute service — model-agnostic GPU job runner.

A small framework (Bearer auth + job manager + executor registry) that runs
GPU work on behalf of the training worker and, generically, any model. Add a model
by writing one Executor and registering it; the service internals never change.
"""
