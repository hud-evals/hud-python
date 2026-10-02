"""HUD eval: the v6 execution engine.

A :class:`~hud.Task` is a row pointing at its env; a :class:`~hud.Taskset`
groups many, and agents run against live :class:`~hud.Run` handles.
:func:`~hud.eval.run.rollout` is the execution atom (one agent, one task, fully
recorded); ``Task.run`` and ``Taskset.run`` are the scheduler over it, both
returning a :class:`~hud.Job` — the platform receipt. There are no standalone
traces: every run reports under a job.

This is the top layer: eval composes :mod:`hud.environment` and
:mod:`hud.agents`, which never import each other and never import eval back —
agents see eval only through the ``Run`` handle they are driven with. (Sole
exception: calling an ``@env.template`` declaration constructs the eval ``Task``
row.)

Its public names are exported from :mod:`hud`. Placement is passed at execution
time (see :mod:`hud.runtime`): ``LocalRuntime`` a local source served
in-process, ``DockerRuntime`` an image, ``Runtime(url)`` an env served
elsewhere, ``HUDRuntime`` a HUD runtime tunnel, or ``HostedRuntime`` to run the
whole rollout remotely on the platform::

    from hud import Taskset
    from hud.runtime import LocalRuntime

    job = await my_task(a=1).run(agent, runtime=LocalRuntime("env.py"))
    job = await Taskset("demo", [my_task(d) for d in range(5)]).run(
        agent, runtime=LocalRuntime("env.py"), group=8
    )
"""
