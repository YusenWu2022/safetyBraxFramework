# Envs

```bash
envs
├── assets
│   ├── json # json files for assets.
│   ├── asset.py # Asset class is the base class for all assets. Asset provides the methods to load or save config files, register itself, and operation related to UID.
│   ├── ant.py # Ant robot.
│   └── ground.py # Ground.
├── tasks
│   ├── task.py # Task is the base class for all tasks. Task provides the methods to register new assets and new collision pairs. The subclasses of Task should implement the method `get_obs`, `calculate_reward`, `calculate_cost`, and `is_done`. Meanwhile, they should specify the `_metrics` and the update method for each metric.
│   └── velocity.py # Velocity task.
├── env.py # Basic class of state, environment, and wrapper.
├── builder.py # Subclass of env.Env, building the whole world using the robot class and the task class.
└── wrappers.py # Wrapper classes, including VectorWrapper, VmapWrapper, EpisodeWrapper, AutoResetWrapper, and EvalWrapper.
```
