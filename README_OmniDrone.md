<img src="./docs/images/banner.png" width="999" border="1"/>


[![Organization](https://img.shields.io/badge/Organization-PKU_MARL-blue.svg "Organization")](https://github.com/PKU-MARL "Organization")[![Unittest](https://img.shields.io/badge/Unittest-passing-green.svg "Unittest")](https://github.com/PKU-MARL "Unittest")[![Docs](https://img.shields.io/badge/Docs-In_development-red.svg "Author")](https://github.com/PKU-MARL "Docs")[![GitHub license](https://img.shields.io/github/license/PKU-MARL/DexterousHands)](https://github.com/PKU-MARL/DexterousHands/blob/main/LICENSE)


**OmniDrone** is a differentiable physics engine that simulates environments made up of rigid bodies, joints, and actuators for safe reinforcement learning, multi-agent reinforcement learning. **OmniDrone** is based on
[Brax](https://github.com/google/brax) and is designed for use on acceleration hardware.

To better help the community study Multi-RL, **OmniDrone** are developed with the following key features:

- **More parallel for rollout**: Based on jax and brax, which are both efficient for single-device simulation, and scalable to massively parallel simulation on multiple devices, without the need for pesky datacenters.

Here we provide a table for comparison of **OmniDrone** and before benchmarks.

|                     Benchmark                      |                                                    Github Stars                                                     |                                              Last Commit                                               | Support Parallell  | Support Differential |    Support GPU     | Support Multi-Constraints | Support Muti-agent |
| :------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :----------------: | :------------------: | :----------------: | :-----------------------: | :----------------: |
| [Safety Gym](https://github.com/openai/safety-gym) | [![GitHub stars](https://img.shields.io/github/stars/oxwhirl/pymarl)](https://github.com/oxwhirl/pymarl/stargazers) | ![GitHub last commit](https://img.shields.io/github/last-commit/openai/safety-gym?label=last%20update) |                    |                      |                    |    :heavy_check_mark:     |                    |
| [OmniDrone](https://github.com/PKU-MARL/OmniDrone) |                                                                                                                     |                                                                                                        | :heavy_check_mark: |  :heavy_check_mark:  | :heavy_check_mark: |    :heavy_check_mark:     | :heavy_check_mark: |


Some policies trained via Brax. Brax simulates these environments at millions of physics steps per second on TPU.


**OmniDrone** also includes a suite of learning algorithms that train agents in seconds
to minutes:

*   Baseline learning algorithms such as

    - Single-agent
      - [PPO(Unsafe)](https://github.com/google/brax/blob/main/brax/training/ppo.py),[SAC(Unsafe)](https://github.com/google/brax/blob/main/brax/training/sac.py),[CPO](https://github.com/google/brax/blob/main/brax/training/ars.py),[Lag-base Algorithms](https://github.com/google/brax/blob/main/brax/training/ars.py),[PID-Lag base Algorithms](https://github.com/google/brax/blob/main/brax/training/ars.py),[PCPO](https://github.com/google/brax/blob/main/brax/training/ars.py),[FOCOPS](https://github.com/google/brax/blob/main/brax/training/ars.py),[RCPO](https://github.com/google/brax/blob/main/brax/training/ars.py),[P3O](https://github.com/google/brax/blob/main/brax/training/ars.py)

    - Multi-agent
      - [MAPPO(Unsafe)](https://github.com/google/brax/blob/main/brax/training/ars.py),[MAPPO-Lag](https://github.com/google/brax/blob/main/brax/training/ars.py),[MACPO](https://github.com/google/brax/blob/main/brax/training/ars.py)
*   Learning algorithms that leverage the differentiability of the simulator, such as [analytic policy gradients](https://github.com/google/brax/blob/main/brax/training/apg.py).


## Using **OmniDrone** locally

To install Brax from pypi, install it with:

```
git clone git@github.com:PKU-MARL/OmniDrone.git
cd OmniDrone
conda create -n safe python=3.8
pip install -e .

# You need to install pytorch and JAX[GPU] manual.
# Training on NVidia GPU is supported, but you must first install
[CUDA, CuDNN, and JAX with GPU support](https://github.com/google/jax#installation).

Fox Jax with GPU support, you can install from pypi, install it with:
pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Installs the wheel compatible with Cuda >= 11.1 and cudnn >= 8.0.5
pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Train
To train a model: You need to change path to ```omnidrone/algo```

```
python train.py --env_name ball_run
```
or set algo and envs (and maybe set exact parameters) in training.py file and directly run
```
python training.py
```


### added result
We designed several different tasks for rl, for example run to fly at a certain direction, circle to fly around a circle as close as possible, target to fly to a certain target, and follow, modified from target, to fly towards and follow a moving target at a set distance, just as shown above.This environment is heavily based on previous Brax and makes good use of its phisical collision features.
Our environment is specially designed for quadcopters. For example we employed mainly two drone control dynamics types: first one called dyn with four forces added on four rotors of drone, vertical to drone surface, which may be a real model of Cruciform UAVs; the second model is simply xyz control by adding direct force to the torso in three directions, adapted from pid control.
Besides our environment supported safe rl train with cost item added to original brax envs.
We tested our envs with standard ppo, sac, ppol and apg, a differentiable algo. The last one performed its outstanding potential.
A rough experimental result is shown below.
![pic](./docs/images/example.jpg)
### Visualization
To visual a model: You need to change path to ```omnidrone/algo```

```
python vis.py --env_name ball_run
```
During training, algos will keep no-traced experiment result(drone actions and status) as html in correspending folders in ```omnidrone/log/outputs``` file folder.
If need with-traces replay, run like following to employ params saved in training to replay a whole trajectary
```
python replay.py --env_name drone_circle_dyn --action_size 4 --name params.pt
```
Also, to we provided a complete graph maker by running like(this will search for all fitted stats in root path)
```
python tools/logger/tools.py --root_dir tools/logger/run --alg_name apg
```
to store multi algo results with cloud in several graphs;
then produce the final graph with multi algo results by
```
python tools/logger/plotter.py --shaded-std --root_dir tools/logger/follow --smooth 75 --title follow_reward
```
in which 'name' is refered as params model name.




## Some know usage bugs

### Problem1
```
jaxlib.xla_extension.XlaRuntimeError:
INVALID_ARGUMENT: DLPack tensor is on GPU, but no GPU backend was provided.
```
You need to install JAX[GPU] version,
for example
```
pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Installs the wheel compatible with Cuda >= 11.1 and cudnn >= 8.0.5
pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Citing **OmniDrone**

If you would like to reference Brax in a publication, please use:

```
arxiv
```


## Tasks Visualization
|                                                    Drone Run                                                    |            Drone Circle             |            Drone Target             |            Drone S-bend             |
| :-------------------------------------------------------------------------------------------------------------: | :---------------------------------: | :---------------------------------: | :---------------------------------: |
<<<<<<< HEAD
| ![image](https://user-images.githubusercontent.com/75480302/183473681-7b4e9fe1-0f31-4eb6-befe-0c3d2e6cb937.png) | ![pic](./docs/img/drone_circle.jpg) | ![pic](./docs/img/drone_target.jpg) | ![pic](./docs/img/drone_S-bend.jpg) |
=======
| ![image](https://user-images.githubusercontent.com/75480302/183473681-7b4e9fe1-0f31-4eb6-befe-0c3d2e6cb937.png) | ![pic](./docs/images/drone_circle.jpg) | ![pic](./docs/images/drone_target.jpg) | ![pic](./docs/images/drone_S-bend.jpg) |
>>>>>>> 5f11379e3d6ecea0dc675adab60f8914ae802c73


## Acknowledgements

The development of **OmniDrone** relies on many excellent open source repositories. We sincerely appreciate the following open source github repo:

* [JAX](https://github.com/google/jax). JAX is Autograd and XLA, brought together for high-performance machine learning research.

* [Brax](https://github.com/google/brax). Brax is a differentiable physics engine that simulates environments made up of rigid bodies, joints, and actuators.
