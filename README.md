<h1 align="center">
Deep Reinforcement Learning – Intelligent Traffic Signal Control with Partial Detection
</h1>

<p align="center">
  <em>
    Implementation of a policy-based Deep Reinforcement Learning approach (REINFORCE)
    for automated traffic signal control at urban intersections using SUMO.
  </em>
</p>
<p align="center">
  <em>
    BSc in Computer Science & Mathematics 3rd Year research project
  </em>
</p>
<p align="center">

  
<img width="410" height="123" alt="images" src="https://github.com/user-attachments/assets/43281565-30d0-4af2-8d5b-e830d73d8e99" />
</p>



## Overview

This project implements a **REINFORCE Deep Reinforcement Learning agent** to control
traffic lights at a road intersection under **partial vehicle detection**.

The objective is to optimize traffic flow by dynamically selecting traffic signal phases
based on real-time observations, using a **SUMO-based traffic simulation** integrated
into a Gym-compatible environment.

The work combines **theoretical reinforcement learning foundations** with a
**realistic traffic control application**, and provides quantitative comparisons with
state-of-the-art baselines.

---

## Traffic Control Scenario

The considered scenario is a **two-phase, double-lane urban intersection** with
permitted conflicting left turns, representative of common real-world layouts.

<img width="551" height="213" alt="Screenshot 2025-12-12 at 08 59 47" src="https://github.com/user-attachments/assets/e1486903-d483-48a9-8ab3-1e32907c0f33" />


The agent alternates between two traffic signal phases controlling the
north–south and east–west axes.

---

## Simulation Environment

The intersection is simulated using **SUMO** and visualized via **SUMO-GUI**,
allowing qualitative inspection of the learned control policy.


<img width="479" height="249" alt="Screenshot 2025-12-12 at 08 59 53" src="https://github.com/user-attachments/assets/30e469ae-ce9f-4755-ac33-2b406992dfd8" />


The environment is connected to Python through **TraCI**, enabling real-time
observation of vehicle states and direct traffic light control.

---

## Methodology

### Step 1 – Generic REINFORCE Agent

A generic REINFORCE agent is first implemented and validated on standard Gym environments:

- **ReinforceAgent_MLP.py**
  - Two-layer MLP policy network
  - Tested on `CartPole-v1`

- **ReinforceAgent_CNN.py**
  - Convolutional policy network for image-based environments
  - Frame stacking, grayscale preprocessing, softmax action sampling
  - Tested on `MiniGrid-Empty-5x5-v0`
  - Training monitored with TensorBoard

To train and visualize:
```bash
python ReinforceAgent_CNN.py
tensorboard --logdir=runs/reinforce
```

### Step 2 – Intelligent Traffic Signal Control with SUMO

The REINFORCE agent is integrated into a custom SUMO Gym environment to control
traffic signals at the intersection.

  A traffic light scheduler enforces:
  - Minimum phase durations (anti-flickering)
  - Congestion-aware action triggering
  - Safety constraints on phase switching

---

## Reward Design

The reward function is designed to minimize total travel delay by penalizing
speed loss relative to free-flow conditions.

It is based on the total squared delay across all vehicles, encouraging both
traffic efficiency and fairness between road users.

---

## Key Performance Indicators (KPIs)

After each phase decision, the following KPIs are computed:

  - Accumulated waiting time
  - Delay
  - Queue length
  - Traffic volume

These metrics are logged during training and used for quantitative evaluation.

---

## Results

###KPI Statistics

Comparison of REINFORCE against baseline controllers
(DQN, MaxPressure, SOTL):

<img width="564" height="312" alt="Screenshot 2025-12-12 at 08 59 59" src="https://github.com/user-attachments/assets/b47cbbd0-3948-41df-8864-082da719e382" />


### Comparative Performance

Relative KPI comparison between REINFORCE and competing approaches:

<img width="555" height="159" alt="Screenshot 2025-12-12 at 09 00 06" src="https://github.com/user-attachments/assets/8371336a-6050-4ffc-bc61-e3b9b1546b32" />

REINFORCE achieves a strong overall compromise, particularly excelling in
delay and queue length reduction, at the cost of higher variance.

## Discussion

Policy-based methods such as REINFORCE exhibit higher variance than value-based methods. REINFORCE achieves competitive average performance with reduced stability. Variance originates from Monte Carlo policy gradient estimation. These observations motivate extensions toward more stable architectures.

## Possible Improvements

  - Baseline or advantage functions for variance reduction
  - Actor–critic architectures
  - Entropy regularization for improved exploration
  - Multi-intersection and network-level traffic control

## Acknowledgements

_This project was carried out as a First-Year Research Project (2025)
under the supervision of Nadir Farhi and Zoi Christophorou
(GRETTIA Laboratory, Université Gustave Eiffel)._

_We also acknowledge prior work by Romain Ducrocq on DQN-based traffic signal control,
which provided valuable benchmarking references._
