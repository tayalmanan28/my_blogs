---
toc: true
layout: post
description: Safe Model-Based Reinforcement Learning using Robust Control Barrier Functions.
image: images/framework_diagram.png
categories: [Reinforcement-Learning,Robotics]
title: Safe Reinforcement Learning
---
# Safe Reinforcement Learning
Safe Reinforcement Learning can be defined as the process of learning policies that maximize the expectation of the return in problems in which it is important to ensure reasonable system performance and/or respect safety constraints during the learning and/or deployment processes.

# SAC-RCBF 

"Safe  Model-Based  Reinforcement  Learning using Robust Control Barrier Functions". Specifically, an implementation of SAC + Robust Control Barrier Functions (RCBFs) for safe reinforcement learning in two custom environments.

While exploring, an RL agent can take actions that lead the system to unsafe states. Here, we use a differentiable RCBF safety layer that minimially alters (in the least-squares sense) the actions taken by the RL agent to ensure the safety of the agent.


## Robust Control Barrier Functions (RCBFs)

In this work, we focus on RCBFs that are formulated with respect to differential inclusions of the following form:
![framework_diagram](https://user-images.githubusercontent.com/42448031/168827781-e7d59ff5-5cf8-4511-b737-84d0ec30512c.png)
![diff_inc](https://user-images.githubusercontent.com/42448031/168827930-f4ea14fa-3443-4698-86d2-3ad3207996ee.png)


Here `D(x)` is a disturbance set unkown apriori to the robot, which we learn online during traing via Gaussian Processes (GPs). The underlying library is GPyTorch. 
 
The QP used to ensure the system's safety is given by:

![qp](https://user-images.githubusercontent.com/42448031/168828078-3474e18e-4832-4f9f-bfaf-d9d0103a969b.png)

where `h(x)` is the RCBF, and `u_RL` is the action outputted by the RL policy. As such, the final (safe) action taken in the environment is given by `u = u_RL + u_RCBF` as shown in the following diagram:

![policy_diagram](https://user-images.githubusercontent.com/42448031/168828289-0711a2f5-5067-4764-a1b9-f192cabebebf.png)


## Coupling RL & RCBFs to Improve Training Performance

The above is sufficient to ensure the safety of the system, however, we would also like to improve the performance of the learning by letting the RCBF layer guide the training. This is achieved via:
* Using a differentiable version of the safety layer that allows us to backpropagte through the RCBF based Quadratic Program (QP) resulting in an end-to-end policy.
* Using the GPs and the dynamics prior to generate synthetic data (model-based RL).

