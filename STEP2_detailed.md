**INTEGRATING REINFORCE AGENT TO A SUMO ENVIRONMENT**
The goal of the second part of this project is to use the REINFORCE algorithm to control traffic signals at an intersection. 
To do so I use the already built [DQN framework for SUMO](https://github.com/romainducrocq/DQN-ITSCwPD.git) by Ducrocq Romain, who implemented several DQN type algorithms. 

**Task** = transitionning from DQN to Policy Based : 

- Action selection : Sample from $\pi(a,s)$
- On policy training step (uses episode trajectory)
- Loss function = policy gradient
- Policy network (probabilities instead of values)
- Output = probability distribution over actions

**Integration approach : **
