# Trajectories in Hand Synergy Spaces for in-hand
The branches in this repo represent the different attempts at solving the problem described in this work.

The main branch contains the final approach taken, as well as the documents regarding both results, and the written works that accompany this project.

It was also added a folder called IsaacGym which has the code files for use in the NVidia IsaacGym environment. 

For a quick description of the files present>

utils.py: contains mathematical functions used in differnet part of the project, including the loss function equations.
data_loader.py: contains functions that help loading the data used in this work
cvae.py: contains the CVAE model, specifically the layers present in it
train_cvae.py: contains the training mechanism of the model. contains the metric tensor, the matrix G, with 3 differnt approaches.
create_trajectories.py: creates trajectories for a in hand manipulation task, outputs a showing of the trajectoy in the current space, as well as smoothness values for a trajectory.
check_pairs.py: aux file, to see if the 2 grasps that belong in a pair for trajectory calculation belong to the same object. 

For any questions, contact me via email @ jorge.c.telo@tecnico.ulisboa.pt or on discord, @ jtelo.
