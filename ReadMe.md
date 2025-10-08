# Thought Experiment -  Crossing Streets

The idea for this project originated from the simple, everyday thought experiment: 
"You need to get across the street down the road. Along this street there are multiple opportunities for crossing. What would be your actions? Would you  wait in front of the first crossing till you can go or just walk as long as you can till crossing the street is possible?



## Environment
I model a simplified version of the pedestrian walking routes in front of the University of Mannheim with [gymnasium](https://gymnasium.farama.org/).

I choose start and target locations who are the most common places for students. E.g. the castle, the library, the cafeteria, a coffee shop and a grocery store.

The light sequences are arbitrary choosen and doesn't reflect real-world conditions.

# Results
![Pygame Env](https://github.com/Frederik-Roeckle/crossing-streets-rl/imgs/RL_pygame_env.gif)

## Training
### Training in Docker Container
Run command `docker run walkorwait-train` to save the generated policy from the output dir. To transfer the generated policies back from a server to you, use `scp [USER]@[SERVER_ADDR]:[PATH_TO_FILE] .`