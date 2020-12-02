# Reinforcement Learning Agent for Crew of One

As the title suggests, this is a Machine Learning agent build to play the game "Crew of One". The original game was made by a team including myself for the GMTK 2020 Game Jam. I have since made some adaptations to interface with the ML Agent. The agent is built in python using the Keras libraries. The model is using the [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) (DDPG) algorithm to learn a policy over a couple thousand lives in "Crew of One".

For instructions on running the network for youreself, scroll down to the "Getting Started" Section

## About the Project

The Project consists of the game and the Network, each runs on a cycle and they communicate with each other through the Redis database. The game itself can run without the network running, but in this case the player will never change direction. Every frame the game reads the current input value off of redis and plays with that input, it also outputs state of play each frame. This state includes the positions of every important element on screen, as well as some data on the player such as its current position and rotation.

On the Network side, it is also running in a loop, but it is not syncronized with the game's internal loop this means the network does not necesarily run once every frame. instead the Network processes the most recent gamestate each time it loops and returns a next move. This move is then used in all subsequent frames untill it is updated again by the Network.

This general archetecture, including the DDPG algorithm, was fairly straight forward to set up. I had a little trouble with connecting the game in unity to the redis server and properly transmitting data, but it was not long before I had player making random moves directed by the Network. Unfortunately, the actual training was not as simple.

## Getting Started

In order to run Crew of One AI on your machine you will need the AI enabled copy of the game and the network itself, these can be found in this repository. You will also need a redis server running on the default port.

### Prerequisites

For the project to run it needs a redis server running at 
For information on downloading and setting up redis see: https://redis.io/

You will also need to be able to run python scripts.

### Running on Windows

Once the files have been downloaded, run the python script "Network.py". Depending on the details of your computer it might print a warning about not having a GPU, this can be ignored. Once the script has printed "waiting for start", open the game.

Opening Crew of One is as simple as running the .exe file included, once the game has started click "Start AI Game" to begin training. Double check that the player is in fact making moves (you should see it change direction a couple times a second), and that python is printing score updates after each death. These indicate that the network is in fact running. The game will restart autimatically after each death unless you check the box in the lower left of the screen during gameplay.

## Built With

* [Unity Engine](https://unity.com/) - To build the game: "Crew of One"
* [Crew of One](https://singularitysystems.itch.io/crew-of-one) - Link to a human playable version of the game.
* [Keras](https://keras.io/examples/rl/ddpg_pendulum/) - Pyhton Library, for the Neural Network
* [Redis](https://redis.io/) - Database system

## Authors

* **Zack Johnson** - *Initial work* - [Zackamo](https://github.com/Zackamo)

See also the list of [contributors](https://github.com/Zackamo/CrewOFOneAI/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks to Will Beddow for pointing me towards redis and helping me troubleshoot it.
