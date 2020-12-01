# Reinforcement Learning Agent for Crew of One

As the title suggests, this is a Machine Learning agent build to play the game "Crew of One". The original game was made by a team including myself for the GMTK 2020 Game Jam. I have since made some adaptations to interface with the ML Agent. The agent is built in python using the Keras libraries. The model is using the Deep Deterministic Policy Gradient algorythm to learn a policy over a couple thousand lives in "Crew of One".

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
* [Redis] 

## Authors

* **Zack Johnson** - *Initial work* - [Zackamo](https://github.com/Zackamo)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
