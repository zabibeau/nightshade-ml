# Nightshade

This repo contains code and information for a project for CS 5806

## Setup Instructions
1. Install python if you haven't (use any version except for 3.13)
2. Run `python3 -m venv nightshade_env` in a terminal
3. Run `source nightshade_env/bin/activate` to activate the venv, and `deactivate` to deactivate it
4. Install pytorch by going to: [https://pytorch.org/get-started/locally/]
5. To install all requirements, run `pip3 install -r requirements.txt`
6. To download all training data, run `python3 download_data.py`. This will take a little bit, as we're downloading data from the 2014 Coco dataset (it's around 13GB)