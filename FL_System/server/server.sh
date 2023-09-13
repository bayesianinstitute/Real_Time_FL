#!/bin/bash

# Run the main.py command in the background
gnome-terminal -- bash -c "python main.py --num_workers 3 --num_rounds 1; exec bash"
