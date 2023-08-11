#!/bin/bash

# Run the server in the background
gnome-terminal -- python main.py --num_workers 3 --num_rounds 1

# Run each worker script in separate terminal windows
gnome-terminal -- python Worker1.py
gnome-terminal -- python Worker2.py
gnome-terminal -- python Worker3.py
gnome-terminal -- python Worker4.py
gnome-terminal -- python Worker5.py
gnome-terminal -- python Worker6.py
