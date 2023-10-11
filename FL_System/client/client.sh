# Define the number of workers
num_workers=3

# Run the workers in separate terminals
for ((i=1; i<=num_workers; i++)); do
    gnome-terminal -- bash -c "echo 'Running Worker$i.py'; python3 Worker$i.py; exec bash"

    # Sleep briefly to allow the new terminal to open before the next one starts
    sleep 1
done