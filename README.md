## Installation

### Virtual Environment
1. Create a virtual environment
   Run the following command to create a virtual environment named "env" (you can replace "env" with your preferred name):

```bash
python -m venv env
```

2. Activate the Virtual Environment:
   To activate the virtual environment, run:

```bash
source env/bin/activate
```

3. Install Packages and Run Your Project:
   After activating the virtual environment, you can install packages and run your project as usual:

```bash
pip install -r requirements.txt
chmod +x run_server.sh
.\run_server.sh
```

5. Deactivate the Virtual Environment:
   When you're done working in the virtual environment, you can deactivate it using the following command:

```bash
deactivate
```

The virtual environment will be deactivated, and you'll return to the regular Command Prompt.

### Running

### Server
#### Steps
1) Change Directory
```bash
FL_System/server
```
2) Check your system ip address and replace the host address in the config_app.py
```bash
HOST = ''
```
3) Run 
```bash
bash server.py
```

### Client 
#### Steps
1) Change Directory
```bash
FL_System/client
```
2) Put the same host address that put on server side in the config_app.py
```bash
HOST = ''
```
3) Run 
```bash
python3 Worker1.py
```

Note: Repeat the same process for other client as well



