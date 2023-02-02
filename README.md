# pathwalker
an individual-based model for ecological and conservation science which simulates movement paths and predicts landscape connectivity 
documentation can be found in the pathwalker directory

## Setup
The dependencies required for running `pathwalker` are stored in `requirements.txt`.
You can install these however you like, but the simplest way is to create a new virtual environment for running pathwalker and then installing the dependencies into it.
You can create the environment using the following command (please fill in the path as appropriate):
```bash
python3 -m venv <your_path_here>
```
Then activate the environment and install requirements by running
```bash
source <your_path_here>/bin/activate
pip install -r requirements.txt
```
You can run `pathwalker` within this environment.
When you are finished, deactivate the environment by running
```bash
deactivate
```