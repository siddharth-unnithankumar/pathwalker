import os
import sys


# get the current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
# add the package directory to PYTHONPATH
sys.path.append(os.path.abspath(f"{dir_path}/../pathwalker"))