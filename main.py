import sys
import os 

# Obtaining the absolute route for the root folder 
root_folder = os.path.abspath('.')

# Aggregating the rout to the PYTHONPATH 
sys.path.append(root_folder)