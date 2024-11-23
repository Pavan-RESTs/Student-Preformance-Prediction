import os
import sys

import numpy as np
import pandas as pd
import pickle

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging

def save_object(file_path, obj):
    try:
        with open(file_path,'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException(e,sys)
                
                