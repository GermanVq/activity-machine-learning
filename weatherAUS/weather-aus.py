# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:07:51 2020


                   WEATTHER-AUS

@author: German
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing Dataset
url = 'weatherAUS.csv'
data = pd.read_csv(url)
