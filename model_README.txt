README For Model


The model.py file is making some assumptions:

1) The original 3 data sets are still present in artists folder

2) The `[new-artist-name].csv` and `[new-artist-name]-test.csv` have the EXACT same columns as the original dataset. Specifically, the test set should have a 'hammer-price' column(can be all -1), this will be dropped later. There is a specific reason for this - Since I am making dummy features, it's important that I first merge all data together, preprocess everything and then split into test and train so that all the data eventually has the same columns. This would be much easier if I were to use an individual model as opposed to a pooled one. However, I am using the pooled model because I have no information about the size of the new training data and it makes more sense to use a pooled model.

3) The final histogram is just a histogram of predictions, as described, the test data only has 10 sample, thus the histogram will probably not show too much information. It's not exactly what is asked for part 2.2(KDE of the estimated prices). 

4) Requirements:

A modern computer(linux based system -Mac/ubuntu is better)
Python version 3.x
from utils import preprocess_all
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
import matplotlib.pyplot as plt


5) Running Instructions:

	a) Clone the repository
	b) From the terminal, go to the local repository
	c) From the terminal, run - python model.py
	d) Follow on screen instructions(inputting new data paths essentially)
	e) That should do it!


