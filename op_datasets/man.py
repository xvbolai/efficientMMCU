import pandas as pd

import numpy as np
import pandas as pd

df = pd.read_csv('dataset_conv2d_large_cnn.csv')
df = df.drop_duplicates(subset = None, keep ='first', inplace = False)
df = df[df['GP'] != df['IC']]
df = df[df['KH'] != 1]
df.to_csv('convolution.csv',index = False ,sep = ',')