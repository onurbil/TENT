import numpy as np
from visualization import attention_plotter


# Test for attention plotter: 
heads = 8
batch = 16
y_label = ['2012-10-01\n 17:00','2012-10-01\n 18:00','2012-10-01\n 19:00','2012-10-01\n 20:00','2012-10-01\n 21:00',
'2012-10-01\n 17:00','2012-10-01\n 18:00','2012-10-01\n 19:00','2012-10-01\n 20:00','2012-10-01\n 21:00',
'2012-10-01\n 17:00','2012-10-01\n 18:00', '2012-10-01\n 19:00', '2012-10-01\n 20:00','2012-10-01\n 21:00',
'2012-10-01\n 17:00']
attention = np.random.random((batch,heads))
# Weights are already normalized during training. Therefore next two steps are only for test:
max_attention = attention.sum(axis=0)
norm_attention = attention/max_attention
attention_plotter(norm_attention, y_label, save=False)