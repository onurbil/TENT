
from debugging_tools import *

import numpy as np
import pandas as pd

# Column number must be x times 9:
test = np.arange(360)
test = pd.DataFrame(test.reshape([10, 36]))

print(test)

#######################

# Test debug() function:

print('debugging of a pandas dataframe:')
debug(test, p=True)
print('')

print('debugging of a list:')
debug([test], p=False)

########################

# Test pause() function:

print('')
pause()
print('')

########################

# Test printall() function:

printall(test)

########################

# Test getTime() function:

print(getTime())

########################

# Test execTime functions:

execTime(getTime())
