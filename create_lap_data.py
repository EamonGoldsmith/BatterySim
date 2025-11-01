
# generates current draw and time value pairs in CSV format.
# the array c contains key values which are interpolated between to create the current draw data

import numpy as np
import pandas as pd

c = [0, 1, 2, 3, 4, 12, 20, 29, 20, 21, 22, 4, 2, 8, 9, 35, 40, 20, 18, 5]
t0 = np.linspace(0, len(c) - 1, len(c))
t1 = np.linspace(0, len(c) - 1, 200)
ci = np.interp(t1, t0, c)
ts = np.linspace(0, 100, 200)

df = pd.DataFrame({"t": ts, "i": ci})
df.to_csv("lap_data.csv", index=False, header=False)
