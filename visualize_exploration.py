import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./2.5M_Agent/training.log", header=None)
print(type(df))
df.plot.line()
plt.show()