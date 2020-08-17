import timing_argument as ta
import seaborn as sns
import matplotlib.pyplot as plt

(th, sc) = ta.mass_estimate()

sns.lineplot(th, sc)
#plt.show()

