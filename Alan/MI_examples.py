import numpy as np
import matplotlib.pylab as plt
import minepy as mpy

x = np.linspace(-1, 1, 200)
y1 = x + np.random.normal(0, 1, 200)

plt.plot(x, y1)
plt.show()

y2 = x ** 2 + 10 * np.random.normal(0, 1, 200)
plt.plot(x, y2)
plt.show()

mine = mpy.MINE(alpha=0.5, est="mic_approx")
mine.compute_score(x, y1)
mine.mic()

mine2 = mpy.MINE(alpha=0.5, est="mic_approx")
mine2.compute_score(x, y2)
mine2.mic()

y15 = x
min3 = mpy.MINE(alpha=0.5, est="mic_approx")
min3.compute_score(x, y15)
min3.mic()
