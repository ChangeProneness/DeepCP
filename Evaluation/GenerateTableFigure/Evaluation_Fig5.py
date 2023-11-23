import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

plt.figure(figsize=(7, 5))


def smooth_xy(lx, ly):
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y, k=2)(x_smooth)
    return x_smooth, y_smooth


x = [1, 3, 5, 7, 9]
accumulo = [0.645, 0.615, 0.572, 0.566, 0.503]
commonsio = [0.702, 0.632]
cxf = [0.722, 0.634, 0.649, 0.650, 0.643]
druid = [0.708, 0.686, 0.672, 0.637, 0.613]
hive = [0.740, 0.716, 0.742, 0.747, 0.755]
maven = [0.647, 0.679, 0.691, 0.694, 0.692]
pdfbox = [0.682, 0.690, 0.700, 0.680, 0.687]
poi = [0.687, 0.607, 0.516, 0.512, 0.501]
rocketmq = [0.683, 0.692, 0.711, 0.718, 0.725]
tika = [0.677, 0.592, 0.707, 0.712, 0.726]
plt.xlabel('Number of commit windows')
plt.ylabel('AUC')

plt.scatter(x, accumulo, marker='o', alpha=0.5, s=20)
plt.scatter(x[:2], commonsio, marker='o', alpha=0.5, s=20)
plt.scatter(x, cxf, marker='o', alpha=0.5, s=20)
plt.scatter(x, druid, marker='o', alpha=0.5, s=20)
plt.scatter(x, hive, marker='o', alpha=0.5, s=20)
plt.scatter(x, maven, marker='o', alpha=0.5, s=20)
plt.scatter(x, pdfbox, marker='o', alpha=0.5, s=20)
plt.scatter(x, poi, marker='o', alpha=0.5, s=20)
plt.scatter(x, rocketmq, marker='o', alpha=0.5, s=20)
plt.scatter(x, tika, marker='o', alpha=0.5, s=20)

xx, yy = smooth_xy(x, accumulo)
plt.plot(xx, yy, marker='o', markersize=0)
plt.plot(x[:2], commonsio, marker='o', markersize=0)
xx, yy = smooth_xy(x, cxf)
plt.plot(xx, yy, marker='o', markersize=0)
xx, yy = smooth_xy(x, druid)
plt.plot(xx, yy, marker='o', markersize=0)
xx, yy = smooth_xy(x, hive)
plt.plot(xx, yy, marker='o', markersize=0)
xx, yy = smooth_xy(x, maven)
plt.plot(xx, yy, marker='o', markersize=0)
xx, yy = smooth_xy(x, pdfbox)
plt.plot(xx, yy, marker='o', markersize=0)
xx, yy = smooth_xy(x, poi)
plt.plot(xx, yy, marker='o', markersize=0)
xx, yy = smooth_xy(x, rocketmq)
plt.plot(xx, yy, marker='o', markersize=0)
xx, yy = smooth_xy(x, tika)
plt.plot(xx, yy, marker='o', markersize=0)

plt.legend(['Accumulo', 'Commons IO', 'CXF', 'Druid', 'Hive', 'Maven', 'PDFBox', 'POI', 'RocketMQ', 'Tika'],
           loc='center right', frameon=False)
# plt.subplots_adjust(right=0.99)
# plt.subplots_adjust(left=0.079)
# plt.subplots_adjust(bottom=0.086)
# plt.subplots_adjust(top=0.983)
plt.subplots_adjust(left=0.1)
plt.subplots_adjust(right=0.98)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(bottom=0.15)
plt.xticks([1, 3, 5, 7, 9])
plt.xlim([0.7, 11.5])

# plt.show()

plt.savefig("history.eps", format="eps")
