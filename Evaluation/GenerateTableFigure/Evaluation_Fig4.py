from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

f = open('../ExperimentResults/cnn_result')
lines = f.read().strip().split('\n')
proj_cnn_auc = defaultdict(list)
for line in lines:
    temp = line.split(' ')
    proj, auc = temp[1], temp[2]
    proj_cnn_auc[proj].append(float(auc))

proj_dep_aucs = defaultdict(lambda: defaultdict(list))
df = pd.read_csv('../ExperimentResults/deepcp_result.csv')
grouped = df.groupby(['project'])
for project, other in grouped:
    dep_num_auc = other.groupby('dep_num')
    for xxx, yyy in dep_num_auc:
        proj_dep_aucs[project[0]][xxx] = yyy['auc'].to_list()

my_res = []
cnn_res = []
for project in proj_cnn_auc.keys():
    print(project, end='\t')
    group1 = proj_dep_aucs[project][0]
    group2 = proj_cnn_auc[project]
    my_res.append(group1)
    cnn_res.append(group2)


fig, ax = plt.subplots()
flierprops = dict(marker='o', markerfacecolor='gray', markersize=3, linestyle='none', markeredgecolor='gray',
                  markeredgewidth=0, alpha=0.2)

boxplot = ax.boxplot([my_res[0], cnn_res[0]], positions=[1, 2], patch_artist=True, widths=0.65, flierprops=flierprops)
boxplot['boxes'][0].set_facecolor('#C7DDEC')
boxplot['boxes'][1].set_facecolor('#CBE7CB')
boxplot = ax.boxplot([my_res[1], cnn_res[1]], positions=[4, 5], patch_artist=True, widths=0.65, flierprops=flierprops)
boxplot['boxes'][0].set_facecolor('#C7DDEC')
boxplot['boxes'][1].set_facecolor('#CBE7CB')
boxplot = ax.boxplot([my_res[2], cnn_res[2]], positions=[7, 8], patch_artist=True, widths=0.65, flierprops=flierprops)
boxplot['boxes'][0].set_facecolor('#C7DDEC')
boxplot['boxes'][1].set_facecolor('#CBE7CB')
boxplot = ax.boxplot([my_res[3], cnn_res[3]], positions=[10, 11], patch_artist=True, widths=0.65, flierprops=flierprops)
boxplot['boxes'][0].set_facecolor('#C7DDEC')
boxplot['boxes'][1].set_facecolor('#CBE7CB')
boxplot = ax.boxplot([my_res[4], cnn_res[4]], positions=[13, 14], patch_artist=True, widths=0.65, flierprops=flierprops)
boxplot['boxes'][0].set_facecolor('#C7DDEC')
boxplot['boxes'][1].set_facecolor('#CBE7CB')
boxplot = ax.boxplot([my_res[5], cnn_res[5]], positions=[16, 17], patch_artist=True, widths=0.65, flierprops=flierprops)
boxplot['boxes'][0].set_facecolor('#C7DDEC')
boxplot['boxes'][1].set_facecolor('#CBE7CB')
boxplot = ax.boxplot([my_res[6], cnn_res[6]], positions=[19, 20], patch_artist=True, widths=0.65, flierprops=flierprops)
boxplot['boxes'][0].set_facecolor('#C7DDEC')
boxplot['boxes'][1].set_facecolor('#CBE7CB')
boxplot = ax.boxplot([my_res[7], cnn_res[7]], positions=[22, 23], patch_artist=True, widths=0.65, flierprops=flierprops)
boxplot['boxes'][0].set_facecolor('#C7DDEC')
boxplot['boxes'][1].set_facecolor('#CBE7CB')
boxplot = ax.boxplot([my_res[8], cnn_res[8]], positions=[25, 26], patch_artist=True, widths=0.65, flierprops=flierprops)
boxplot['boxes'][0].set_facecolor('#C7DDEC')
boxplot['boxes'][1].set_facecolor('#CBE7CB')
boxplot = ax.boxplot([my_res[9], cnn_res[9]], positions=[28, 29], patch_artist=True, widths=0.65, flierprops=flierprops)
boxplot['boxes'][0].set_facecolor('#C7DDEC')
boxplot['boxes'][1].set_facecolor('#CBE7CB')
ax.legend([boxplot['boxes'][0], boxplot['boxes'][1]], ['DeepCP', 'CNN'], loc='upper right')

ax.set_xticks([x * 3 + 1.5 for x in range(10)])
ax.set_xticklabels(['Accumulo', 'Commons IO', 'CXF', 'Druid', 'Hive', 'Maven', 'PDFBox', 'POI', 'RocketMQ', 'Tika'],
                   rotation=20)
ax.set_xlabel("Projects")
ax.set_ylabel("AUC")
plt.subplots_adjust(left=0.1)
plt.subplots_adjust(right=0.98)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(bottom=0.15)

plt.savefig("fig1.eps", format="eps")

plt.show()
