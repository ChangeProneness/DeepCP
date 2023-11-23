import pandas as pd
from collections import defaultdict
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta


def cliff(group1, group2):
    print("{:.3f}".format(sum(group1) / len(group1)), "{:.3f}".format(sum(group2) / len(group2)), sep='\t', end='\t')
    stat, p_value = mannwhitneyu(group1, group2)
    if p_value < 0.05:
        print('<0.05', end='\t')
    else:
        print("{:.3f}".format(p_value), end='\t')
    delta, size = cliffs_delta(group1, group2)
    print(('+' if delta > 0 else '-') + size, sep='\t')


f = open('./GenerateTableFigure/cnn_result')
lines = f.read().strip().split('\n')
proj_cnn_auc = defaultdict(list)
for line in lines:
    temp = line.split(' ')
    proj, auc = temp[1], temp[2]
    proj_cnn_auc[proj].append(float(auc))

proj_dep_aucs = defaultdict(lambda: defaultdict(list))
df = pd.read_csv('./GenerateTableFigure/deepcp_result.csv')
grouped = df.groupby(['project'])
for project, other in grouped:
    dep_num_auc = other.groupby('dep_num')
    for xxx, yyy in dep_num_auc:
        proj_dep_aucs[project[0]][xxx] = yyy['auc'].to_list()

for project in proj_cnn_auc.keys():
    print(project, end='\t')
    group1 = proj_dep_aucs[project][0]
    group2 = proj_cnn_auc[project]
    cliff(group1, group2)
