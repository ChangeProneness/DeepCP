import pandas as pd
from collections import defaultdict

df = pd.read_csv('./GenerateTableFigure/deepcp_result.csv')
grouped = df.groupby(['project'])

num_num = defaultdict(int)

proj_top_num = defaultdict(lambda :defaultdict(int))
for i in range(1, 6, 4):
    print()
    print(i)
    for project, other in grouped:
        df_descending = other.sort_values(by='auc', ascending=False)
        head_rows = df_descending.head(i)
        temp = head_rows['dep_num'].value_counts().sort_index()
        one_row = [project[0]]
        for j in range(6):
            proj_top_num[project][j] += 1
            if temp.__contains__(j):
                one_row += [temp[j]]
            else:
                one_row += [0]
        if sum(one_row[3:]) > one_row[2]:
            num_num[i] += 1
            print('\t'.join([str(x) for x in one_row]))
        else:
            print('\t'.join([str(x) for x in one_row]))

