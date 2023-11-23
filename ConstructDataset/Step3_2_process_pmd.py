# /pmd check -d ~/Downloads/tomcat -R quickstart.xml -f csv -t 16 --report-file a.csv
import pandas as pd
from collections import defaultdict
import csv
import os

from ConstructDataset.Step0_utils import result_path


def main():
    for root, dirs, files in os.walk(result_path):
        for file in files:
            file_path = os.path.join(root, file)
            if '_pmd_ori.csv' in file_path:
                rule_set_cols = ["Best Practices", "Code Style", "Design", "Documentation", "Error Prone", "Multithreading", "Performance", "Security"]
                priority_cols = ['1', '2', '3', '4', '5']
                rule_set_cols = ['Rule_Set_' + x for x in rule_set_cols]
                priority_cols = ['Priority_' + x for x in priority_cols]
                columns = ['File', 'Problem_Num'] + priority_cols + rule_set_cols

                file_metric_num = defaultdict(lambda: defaultdict(int))

                df = pd.read_csv(file_path)
                for index, row in df.iterrows():
                    file_metric_num[row['File']]['Priority_' + str(row['Priority'])] += 1
                    file_metric_num[row['File']]['Rule_Set_' + str(row['Rule set'])] += 1
                    file_metric_num[row['File']]['Problem_Num'] += 1

                with open(os.path.join(root, file.split('_')[0] + '_pmd_metrics.csv'), 'w') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(columns)
                    for file, metric_num in file_metric_num.items():
                        one_row = [file]
                        one_row += [file_metric_num[file][metric] for metric in columns[1:]]
                        csv_writer.writerow(one_row)

if __name__ == '__main__':
    main()