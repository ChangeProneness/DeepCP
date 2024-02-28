from Step0_utils import read_commits, hash_interval, result_path
import pandas as pd

def main():
    import csv
    all_commit_hashes = read_commits()
    depend_num = 5
    for commit_hash_index in range(2 * hash_interval - 1, len(all_commit_hashes), hash_interval):
        if commit_hash_index + hash_interval > len(all_commit_hashes):
            break
        hash_value = all_commit_hashes[commit_hash_index]

        with open(result_path + str(hash_value) + '_all_dep_metrics.csv', 'w') as f:
            csv_writer = csv.writer(f)
            df = pd.read_csv(result_path + all_commit_hashes[commit_hash_index] + '_all_metrics.csv')
            columns = df.columns.tolist()
            headers = columns[:]
            for i in range(depend_num):
                headers += ['dep' + str(i) + '_' + x for x in columns[3:]]
            csv_writer.writerow(headers)

            dep = pd.read_csv(result_path + 'understands/' + str(all_commit_hashes[commit_hash_index]) + '/depends.csv')
            for index, row in df.iterrows():
                if row['label_evo_file_commit_num'] < 0:
                    continue
                cur_line = [row['file'], row['label_evo_file_commit_num'], row['evo_release_index']]
                metrics = row.iloc[3:].to_list()
                cur_line += metrics
                depends_lines = dep[dep['From File'] == row['file']]
                depends_lines = depends_lines.sort_values(by='To Entities', ascending=False)
                depend_files = depends_lines['To File'].tolist()
                for i in range(depend_num):
                    if i >= len(depend_files):
                        cur_line += [0] * len(metrics)
                    else:
                        depend = depend_files[i]
                        try:
                            one_depend_features = df[df['file'] == depend].iloc[:, 3:]
                            cur_line += one_depend_features.iloc[0].to_list()
                        except:
                            cur_line += [0] * len(metrics)
                csv_writer.writerow(cur_line)



if __name__ == '__main__':
    main()
