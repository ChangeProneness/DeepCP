import pandas as pd

from Step0_utils import result_path, read_commits, hash_interval


def process_class_ck(hash_value):
    csv_file_path = result_path + hash_value + '_ck_class.csv'
    df = pd.read_csv(csv_file_path)
    df = df.fillna(0)
    df = df.drop('class', axis=1)
    df = df.drop('type', axis=1)
    df = df.add_prefix('class_')
    df = df.groupby('class_file')
    df = df.sum()
    return df

def process_pmd(hash_value):
    csv_file_path = result_path + hash_value + '_pmd_metrics.csv'
    df = pd.read_csv(csv_file_path)
    df = df.fillna(0)
    df = df.add_prefix('pmd_')
    df = df.groupby('pmd_File')
    df = df.sum()
    return df


def process_evo(hash_value):
    csv_file_path = result_path + hash_value + '_evo_metrics.csv'
    df = pd.read_csv(csv_file_path)
    df = df.fillna(0)
    df = df.drop('release_hash', axis=1)
    df = df.replace(True, 1)
    df = df.replace(False, 0)

    df = df.add_prefix('evo_')
    df = df.groupby('evo_file')
    df = df.sum()
    return df


def get_labels(files, next_hash_value):
    res = {}
    evo_stat = process_evo(str(next_hash_value))
    for file in files:
        if file in evo_stat.index:
            res[file] = evo_stat.loc[file]['evo_file_commit_num']
        else:
            res[file] = -1
    return res

def main():
    all_commit_hashes = read_commits()

    for commit_hash_index in range(2 * hash_interval - 1, len(all_commit_hashes), hash_interval):
        if commit_hash_index + hash_interval > len(all_commit_hashes):
            break

        hash_value = all_commit_hashes[commit_hash_index]
        class_ck = process_class_ck(hash_value)
        smell_stat = process_pmd(hash_value)
        evo_stat = process_evo(hash_value)

        files = set(class_ck.index.to_list()).intersection(evo_stat.index.to_list())
        class_cols = class_ck.keys().to_list()
        class_cols.remove('class_tcc')
        class_cols.remove('class_lcc')
        smell_cols = smell_stat.keys().to_list()
        evo_cols = evo_stat.keys().to_list()
        evo_cols.remove('evo_file_commit_num')
        evo_cols.remove('evo_release_index')
        evo_cols = ['evo_release_index', 'evo_file_commit_num'] + evo_cols

        evo_stat_labels = get_labels(files, all_commit_hashes[commit_hash_index + hash_interval])
        files = evo_stat_labels.keys()

        header = ['file', 'label_evo_file_commit_num'] + evo_cols + class_cols + smell_cols
        import csv
        with open(result_path + str(hash_value) + '_all_metrics.csv', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
            for file in files:
                one_row = [file, evo_stat_labels[file]]
                one_row += [evo_stat.loc[file][x] for x in evo_cols]
                one_row += [class_ck.loc[file][x] for x in class_cols]
                if file in smell_stat.index:
                    one_row += [smell_stat.loc[file][x] for x in smell_cols]
                else:
                    one_row += [0 for _ in smell_cols]
                csv_writer.writerow(one_row)


if __name__ == '__main__':
    main()
