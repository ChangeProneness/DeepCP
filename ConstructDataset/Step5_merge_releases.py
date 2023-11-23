from Step0_utils import read_commits, hash_interval, result_path, project_name
import pandas as pd

def main():
    all_commit_hashes = read_commits()
    concatenated_df = pd.DataFrame()
    for commit_hash_index in range(2 * hash_interval - 1, len(all_commit_hashes), hash_interval):
        if commit_hash_index + hash_interval > len(all_commit_hashes):
            break
        df = pd.read_csv(result_path + all_commit_hashes[commit_hash_index] + '_all_dep_metrics.csv')
        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)

    a = concatenated_df.groupby('file')['evo_file_commit_num'].sum()
    concatenated_df = concatenated_df[concatenated_df['file'].isin(a[a > 5].index.tolist())]
    concatenated_df.to_csv('Dataset/' + project_name + '_' + str(hash_interval) + '_concatenated.csv', index=False)


if __name__ == '__main__':
    main()
