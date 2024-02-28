import pandas as pd

projects = ['accumulo', 'commons-io', 'cxf', 'druid', 'hive', 'maven', 'pdfbox', 'poi', 'rocketmq', 'tika']
for p in projects:
    df = pd.read_csv('Dataset/' + p + '_500_concatenated.csv')
    min_release = int(df['evo_release_index'].min())
    max_release = int(df['evo_release_index'].max())

    grouped = df.groupby('evo_release_index')
    keys = df['evo_release_index'].unique()

    ratios = []
    files = []
    for release in keys:
        min_df = grouped.get_group(release)
        median = min_df['label_evo_file_commit_num'].median()
        y = min_df['label_evo_file_commit_num'].apply(lambda x: 1 if x > median else 0)
        files.append(min_df.shape[0])
        ratios.append(sum(y) / y.shape[0])
    print(p, sum(files)/len(files), sum(ratios)/len(ratios), sep='\t')

