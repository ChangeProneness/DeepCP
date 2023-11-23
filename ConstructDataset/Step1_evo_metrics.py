import lizard
import pydriller
from collections import defaultdict
import diff
from Step0_utils import repo_path, result_path, hash_interval, read_commits, read_file


def calc_base():
    tag_file_content = defaultdict(lambda: defaultdict(str))
    tag_file_boc = defaultdict(lambda: defaultdict(int))
    tag_file_tach = defaultdict(lambda: defaultdict(int))
    tag_file_fch = defaultdict(lambda: defaultdict(int))
    tag_file_lch = defaultdict(lambda: defaultdict(int))
    tag_file_frch = defaultdict(lambda: defaultdict(int))
    tag_file_loc = defaultdict(lambda: defaultdict(int))
    tag_file_commit = defaultdict(lambda: defaultdict(int))

    repo = pydriller.Git(repo_path)

    all_commit_hashs = read_commits()

    file2boc = {}
    file2fch = {}
    file2lch = {}
    file2freq = defaultdict(int)

    repo.checkout(all_commit_hashs[hash_interval - 1])
    files = [x for x in repo.files() if x.endswith('.java')]
    for x in files:
        tag_file_tach[1][x] = 0
        tag_file_fch[1][x] = 0
        tag_file_lch[1][x] = 0
        tag_file_frch[1][x] = 0
        file2boc[x] = 1
        tag_file_boc[1][x] = file2boc[x]
        tag_file_content[1][x] = read_file(x)
        temp = lizard.analyze_file(x).nloc
        if temp == 0:
            print(0)
        tag_file_loc[1][x] = temp if temp > 0 else 1
        tag_file_commit[1][x] = 0

    for end_commit_hash_index in range(2 * hash_interval - 1, len(all_commit_hashs), hash_interval):
        start_commit_hash_index = end_commit_hash_index - hash_interval + 1
        release_num = start_commit_hash_index // hash_interval + 1
        file2commits = defaultdict(int)
        file2total_lines = defaultdict(int)
        print(release_num, len(all_commit_hashs) // hash_interval,
              start_commit_hash_index, end_commit_hash_index,
              all_commit_hashs[start_commit_hash_index], all_commit_hashs[end_commit_hash_index],
              sep='\t')
        for commit_hash in all_commit_hashs[start_commit_hash_index: end_commit_hash_index + 1]:
            commit = repo.get_commit(commit_hash)
            modified_files = [x for x in commit.modified_files if x.filename.endswith('.java')]
            for modified_file in modified_files:
                if modified_file.old_path == modified_file.new_path:
                    file_path = repo_path + modified_file.old_path
                    if file_path not in file2fch:
                        file2fch[file_path] = release_num
                    file2lch[file_path] = release_num
                    file2freq[file_path] += 1
                    file2total_lines[file_path] += (modified_file.added_lines + modified_file.deleted_lines)
                    file2commits[file_path] += 1

        repo.checkout(all_commit_hashs[end_commit_hash_index])
        files = [x for x in repo.files() if x.endswith('.java')]
        for x in files:
            if x not in file2boc:
                file2boc[x] = release_num
            tag_file_boc[release_num][x] = file2boc[x]
            tag_file_tach[release_num][x] = file2total_lines[x]
            tag_file_fch[release_num][x] = file2fch[x] if x in file2fch.keys() else 0
            tag_file_lch[release_num][x] = file2lch[x] if x in file2lch.keys() else 0
            tag_file_frch[release_num][x] = file2freq[x]
            tag_file_content[release_num][x] = read_file(x)
            temp = lizard.analyze_file(x).nloc
            if temp == 0:
                print(0)
            tag_file_loc[release_num][x] = temp if temp > 0 else 1
            tag_file_commit[release_num][x] = file2commits[x]
    return tag_file_content, tag_file_boc, tag_file_tach, tag_file_fch, \
           tag_file_lch, tag_file_frch, tag_file_loc, tag_file_commit, all_commit_hashs


def calc_cho(tag_file_tach):
    tag_file_cho = defaultdict(lambda: defaultdict(int))
    for tag_index, file_metric in tag_file_tach.items():
        for file in file_metric.keys():
            tag_file_cho[tag_index][file] = 1 if tag_file_tach[tag_index][file] > 0 else 0
    return tag_file_cho


def calc_chd(tag_file_tach, tag_file_loc):
    tag_file_chd = defaultdict(lambda: defaultdict(int))
    for tag_index, file_metric in tag_file_tach.items():
        for file in file_metric.keys():
            tag_file_chd[tag_index][file] = tag_file_tach[tag_index][file] / tag_file_loc[tag_index][file]
    return tag_file_chd


def calc_wch(tag_file_tach, tag_file_boc):
    tag_file_wch = defaultdict(lambda: defaultdict(int))
    for tag_index, file_metric in tag_file_tach.items():
        for file in file_metric.keys():
            n = tag_index
            tag_file_wch[tag_index][file] = 0
            j = tag_file_boc[tag_index][file]
            for r in range(j + 1, n + 1):
                if file in tag_file_tach[r]:
                    tag_file_wch[tag_index][file] += tag_file_tach[r][file] * (2 ** (r - n))
    return tag_file_wch


def calc_wcd(tag_file_chd, tag_file_boc):
    tag_file_wcd = defaultdict(lambda: defaultdict(int))
    for tag_index, file_metric in tag_file_chd.items():
        for file in file_metric.keys():
            n = tag_index
            tag_file_wcd[tag_index][file] = 0
            j = tag_file_boc[tag_index][file]
            for r in range(j + 1, n + 1):
                if file in tag_file_chd[r]:
                    tag_file_wcd[tag_index][file] += tag_file_chd[r][file] * (2 ** (r - n))
    return tag_file_wcd


def calc_wfr(tag_file_cho):
    tag_file_wfr = defaultdict(lambda: defaultdict(int))
    for tag_index, file_metric in tag_file_cho.items():
        for file in file_metric.keys():
            n = tag_index
            tag_file_wfr[tag_index][file] = 0
            for r in range(2, n + 1):
                if file in tag_file_cho[r]:
                    tag_file_wfr[tag_index][file] += (tag_file_cho[r][file] * (r - 1))
    return tag_file_wfr


def calc_ataf(tag_file_tach, tag_file_frch):
    tag_file_ataf = defaultdict(lambda: defaultdict(int))
    for tag_index, file_metric in tag_file_tach.items():
        for file in file_metric.keys():
            n = tag_index
            if tag_file_frch[n][file] > 0:
                for r in range(2, n + 1):
                    if file in tag_file_tach[r]:
                        tag_file_ataf[tag_index][file] += tag_file_tach[r][file]
                tag_file_ataf[tag_index][file] /= tag_file_frch[tag_index][file]
            else:
                tag_file_ataf[tag_index][file] = 0
    return tag_file_ataf


def calc_lca(tag_file_tach, tag_file_lch):
    tag_file_lca = defaultdict(lambda: defaultdict(int))
    for tag_index, file_metric in tag_file_tach.items():
        for file in file_metric.keys():
            n = tag_index
            if tag_file_lch[n][file] == 0 or tag_file_lch[n][file] == n:
                tag_file_lca[tag_index][file] = tag_file_tach[tag_index][file]
            else:
                r = tag_file_lch[n][file]
                if file in tag_file_tach[r]:
                    tag_file_lca[tag_index][file] = tag_file_tach[r][file]
                else:
                    tag_file_lca[tag_index][file] = tag_file_tach[tag_index][file]

    return tag_file_lca


def calc_lcd(tag_file_chd, tag_file_lch):
    tag_file_lcd = defaultdict(lambda: defaultdict(int))
    for tag_index, file_metric in tag_file_chd.items():
        for file in file_metric.keys():
            n = tag_index
            if tag_file_lch[n][file] == 0 or tag_file_lch[n][file] == n:
                tag_file_lcd[tag_index][file] = tag_file_chd[tag_index][file]
            else:
                r = tag_file_lch[n][file]
                if file in tag_file_chd[r]:
                    tag_file_lcd[tag_index][file] = tag_file_chd[r][file]
                else:
                    tag_file_lcd[tag_index][file] = tag_file_chd[tag_index][file]
    return tag_file_lcd


def calc_csb(tag_file_content, tag_file_boc):
    tag_file_csb = defaultdict(lambda: defaultdict(int))
    for tag_index, file_metric in tag_file_boc.items():
        for file in file_metric.keys():
            boc = tag_file_boc[tag_index][file]
            if tag_index <= boc:
                tag_file_csb[tag_index][file] = 0
            else:
                start_version = tag_file_content[boc][file]
                current_version = tag_file_content[tag_index][file]
                diff_content = diff.diff(start_version, current_version)
                if diff_content is None:
                    modified_line_num = 0
                else:
                    modified_line_num = sum(1 for line in diff_content.explain().split('\n') if (len(line) > 0 and (line[0] == '+' or line[0] == '-')))
                tag_file_csb[tag_index][file] = modified_line_num

    return tag_file_csb


def calc_csbs(tag_file_csb, tag_file_loc, tag_file_boc):
    tag_file_csbs = defaultdict(lambda: defaultdict(int))
    for tag_index, file_metric in tag_file_csb.items():
        for file in file_metric.keys():
            boc = tag_file_boc[tag_index][file]
            if tag_index <= boc:
                tag_file_csbs[tag_index][file] = 0
            else:
                tag_file_csbs[tag_index][file] = tag_file_csb[tag_index][file] / tag_file_loc[boc][file]

    return tag_file_csbs


def calc_acdf(tag_file_chd, tag_file_frch):
    tag_file_acdf = defaultdict(lambda: defaultdict(int))
    for tag_index, file_metric in tag_file_chd.items():
        for file in file_metric.keys():
            n = tag_index
            if tag_file_frch[n][file] > 0:
                for r in range(2, n + 1):
                    if file in tag_file_chd[r]:
                        tag_file_acdf[tag_index][file] += tag_file_chd[r][file]
            else:
                tag_file_acdf[tag_index][file] = 0
    return tag_file_acdf


def write_csv(metrics, all_commit_hashs):
    import csv
    headers = 'release_index', 'release_hash', 'file', 'file_commit_num', 'tach', 'boc', 'fch', 'lch', 'frch', 'loc', 'cho', 'chd', 'wch', 'wcd', 'wfr', 'ataf', 'lca', 'lcd', 'csb', 'csbs'
    metric = metrics[0]
    tag2files = {}
    for release_num, file_metric in metric.items():
        files = list(file_metric.keys())
        files.sort()
        tag2files[release_num] = files
    for release_num, files in tag2files.items():
        hash_value = all_commit_hashs[release_num * hash_interval - 1]
        with open(result_path + hash_value + '_evo_metrics.csv', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(headers)
            for file in files:
                temp = [release_num, hash_value, file] + [metric[release_num][file] for metric in metrics[1:]]
                temp = [str(x) for x in temp]
                csv_writer.writerow(temp)


def main():
    tag_file_content, tag_file_boc, tag_file_tach, tag_file_fch, \
    tag_file_lch, tag_file_frch, tag_file_loc, tag_file_commit, all_commit_hashs = calc_base()

    print("calc_cho")
    tag_file_cho = calc_cho(tag_file_tach)
    print("calc_chd")
    tag_file_chd = calc_chd(tag_file_tach, tag_file_loc)
    print("calc_wch")
    tag_file_wch = calc_wch(tag_file_tach, tag_file_boc)
    print("calc_wcd")
    tag_file_wcd = calc_wcd(tag_file_chd, tag_file_boc)
    print("calc_wfr")
    tag_file_wfr = calc_wfr(tag_file_cho)
    print("calc_ataf")
    tag_file_ataf = calc_ataf(tag_file_tach, tag_file_frch)
    print("calc_lca")
    tag_file_lca = calc_lca(tag_file_tach, tag_file_lch)
    print("calc_lcd")
    tag_file_lcd = calc_lcd(tag_file_chd, tag_file_lch)
    print("calc_csb")
    tag_file_csb = calc_csb(tag_file_content, tag_file_boc)
    print("calc_csbs")
    tag_file_csbs = calc_csbs(tag_file_csb, tag_file_loc, tag_file_boc)

    metric_list = [tag_file_content, tag_file_commit, tag_file_tach, tag_file_boc, tag_file_fch, tag_file_lch,
                   tag_file_frch, tag_file_loc, tag_file_cho, tag_file_chd, tag_file_wch, tag_file_wcd, tag_file_wfr,
                   tag_file_ataf, tag_file_lca, tag_file_lcd, tag_file_csb, tag_file_csbs]

    print('write')
    write_csv(metric_list, all_commit_hashs)


if __name__ == '__main__':
    main()
    print('END')
