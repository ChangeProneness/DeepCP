import pydriller
from Step1_evo_metrics import read_commits
from Step0_utils import repo_path, result_path, project_name, exec_command, hash_interval


def process_one_project():
    repo = pydriller.Git(repo_path)
    all_commit_hashes = read_commits()

    for commit_hash_index in range(hash_interval - 1, len(all_commit_hashes), hash_interval):
        commit_hash = all_commit_hashes[commit_hash_index]
        repo.checkout(commit_hash)

        command = 'pmd ' \
              'check ' \
              '-d ' + repo_path + \
              ' ' \
              '-R quickstart.xml ' \
              '-f csv ' \
              '-t 16 ' \
              '--report-file ' + result_path + commit_hash + '_pmd_ori.csv'

        exec_command(command)


def main():
    process_one_project()


if __name__ == '__main__':
    main()
