import os.path

import pydriller
from Step1_evo_metrics import read_commits
from Step0_utils import repo_path, result_path, project_name, exec_command, hash_interval


def executeUnd(hash_value):
    if os.path.exists(result_path + "understands/" + hash_value + "/depends.csv"):
        return
    commands = ["und -db " + result_path + "understands/" + hash_value + "/understand.und",
                "create -languages Java -gitcommit " + hash_value + " -gitrepo " + repo_path,
                "add " + repo_path,
                "analyze -errors",
                "export -dependencies -format long file csv " + result_path + "understands/" + hash_value + "/depends.csv"]
    command = " ".join(commands)
    print(command)
    exec_command(command)


def process_one_project():
    repo = pydriller.Git(repo_path)
    all_commit_hashes = read_commits()

    for commit_hash_index in range(hash_interval - 1, len(all_commit_hashes), hash_interval):
        commit_hash = all_commit_hashes[commit_hash_index]
        repo.checkout(commit_hash)
        executeUnd(commit_hash)


def main():
    process_one_project()


if __name__ == '__main__':
    main()
