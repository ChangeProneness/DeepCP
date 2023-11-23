import os.path
import subprocess

hash_interval = 500
project_name = 'druid'
repo_path = "Documents/Proneness/repos/" + project_name + "/"
result_path = "Documents/Proneness/raw_metrics/" + project_name + "/" + str(hash_interval) + '/'
if not os.path.exists(result_path):
    os.makedirs(result_path)


def read_file(filename):
    encodings = ['utf-8', 'latin-1', 'gbk', 'utf-16']
    for encoding in encodings:
        try:
            with open(filename, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            pass
    return None


def exec_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(result.stderr)


def read_commits():
    # commit_list = read_commits('cd ' + repoPath + '; git rev-list --no-merges TRUNK --date-order --reverse')
    with open('CommitList/' + project_name + '_commits') as f:
        all_commits = f.read().strip().split('\n')
    return all_commits[: len(all_commits) // hash_interval * hash_interval]
