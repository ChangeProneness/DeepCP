import torch
import csv
import random
import numpy as np


def to_tensor(data):
    data = torch.from_numpy(data).type(torch.float32)
    data = data.unsqueeze(0)
    return data


def outfile(file, headers, rows, n):
    if n == 0:
        f = open(file, 'a+', newline='')
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)
    else:
        f = open(file, 'a+', newline='')
        f_csv = csv.writer(f)
        # f_csv.writerow(headers)
        f_csv.writerows(rows)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
