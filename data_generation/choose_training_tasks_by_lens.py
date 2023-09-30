import json
import numpy as np
import os
import argparse


def sort_tasks_by_lens(tasks):
    dict_keys = list(tasks.keys())
    #print(tasks[dict_keys[0]])
    cleaned_tasks = {}
    for i in dict_keys:
        if tasks[i]['n_ex'] == 300:
            cleaned_tasks[i] = tasks[i]
            """if tasks[i]['mean_len'] == -1:
                print(tasks[i])"""
            #print(tasks[i]['mean_len'])
    #print(tasks)
    sorted_tasks = {k: v for k, v in sorted(cleaned_tasks.items(), key=lambda item: item[1]['mean_len'])}
    return sorted_tasks

def choose_n_tasks(sorted_tasks, n_tasks):
    assert len(sorted_tasks.keys())>=n_tasks
    #print(sorted_tasks)
    print("number of available tasks : "+str(len(list(sorted_tasks.keys()))))
    #sampling_step = len(sorted_tasks)//(n_tasks)
    tasks = np.array(list(sorted_tasks.keys()))
    ids_in_sorted = np.linspace(0, len(sorted_tasks)-1, n_tasks).astype(int)#np.arange(0, len(sorted_tasks)+sampling_step, sampling_step)#
    print(ids_in_sorted)
    print(tasks)
    tasks_ids = tasks[ids_in_sorted]
    #print(tasks_ids)
    for id in tasks_ids:
        print(sorted_tasks[str(id)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str, help='path to demos_dataset. ex: ./my_dataset/', required=True)
    parser.add_argument('--n_tasks', type=str, help='number of training tasks', required=True)
    args = parser.parse_args()
    tasks_lens = json.load(open(args.path_dataset+'tasks_lenght_stats.json', 'r'))
    sorted_tasks = sort_tasks_by_lens(tasks_lens)
    choose_n_tasks(sorted_tasks, int(args.n_tasks))

