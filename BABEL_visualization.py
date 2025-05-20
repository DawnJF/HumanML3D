import json
import os
from os.path import join as ospj

import numpy as np
import random
import pprint

from tqdm import tqdm

from amass2pose import amass_to_pose

pp = pprint.PrettyPrinter()


# d_folder = "/Users/majianfei/Downloads/babel_v1.0_release"  # Data folder
d_folder = "/liujinxin/dataset/BABEL/babel_v1.0_release"  # Data folder
l_babel_dense_files = ["train", "val", "test"]
l_babel_extra_files = ["extra_train", "extra_val"]

# BABEL Dataset
babel = {}
for file in l_babel_dense_files:
    babel[file] = json.load(open(ospj(d_folder, file + ".json")))

for file in l_babel_extra_files:
    babel[file] = json.load(open(ospj(d_folder, file + ".json")))


def get_random_babel_ann():
    """Get annotation from random sequence from a random file"""
    file = np.random.choice(l_babel_dense_files + l_babel_extra_files)
    seq_id = np.random.choice(list(babel[file].keys()))
    print(
        'We are visualizing annotations for seq ID: {0} in "{1}.json"'.format(
            seq_id, file
        )
    )
    ann = babel[file][seq_id]
    return ann, file


def get_babel_ann(seq_id):
    for file in l_babel_dense_files + l_babel_extra_files:
        if seq_id in babel[file]:
            print(
                'We are visualizing annotations for seq ID: {0} in "{1}.json"'.format(
                    seq_id, file
                )
            )
            return babel[file][seq_id], file
    print("No such sequence ID in BABEL dataset")
    return None, None


def get_all_file():

    all_file_path = []
    all_file_name = []
    for file in l_babel_dense_files + l_babel_extra_files:

        for item in list(babel[file].values()):
            path = item["feat_p"]
            _, _, path = path.partition("/")  # 第一次出现 "/" 为分割点
            # print(new_path)  # 输出: "MPI_Limits/03099/op9_poses.npz"
            name = path.replace("/", "_")

            all_file_path.append(path)
            all_file_name.append(name)
    return all_file_path, all_file_name


def get_vid_html(url):
    """Helper code to embed a URL in a notebook"""
    html_code = '<div align="middle"><video width="80%" controls>'
    html_code += f'<source src="{url}" type="video/mp4">'
    html_code += "</video></div>"
    return html_code


def get_labels(ann, file):
    # Get sequence labels and frame labels if they exist
    seq_l, frame_l = None, None
    if "extra" not in file:
        if ann["seq_ann"] is not None:
            seq_l = [seg["raw_label"] for seg in ann["seq_ann"]["labels"]]
        if ann["frame_ann"] is not None:
            frame_l = [
                (seg["raw_label"], seg["start_t"], seg["end_t"])
                for seg in ann["frame_ann"]["labels"]
            ]
    else:
        # Load labels from 1st annotator (random) if there are multiple annotators
        if ann["seq_anns"] is not None:
            seq_l = [seg["raw_label"] for seg in ann["seq_anns"][0]["labels"]]
        if ann["frame_anns"] is not None:
            frame_l = [
                (seg["raw_label"], seg["start_t"], seg["end_t"])
                for seg in ann["frame_anns"][0]["labels"]
            ]
    return seq_l, frame_l


def visualize():
    ann, file = get_random_babel_ann()
    # ann, file = get_babel_ann("5475")
    seq_l, frame_l = get_labels(ann, file)
    print("Sequence labels: ", seq_l)
    print("Frame labels: (action label, start time, end time)")
    if frame_l is None:
        frame_l = []
    else:
        frame_l = sorted(frame_l, key=lambda x: x[1])
    for label in frame_l:
        print(label)
    # HTML(get_vid_html(ann["url"]))
    print(ann["url"])


# then, and then, ->,


def generate_action_sequence_text(actions):
    assert isinstance(actions, list), "actions should be a list"
    assert len(actions) <= 3, "actions should be at most 3"

    if len(actions) == 1:
        return actions[0]

    # 拼接模板池（2动作和3动作）
    two_action_templates = [
        "{0} then {1}",
        "{0} and then {1}",
        "{1} after {0}",
        "{0}, then {1}",
        "{0}, next {1}",
        "{0} followed by {1}",
        "{0} subsequently {1}",
    ]

    three_action_templates = [
        "{0}, then {1}, and finally {2}",
        "{0} followed by {1}, then {2}",
        "{2} after {1} after {0}",
        "First {0}, then {1}, finally {2}",
        "{0}, next {1}, and then {2}",
        "{0}, followed by {1}, subsequently {2}",
        "{0}, then {1}, then {2}",
    ]

    if len(actions) == 2:
        template = random.choice(two_action_templates)
        return template.format(actions[0], actions[1])

    elif len(actions) == 3:
        template = random.choice(three_action_templates)
        return template.format(actions[0], actions[1], actions[2])


"""
Sequence labels(seq_ann) 一个单词的都不太准
looking 不要
有一些数据开头是t-pose

proc_label,raw_label
"""


def extract_BABEL_pose():

    ext_BABEL_folder = "/liujinxin/dataset/BABEL/motion"
    amass_data_folder = "/liujinxin/dataset/amass_data"

    file_paths, file_names = get_all_file()
    for file, name in tqdm(zip(file_paths, file_names)):
        save_path = os.path.join(ext_BABEL_folder, name)
        amass_file = os.path.join(amass_data_folder, file)
        if not os.path.exists(amass_file):
            print(f"no file {name}")
            continue

        fps = amass_to_pose(amass_file, save_path)
        if fps == 0:
            print(f"skip {name}")


if __name__ == "__main__":

    # visualize()

    extract_BABEL_pose()
