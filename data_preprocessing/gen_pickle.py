import os
import pickle
import argparse


def collect_pkl(root, out_path, part):
    """

    Args:
        root: media/Data/enz/hand_bbdd/Briareo + task(IR o RGB) + part
        out_path:
        part: train, test or val
    Returns:

    """
    labels_str = ['g00', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11']
    labels_ids = list(range(0, len(labels_str)))  
    labels_dict = dict(zip(labels_str, labels_ids))

    content = []
    for dir, subdirs, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1] == '.pkl':
                with open(os.path.join(dir, f), 'rb') as file:
                    ann = pickle.load(file)
                    ann['label'] = labels_dict.get(ann['label'])
                    content.append(ann)
    # briareo/task/part.pkl; ej: briareo/IR/train.pkl
    output_path = '{}/{}.pkl'.format(out_path, part)
    with open(output_path, 'wb') as fp:
        pickle.dump(content, fp)


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script collect all the annotation files into one list and save to a single file ')
    parser.add_argument('--dir', type=str, default=None,
                        help='path to ann files directory')
    parser.add_argument('--output', type=str, default='data/briareo', help='path to output ann file')
    parser.add_argument('--task', type=str, default='RGB')
    parser.add_argument('--part', type=str, default='train', help='part of the dataset: train val o test')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)
    task_pth = os.path.join(args.output, args.task)
    os.makedirs(task_pth, exist_ok=True)
    collect_pkl(root=args.dir,
                out_path=task_pth,
                part=args.part)
