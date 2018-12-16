import os
import json

from argparse import ArgumentParser
from split_utils import split_on_keywords, build_index

def read_nyu_splitfile(filename):
    out = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            out.append(line.strip())
    return out

def main(opt):
    print("Building index from root directory {}".format(opt.data_dir))
    index = build_index(opt.data_dir)
    print("Built index of size {}.".format(len(index)))


    train_split = read_nyu_splitfile(opt.train_splitfile)
    test_split = read_nyu_splitfile(opt.test_splitfile)

    train = split_on_keywords(index, train_split)
    print("Training set: {} points.".format(len(train)))
    test = split_on_keywords(index, test_split)
    print("Testing set: {} points.".format(len(test)))

    # check overlap
    # overlap = set(train.keys()) & set(test.keys())
    # print("overlap: {}".format(overlap))
    # missing = set(index.keys()) - (set(train.keys()) | set(test.keys()))
    # print("Missing: {}".format(missing))

    with open(opt.output_train_file, "w") as f:
        json.dump(train, f)
    print("Wrote train split to {}".format(opt.output_train_file))
    with open(opt.output_test_file, "w") as f:
        json.dump(test, f)
    print("Wrote test split to {}".format(opt.output_test_file))

    info = {}
    for entry in index:
        info[entry] = {"keywords": entry.split(os.path.sep)}
    with open(opt.output_info_file, "w") as f:
        json.dump(info, f)
    print("Wrote info file to {}".format(opt.output_info_file))

if __name__ == '__main__':
    parser = ArgumentParser(description="Generate train-val-test splits.")
    parser.add_argument("--split-only", action="store_true", help="Only generate the splits for the chosen directory.")
    parser.add_argument("--small", action="store_true", help="Generate a small dataset for overfitting.")
    parser.add_argument("--data-dir", default=os.path.join("data", "nyu_depth_v2_processed"))
    parser.add_argument("--train-splitfile", default=os.path.join("data", "nyu_train.txt"))
    parser.add_argument("--test-splitfile", default=os.path.join("data", "nyu_test.txt"))
    # parser.add_argument("--output-dir", default=os.path.join("data", "nyu_depth_v2_processed"))
    parser.add_argument("--output-train-file", default=os.path.join("data", "nyu_depth_v2_processed", "train.json"))
    parser.add_argument("--output-test-file", default=os.path.join("data", "nyu_depth_v2_processed", "test.json"))
    parser.add_argument("--output-info-file", default=os.path.join("data", "nyu_depth_v2_processed", "info.json"))
    opt = parser.parse_args()
    main(opt)
