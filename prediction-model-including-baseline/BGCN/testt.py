import argparse

parser = argparse.ArgumentParser(
        description="bpr recommender",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
parser.add_argument(
    "--avg_items",
    type=bool,
    default=False,
    dest="avg_items",
    help="avg_items",
)
parser.add_argument(
    "--train_only",
    default=False,
    dest="train_only",
    action='store_true'
)
parser.add_argument(
    "--dataset_string",
    type=str,
    default="Steam",
    dest="dataset_string",
    help="dataset_string, should be Steam, Youshu, or NetEase"
)

parser.add_argument(
    "--model_to_load",
    type=str,
    default="",
    dest="model_to_load",
    help="model_to_load",
)
parser.add_argument(
    "--use_graph_sampling",
    type=bool,
    default=False,
    dest="use_graph_sampling",
    help="use_graph_sampling",
)
parser.add_argument(
    "--size",
    type=int,
    default="64",
    dest="size",
    help="size of each item/bundle/user vector",
)

args = parser.parse_args()

# print(type(args))
print(args)
args = argparse.Namespace(verbose=False, verbose_1=False)
