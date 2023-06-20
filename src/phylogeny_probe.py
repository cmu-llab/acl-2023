import torch
import numpy as np
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from matplotlib import pyplot as plt
import matplotlib
from pathlib import Path
import os
import argparse


def get_newick(node, parent_dist, leaf_names, newick='') -> str:
    """
    Source: https://stackoverflow.com/questions/28222179/save-dendrogram-to-newick-format
    Convert sciply.cluster.hierarchy.to_tree()-output to Newick format.

    :param node: output of sciply.cluster.hierarchy.to_tree()
    :param parent_dist: output of sciply.cluster.hierarchy.to_tree().dist
    :param leaf_names: list of leaf names
    :param newick: leave empty, this variable is used in recursion.
    :returns: tree in Newick format
    """
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parent_dist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parent_dist - node.dist, newick)
        else:
            newick = ");"
        newick = get_newick(node.get_left(), node.dist, leaf_names, newick=newick)
        newick = get_newick(node.get_right(), node.dist, leaf_names, newick=",%s" % (newick))
        newick = "(%s" % (newick)
        return newick


def dendrogram_to_newick(linkage_matrix, leaf_names):
    tree = to_tree(linkage_matrix, False)
    return get_newick(tree, tree.dist, leaf_names)


def get_lang_embeddings(model, checkpoint):
    if model == "rnn":
        dialect_embeddings = checkpoint['model']['embeddings.lang_embeddings.weight']
        # get rid of <unk>, <pad>, <bos>, <eos>, <*>, <:>, the protolang, and the separators lang
        dialect_embeddings = dialect_embeddings[6:-2, ]
        num_dialects, emb_size = dialect_embeddings.size()
        print(checkpoint['dialect_vocab'].i2v)

        dialects = []
        for n in range(num_dialects):
            dialects.append(checkpoint['dialect_vocab'].i2v[n + 6])
    elif model == "transformer":
        dialect_embeddings = checkpoint['model']['_dialect_embedding._embedding.weight']
        # get rid of <unk>, <pad>, <bos>, <eos>
        dialect_embeddings = dialect_embeddings[4:, ]
        num_dialects, emb_size = dialect_embeddings.size()
        print(checkpoint['dialect_vocab'].i2v)

        dialects = []
        for n in range(num_dialects):
            dialects.append(checkpoint['dialect_vocab'].i2v[n + 4])
    else:
        raise Exception(f'model {model} not supported')

    print(dialects)
    return dialect_embeddings, dialects, num_dialects, emb_size


def generate_distance_matrix(model, model_path):
    saved_info = torch.load(model_path, map_location=torch.device('cpu'))
    dialect_embeddings, dialects, num_dialects, emb_size = get_lang_embeddings(model, saved_info)

    distance_matrix = np.zeros((num_dialects, num_dialects))
    for row in range(num_dialects):
        for col in range(num_dialects):
            distance_matrix[row][col] = cosine(dialect_embeddings[row], dialect_embeddings[col])
    return distance_matrix, dialects


if __name__ == '__main__':
    # load environment variables
    WORK_DIR = os.environ.get('WORK_DIR')
    DATA_DIR = os.environ.get('DATA_DIR')
    # evaluation code does not work on MPS
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rnn', help='rnn')
    parser.add_argument('--dataset', type=str, default='chinese_baxter',
                        help='chinese_wikihan2022/chinese_baxter/romance_ipa/romance_orto')
    args = parser.parse_args()

    # evaluate 10 runs
    runs_dir = os.path.join(WORK_DIR, 'checkpoints', f'{args.model}_{args.dataset}')
    runs = Path(runs_dir).rglob(f'{args.model}_{args.dataset}_gpu*_best_ed.pt')
    runs = list(runs)
    assert len(runs) == 10
    predictions_dir = os.path.join(WORK_DIR, 'predictions', f'{args.model}_{args.dataset}', 'phylogeny')
    Path(predictions_dir).mkdir(parents=True, exist_ok=True)

    for model_path in runs:
        run_name = str(Path(model_path).parent).split('/')[-1]
        distance_matrix, dialects = generate_distance_matrix(args.model, model_path)
        Z = linkage(distance_matrix, 'ward')

        # plot
        if 'chinese' in args.dataset:
            fig = plt.figure(figsize=(25, 10))
        else:
            fig = plt.figure(figsize=(8, 6))
        new_name = {
            'romance_orto': 'Romance orthographic',
            'romance_ipa': 'Romance phonetic',
            'chinese_baxter': 'Chinese'
        }
        plt.title("RNN model on " + new_name[args.dataset])
        matplotlib.rcParams['lines.linewidth'] = 5
        matplotlib.rcParams['axes.titlesize'] = 30 if 'chinese' in args.dataset else 18
        # fig.axes.spines[['left', 'top', 'right', 'bottom']].set_visible(False)
        # fig.axes.axis('off')
        # plt.margins(x=5.)
        dn = dendrogram(Z, labels=dialects, orientation="left" if 'romance' in args.dataset else "top", leaf_font_size=18)
        # plt.show()
        fig.tight_layout()
        preds_path = os.path.join(predictions_dir, run_name)
        plt.savefig(preds_path + '.png')

        newick = dendrogram_to_newick(Z, dialects)
        with open(preds_path + '.newick', "w") as f:
            f.write(newick)
            f.write("\n")
        print('saved to ', preds_path + '.newick')
