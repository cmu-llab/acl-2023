"""
Random daughter baseline - pick a daughter reflex at random and predict this as the protoform
    which assumes that one daughter perfectly preserves the protoform
"""
import random
from panphon.distance import Distance
from utils import get_edit_distance
from lingrex.reconstruct import eval_by_bcubes
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='chinese_wikihan2022/chinese_baxter/romance')  # chinese_baxter = chinese_hou2004
args = parser.parse_args()

with open(f'data/{args.dataset}/test.pickle', 'rb') as fin:
    (langs, test_data) = pickle.load(fin)
dist = Distance()
random.seed(0)
protolang = 'Middle Chinese (Baxter and Sagart 2014)' if 'chinese' in args.dataset else 'Latin'

for daughter_baseline in langs + ['random']:
    if daughter_baseline == (protolang): continue
    ED_sum = 0.
    NED_sum = 0
    FED_sum = 0.  # feature edit distance
    PED_sum = 0.  # phoneme edit distance
    NPED_sum = 0
    correct = 0
    count = 0
    total_target_len, total_target_phoneme_len = 0, 0
    predictions = []
    phoneme_pairs = []
    for character, entry in test_data.items():
        # include the tone (index -1)
        mc = entry['protoform'][protolang]
        if daughter_baseline in entry['daughters']:
            daughter = entry['daughters'][daughter_baseline]
        else:
            daughter = random.choice(list(entry['daughters'].values()))

        mc_str, daughter_str = ''.join(mc), ''.join(daughter)
        if "orto" not in args.dataset:
            # for IPA datasets, replace g with IPA ɡ b/c panphon ignores regular g
            mc_str, daughter_str = mc_str.replace("g", "ɡ"), daughter_str.replace("g", "ɡ")
        #         edit_distance_sum += get_edit_distance(''.join(mc), ''.join(daughter))
        CED = dist.fast_levenshtein_distance(mc_str, daughter_str)
        ED_sum += CED
        NED_sum += CED / len(mc_str)
        FED_sum += dist.feature_edit_distance(mc_str, daughter_str)
        PED = get_edit_distance(mc, daughter)
        PED_sum += PED
        NPED_sum += PED / len(mc)
        correct += int(mc_str == daughter_str)
        count += 1
        total_target_phoneme_len += len(mc)
        total_target_len += len(''.join(mc))
        predictions.append((mc_str, daughter_str))
        phoneme_pairs.append([daughter, mc])

    bcubed_f_score = eval_by_bcubes(phoneme_pairs)

    if "orto" in args.dataset:
        print(f'{daughter_baseline:>10}   ED {ED_sum / count:.4f}  NED {NED_sum / count:.4f}  '
              f'BCFS {bcubed_f_score}  '
              f'Accuracy {correct / count:.4f}')
    else:
        print(f'{daughter_baseline:>10}   '
              f'PED {PED_sum / count:.4f}  NPED {NPED_sum / count}  PER {PED_sum / total_target_phoneme_len:.4f}  '
              f'FED {FED_sum / count:.4f}  FER {dist.feature_error_rate(*zip(*predictions)):.4f}  '
              f'BCFS {bcubed_f_score}  '
              f'Accuracy {correct / count:.4f}')
