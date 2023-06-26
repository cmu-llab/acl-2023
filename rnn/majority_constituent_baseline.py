from collections import defaultdict
import random
from panphon import FeatureTable
from panphon.distance import Distance
from utils import get_edit_distance
from lingrex.reconstruct import eval_by_bcubes
import argparse
import pickle


ft = FeatureTable()


def segment_daughter(daughter_str):
    onset, nucleus, coda = [], [], []
    currently_consonantal = 0
    #                           -1 => nucleus
    # in panphon, seg['cons'] =  0 => tone
    #                            1 => onset or coda

    for seg_str, seg in zip(ft.ipa_segs(daughter_str), ft.word_fts(daughter_str)):
        if seg['cons'] == 1:
            if len(nucleus) == 0:
                onset.append(seg_str)
            else:
                coda.append(seg_str)
        elif seg['cons'] == -1:
            nucleus.append(seg_str)

    return ''.join(onset), ''.join(nucleus), ''.join(coda)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='chinese_wikihan2022/chinese_baxter/romance')  # chinese_baxter = chinese_hou2004
args = parser.parse_args()

with open(f'data/{args.dataset}/test.pickle', 'rb') as fin:
    (langs, test_data) = pickle.load(fin)
dist = Distance()
random.seed(0)
protolang = 'Middle Chinese (Baxter and Sagart 2014)' if 'chinese' in args.dataset else 'Latin'

ED_sum = 0.
FED_sum = 0.
NED_sum = 0.
NPED_sum = 0.
PED_sum = 0.
correct = 0
count = 0
total_target_len, total_target_phoneme_len = 0, 0
predictions = []
phoneme_pairs = []
for character, entry in test_data.items():
    # include the tone
    mc = entry['protoform'][protolang]
    onsets, nuclei, codas = defaultdict(int), defaultdict(int), defaultdict(int)
    for daughter_form in entry['daughters'].values():
        ons, nuc, cod = segment_daughter(''.join(daughter_form))
        if ons: onsets[ons] += 1
        nuclei[nuc] += 1
        if cod: codas[cod] += 1
    chosen_ons = max(onsets, key=lambda x: onsets[x]) if len(onsets) > 0 else ''
    chosen_nuc = max(nuclei, key=lambda x: nuclei[x])
    chosen_cod = max(codas, key=lambda x: codas[x]) if len(codas) > 0 else ''
    my_recon = chosen_ons + chosen_nuc + chosen_cod

    mc_str = ''.join(mc)
    if "orto" not in args.dataset:
        # for IPA datasets, replace g with IPA ɡ b/c panphon ignores regular g
        mc_str, my_recon = mc_str.replace("g", "ɡ"), my_recon.replace("g", "ɡ")
        chosen_ons, chosen_nuc, chosen_cod = chosen_ons.replace("g", "ɡ"), chosen_nuc.replace("g", "ɡ"), chosen_cod.replace("g", "ɡ")
    CED = dist.fast_levenshtein_distance(mc_str, my_recon)
    ED_sum += CED
    NED_sum += CED / len(mc_str)
    FED_sum += dist.feature_edit_distance(mc_str, my_recon)
    PED = get_edit_distance(mc, [chosen_ons, chosen_nuc, chosen_cod])
    PED_sum += PED
    NPED_sum += PED / len(mc)
    correct += int(mc_str == my_recon)
    count += 1
    total_target_phoneme_len += len(mc)
    total_target_len += len(''.join(mc))
    predictions.append((mc_str, my_recon))
    phoneme_pairs.append([my_recon, mc])

bcubed_f_score = eval_by_bcubes(phoneme_pairs)

if "orto" in args.dataset:
    print(f'ED {ED_sum / count:.4f}  NED {NED_sum / count:.4f}  '
          f'BCFS {bcubed_f_score}  '
          f'Accuracy {correct / count:.4f}')
else:
    print(f'PED {PED_sum / count:.4f}  NPED {NPED_sum / count}  PER {PED_sum / total_target_phoneme_len:.4f}  '
          f'FED {FED_sum / count:.4f}  FER {dist.feature_error_rate(*zip(*predictions)):.4f}  '
          f'BCFS {bcubed_f_score}  '
          f'Accuracy {correct / count:.4f}')
