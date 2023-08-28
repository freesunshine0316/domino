import json
import numpy as np
from domino import DominoSlicer

embeddings = np.load("coig_firefly_train_jiping_0720_outputs.jsonl_emb", allow_pickle=True)

rm_scores, entropy = [], []
with open("coig_firefly_train_jiping_0720_outputs.jsonl") as fin:
    for line in fin:
        line = json.loads(line.strip())
        rm_scores.append(line["rm_score"])
        entropy.append(line["entropy"])
assert embeddings.shape[0] == len(rm_scores)
rm_scores = np.array(rm_scores)
entropy = np.array(entropy)
y = np.rint(entropy)

domino = DominoSlicer(init_params="kmeans", n_slices=100, n_mixture_components=100)
domino.fit(data=None, embeddings=embeddings, targets=rm_scores, pred_probs=entropy)
