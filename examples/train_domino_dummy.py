import json
import numpy as np

np.random.seed(0)
from domino import DominoSlicer
from collections import defaultdict

embeddings = np.load("coig_firefly_train_jiping_0720_outputs.jsonl_emb", allow_pickle=True)

data, rm_scores, entropy = [], [], []
with open("coig_firefly_train_jiping_0720_outputs.jsonl") as fin:
    for line in fin:
        data.append(line.strip())
        line = json.loads(line.strip())
        rm_scores.append(line["rm_score"])
        entropy.append(line["entropy"])
assert embeddings.shape[0] == len(rm_scores)
rm_scores = np.array(rm_scores)
entropy = np.array(entropy)

domino = DominoSlicer(init_params="kmeans",
                      n_slices=128,
                      n_mixture_components=128,
                      max_iter=200,
                      y_log_likelihood_weight=16.0,
                      y_hat_log_likelihood_weight=16.0)
indices = domino.fit(data=None, embeddings=embeddings, targets=rm_scores, pred_probs=entropy)

clusters = defaultdict(list)
for i, idx in enumerate(indices):
    clusters[int(idx)].append(data[i])

with open("domino_slice128_iter200_y16_yhat16.json", "w") as fout:
    json.dump(clusters, fout, ensure_ascii=False, indent=2)
