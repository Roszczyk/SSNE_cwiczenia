import pandas as pd
import numpy as np

preds_csv = []

preds_csv.append(pd.read_csv("bert_classifier_preds.csv", names=["ratings"]))
preds_csv.append(pd.read_csv("absa_preds.csv", names=["ratings"]))
preds_csv.append(pd.read_csv("gpt_preds.csv", names=["ratings"]))
test_data = pd.read_csv("test_data.csv", names=["opinions"])

BEST = 2

preds_list = []
for pred in preds_csv:
    preds_list.append(list(pred["ratings"]))

print(len(preds_list[0]), len(preds_list[1]), len(preds_list[2]), "\ntest data: ", len(list(test_data["opinions"])))

final_preds = []

same = 0
best_decide = 0
voting = 0

for i in range(len(preds_list[1])):
    possible = []
    for j in range(len(preds_list)):
        possible.append(preds_list[j][i])
    if len(set(possible)) == 1:
        final_preds.append(possible[0])
        same = same + 1
    elif len(set(possible)) == len(preds_list):
        final_preds.append(possible[BEST])
        best_decide = best_decide + 1
    else:
        biggest = [-1,-1]
        for p in set(possible):
            current = possible.count(p)
            if current >= biggest[0]:
                biggest[0] = current
                biggest[1] = p
        final_preds.append(biggest[1])
        voting = voting + 1

final_preds = np.array(final_preds)

np.savetxt("summary_preds.csv", final_preds, delimiter=",")

print(f"same: {same}\nbest decide: {best_decide}\nvoting: {voting}")