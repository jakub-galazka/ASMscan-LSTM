import os
import sys
import pandas as pd

from util.metricer import calculate_metrics
from config import DIRS_TREE, EVALUATION_COMBS, PREDICTION_COLUMNS, SUMMARY_COLUMNS, RC_FPR_LABEL, SEP, MODEL_NAME, SUMMARY_PATH


model_id = sys.argv[1]

# Evaluating for each test combination
for root, _, files in os.walk(DIRS_TREE[1] % model_id):
    for comb in EVALUATION_COMBS:
        comb_files = []
        summary = ""
        for fold in comb:
            comb_files.extend(filter(lambda f: fold == f.split(".")[0], files))
            summary += fold + "."
        summary = summary[:-1]

        # Collecting data from each cv model for every fold in combination
        preds = {}
        for file in comb_files:
            cv_model = file.split(sep=".")[1]

            if not cv_model in preds:
                preds[cv_model] = {}
                preds[cv_model]["y_true"] = []
                preds[cv_model]["y_pred"] = []
                preds[cv_model]["pos"] = 0
                preds[cv_model]["neg"] = 0

            df = pd.read_csv(os.path.join(root, file), sep=SEP)
            y_true = df[PREDICTION_COLUMNS[2]]
            preds[cv_model]["y_true"].extend(y_true)
            preds[cv_model]["y_pred"].extend(df[PREDICTION_COLUMNS[1]])

            y_true_len = len(y_true)
            if y_true[0]: preds[cv_model]["pos"] += y_true_len 
            else: preds[cv_model]["neg"] += y_true_len

        # Evaluating a combination for each cv model
        data = []
        for cv_model in preds:
            metrics = []
            y_true = preds[cv_model]["y_true"]
            y_pred = preds[cv_model]["y_pred"]

            # TODO: optimize
            roc_auc, pr_auc, tpr = calculate_metrics(y_true, y_pred)

            # Merging metrics
            metrics.extend([
                cv_model,
                preds[cv_model]["pos"],
                preds[cv_model]["neg"],
                roc_auc,
                pr_auc
            ])
            metrics.extend(tpr)

            data.append(metrics)

        # Generating labels for Rc|FPR metrics
        columns = SUMMARY_COLUMNS.copy()
        n = len(data[0]) - len(columns)
        for i in range(1, n + 1):
            columns.append(RC_FPR_LABEL % i)

        #----------------------------------------------------------------------------------------------------

        df = pd.DataFrame(data, columns=columns)

        # Averaging cv models (without combo)
        cvs = df.iloc[:-1]
        cvs_avg = cvs.mean(numeric_only=True)
        cvs_avg[columns[0]] = MODEL_NAME + "-avg"
        df.loc[len(df) - 1.5] = cvs_avg
        df = df.sort_index()

        # Saving
        df.to_csv(SUMMARY_PATH % (model_id, summary), sep=SEP, index=False)
