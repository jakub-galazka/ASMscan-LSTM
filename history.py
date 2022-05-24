import sys
import numpy as np
import matplotlib.pyplot as plt

from util.model_config import ModelConfig
from util.ploter import config_plot, save_plot
from config import DATA_HISTORY_PATH, PLOT_HISTORY_PATH


model_id = sys.argv[1]

# Loading model config
config = ModelConfig(model_id)
FOLDS_QUANTITY = config.getParam("folds_quantity")

# Initializing metrics holders
loss_trn = []
loss_val = []
pr_auc_trn = []
pr_auc_val = []
roc_auc_trn = []
roc_auc_val = []
sens_at_spec_99_trn = []
sens_at_spec_99_val = []
sens_at_spec_999_trn = []
sens_at_spec_999_val = []

# Loading data
for fold_no in range(1, FOLDS_QUANTITY + 1):
    history = np.load(DATA_HISTORY_PATH % (model_id, fold_no), allow_pickle="TRUE").item()

    loss_trn.append(history["loss"])
    loss_val.append(history["val_loss"])
    pr_auc_trn.append(history["pr_auc"])
    pr_auc_val.append(history["val_pr_auc"])
    roc_auc_trn.append(history["roc_auc"])
    roc_auc_val.append(history["val_roc_auc"])
    sens_at_spec_99_trn.append(history["sens_at_spec_99"])
    sens_at_spec_99_val.append(history["val_sens_at_spec_99"])
    sens_at_spec_999_trn.append(history["sens_at_spec_999"])
    sens_at_spec_999_val.append(history["val_sens_at_spec_999"])

# Averaging metric
def mean_with_min_max_label(holder, label):
    mean = np.mean(holder, axis=0)
    std = np.std(holder, axis=0)
    if label == "min":
        i = np.argmin(mean)
    elif label == "max":
        i = np.argmax(mean)
    return mean, (mean[i], std[i])

loss_trn, loss_trn_label = mean_with_min_max_label(loss_trn, "min")
loss_val, loss_val_label = mean_with_min_max_label(loss_val, "min")
pr_auc_trn, pr_auc_trn_label = mean_with_min_max_label(pr_auc_trn, "max")
pr_auc_val, pr_auc_val_label = mean_with_min_max_label(pr_auc_val, "max")
roc_auc_trn, roc_auc_trn_label = mean_with_min_max_label(roc_auc_trn, "max")
roc_auc_val, roc_auc_val_label = mean_with_min_max_label(roc_auc_val, "max")
sens_at_spec_99_trn, sens_at_spec_99_trn_label = mean_with_min_max_label(sens_at_spec_99_trn, "max")
sens_at_spec_99_val, sens_at_spec_99_val_label = mean_with_min_max_label(sens_at_spec_99_val, "max")
sens_at_spec_999_trn, sens_at_spec_999_trn_label = mean_with_min_max_label(sens_at_spec_999_trn, "max")
sens_at_spec_999_val, sens_at_spec_999_val_label = mean_with_min_max_label(sens_at_spec_999_val, "max")

# Plotting
# Loss
config_plot("epoch [-]", "loss [-]", False)
plt.plot(loss_trn, label="trn (%.2g $\pm$ %.2g)" % loss_trn_label)
plt.plot(loss_val, label="val (%.2g $\pm$ %.2g)" % loss_val_label)
save_plot(PLOT_HISTORY_PATH % (model_id, "loss"))

# PR AUC
config_plot("epoch [-]", "average precision [-]")
plt.plot(pr_auc_trn, label="trn (%.2g $\pm$ %.2g)" % pr_auc_trn_label)
plt.plot(pr_auc_val, label="val (%.2g $\pm$ %.2g)" % pr_auc_val_label)
save_plot(PLOT_HISTORY_PATH % (model_id, "average_precision"))

# ROC AUC
config_plot("epoch [-]", "roc auc [-]")
plt.plot(roc_auc_trn, label="trn (%.2g $\pm$ %.2g)" % roc_auc_trn_label)
plt.plot(roc_auc_val, label="val (%.2g $\pm$ %.2g)" % roc_auc_val_label)
save_plot(PLOT_HISTORY_PATH % (model_id, "roc_auc"))

# Sensitivity at Specificity 99
config_plot("epoch [-]", "sensitivity [-]")
plt.plot(sens_at_spec_99_trn, label="trn (%.2g $\pm$ %.2g)" % sens_at_spec_99_trn_label)
plt.plot(sens_at_spec_99_val, label="val (%.2g $\pm$ %.2g)" % sens_at_spec_99_val_label)
save_plot(PLOT_HISTORY_PATH % (model_id, "sens_at_spec_99"))

# Sensitivity at Specificity 999
config_plot("epoch [-]", "sensitivity [-]")
plt.plot(sens_at_spec_999_trn, label="trn (%.2g $\pm$ %.2g)" % sens_at_spec_999_trn_label)
plt.plot(sens_at_spec_999_val, label="val (%.2g $\pm$ %.2g)" % sens_at_spec_999_val_label)
save_plot(PLOT_HISTORY_PATH % (model_id, "sens_at_spec_999"))
