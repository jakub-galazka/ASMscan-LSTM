from util.rep_fold import RepFold

#----------------------------------------------------------------------------------------------------

TEST_MODE_LABEL = "-test"

# model params
FOLDS_QUANTITY = 1
T = 40                  # Input sequence size / max sequence length
V = 26                  # Vocabulary size
EPOCHS = 30
D = 11                  # Embedding dimensionality
M = 14                  # Hidden / Cell state dimensionality

LAST_RETURN_SEQS_LAYER_NAME = "last-return-seqs"
LAYER_BEFORE_CLASSIF_NAME = "before-classif"

#----------------------------------------------------------------------------------------------------

OUT_FOLDER = "models"

# dirs tree
DIRS_TREE = [
    OUT_FOLDER + "/%s/data/history",
    OUT_FOLDER + "/%s/data/prediction",
    OUT_FOLDER + "/%s/data/summary",
    OUT_FOLDER + "/%s/plot/history",
    OUT_FOLDER + "/%s/plot/representation"
]

# data_loader paths
NEGATIVE_DIR = "data/negative/PB40"
POSITIVE_DIR = "data/positive/bass_motif"

NEGATIVE_FOLD_PATH = "data/negative/PB40/PB40_1z20_clu50_%s.fa"
POSITIVE_FOLD_PATH = "data/positive/bass_motif/bass_ctm_motif_%s.fa"

# save paths
MODEL_NAME = "bass-model"
def comb_model_name(model_name, folds_quantity):
    return model_name + "comb" + "".join(str(i) for i in range(1, folds_quantity + 1))

MODEL_ARCHITECTURE_PATH = OUT_FOLDER + "/%s/architecture.txt"
MODEL_CONFIG_PATH = OUT_FOLDER + "/%s/config.json"
MODEL_PATH = OUT_FOLDER + "/%s/model/" + MODEL_NAME + "%d-lstm"

DATA_HISTORY_PATH = DIRS_TREE[0] + "/" + MODEL_NAME + "%d.npy"
DATA_PREDICTION_PATH = DIRS_TREE[1] + "/%s.csv"

SUMMARY_PATH = DIRS_TREE[2] + "/%s.csv"

PLOT_HISTORY_PATH = DIRS_TREE[3] + "/%s.png"
REPRESENTATION_PATH = DIRS_TREE[4] + "/%s.png"

#----------------------------------------------------------------------------------------------------

# evaluate
EVALUATION_COMBS = [
    ["PB40_1z20_clu50", "bass_ntm_motif"],
    ["PB40_1z20_clu50", "bass_ntm_motif_env5"],
    ["PB40_1z20_clu50", "bass_ntm_motif_env10"],
    ["PB40_1z20_clu50", "fass_ctm_motif"],
    ["PB40_1z20_clu50", "fass_ctm_motif_env5"],
    ["PB40_1z20_clu50", "fass_ctm_motif_env10"],
    ["PB40_1z20_clu50", "fass_ntm_motif"],
    ["PB40_1z20_clu50", "fass_ntm_motif_env5"],
    ["PB40_1z20_clu50", "fass_ntm_motif_env10"],
    ["PB40_1z20_clu50", "bass_ntm_domain"],
    ["PB40_1z20_clu50", "fass_ctm_domain"],
    ["PB40_1z20_clu50", "fass_ntm_domain"],
    ["NLReff", "bass_ntm_domain"],
    ["NLReff", "bass_other_ctm_domain", "bass_other_ntm_domain"],
    ["NLReff", "fass_ntm_domain"],
    ["NLReff", "fass_ctm_domain"],
    ["NLReff", "fass_ntm_domain"],
    ["NLReff", "het-s_ntm_domain", "pp_ntm_domain", "sigma_ntm_domain"],
]

#----------------------------------------------------------------------------------------------------

# predict
TST_FOLDS = {
    "PB40_1z20_clu50": {
        "class": 0,
        "path": "data/negative/PB40/PB40_1z20_clu50_test.fa"
    },
    "PB40_1z20_clu50_sampled10000": {
        "class": 0,
        "path": "data/negative/PB40/PB40_1z20_clu50_test_sampled10000.fa"
    },
    "DisProt": {
        "class": 0,
        "path": "data/negative/DisProt/DisProt_test.fa"
    },
    "NLReff": {
        "class": 0,
        "path": "data/negative/NLReff/NLReff_test.fa"
    },
    "bass_ntm_motif": {
        "class": 1,
        "path": "data/positive/bass_motif/bass_ntm_motif_test.fa"
    },
    "bass_ntm_motif_env5": {
        "class": 1,
        "path": "data/positive/bass_motif/bass_ntm_motif_env5_test.fa"
    },
    "bass_ntm_motif_env10": {
        "class": 1,
        "path": "data/positive/bass_motif/bass_ntm_motif_env10_test.fa"
    },
    "bass_ntm_domain": {
        "class": 1,
        "path": "data/positive/bass_domain/bass_ntm_domain_test.fa"
    },
    "bass_other_ctm_domain": {
        "class": 1,
        "path": "data/positive/bass_domain/bass_other_ctm_domain_test.fa"
    },
    "bass_other_ntm_domain": {
        "class": 1,
        "path": "data/positive/bass_domain/bass_other_ntm_domain_test.fa"
    },
    "fass_ctm_domain": {
        "class": 1,
        "path": "data/positive/fass_domain/fass_ctm_domain_test.fa"
    },
    "fass_ntm_domain": {
        "class": 1,
        "path": "data/positive/fass_domain/fass_ntm_domain_test.fa"
    },
    "het-s_ntm_domain": {
        "class": 1,
        "path": "data/positive/fass_domain/het-s_ntm_domain_test.fa"
    },
    "pp_ntm_domain": {
        "class": 1,
        "path": "data/positive/fass_domain/pp_ntm_domain_test.fa"
    },
    "sigma_ntm_domain": {
        "class": 1,
        "path": "data/positive/fass_domain/sigma_ntm_domain_test.fa"
    },
    "fass_ctm_motif": {
        "class": 1,
        "path": "data/positive/fass_motif/fass_ctm_motif_test.fa"
    },
    "fass_ctm_motif_env5": {
        "class": 1,
        "path": "data/positive/fass_motif/fass_ctm_motif_env5_test.fa"
    },
    "fass_ctm_motif_env10": {
        "class": 1,
        "path": "data/positive/fass_motif/fass_ctm_motif_env10_test.fa"
    },
    "fass_ntm_motif": {
        "class": 1,
        "path": "data/positive/fass_motif/fass_ntm_motif_test.fa"
    },
    "fass_ntm_motif_env5": {
        "class": 1,
        "path": "data/positive/fass_motif/fass_ntm_motif_env5_test.fa"
    },
    "fass_ntm_motif_env10": {
        "class": 1,
        "path": "data/positive/fass_motif/fass_ntm_motif_env10_test.fa"
    }
}

#----------------------------------------------------------------------------------------------------

# represent
REP_COMBS = [
    [RepFold("PB40_1z20_clu50_sampled10000"), RepFold("NLReff"), RepFold("bass_ntm_domain"), RepFold("fass_ntm_domain"), RepFold("fass_ctm_domain")],
    [RepFold("PB40_1z20_clu50_sampled10000"), RepFold("NLReff"), RepFold("bass_ntm_motif"), RepFold("fass_ntm_motif"), RepFold("fass_ctm_motif")],
    [RepFold("PB40_1z20_clu50_sampled10000"), RepFold("NLReff"), RepFold("bass_ntm_motif"), RepFold("bass_ntm_motif_env5"), RepFold("bass_ntm_motif_env10")],
    [RepFold("PB40_1z20_clu50_sampled10000"), RepFold("NLReff"), RepFold("fass_ntm_motif"), RepFold("fass_ntm_motif_env5"), RepFold("fass_ntm_motif_env10")],
    [RepFold("PB40_1z20_clu50_sampled10000"), RepFold("NLReff"), RepFold("fass_ctm_motif"), RepFold("fass_ctm_motif_env5"), RepFold("fass_ctm_motif_env10")],
    [RepFold("PB40_1z20_clu50_sampled10000"), RepFold("NLReff"), RepFold("bass_ntm_domain", cut_type_rule="_")],
    [RepFold("PB40_1z20_clu50_sampled10000"), RepFold("NLReff"), RepFold("bass_ntm_motif", cut_type_rule=7)],
    [RepFold("PB40_1z20_clu50_sampled10000"), RepFold("NLReff"), RepFold("bass_ntm_motif_env5", cut_type_rule=7)],
    [RepFold("PB40_1z20_clu50_sampled10000"), RepFold("NLReff"), RepFold("bass_ntm_motif_env10", cut_type_rule=7)],
    [RepFold("PB40_1z20_clu50_sampled10000"), RepFold("NLReff"), RepFold("fass_ntm_motif", cut_type_rule="_")],
    [RepFold("PB40_1z20_clu50_sampled10000"), RepFold("NLReff"), RepFold("fass_ntm_motif_env5", cut_type_rule="_")],
    [RepFold("PB40_1z20_clu50_sampled10000"), RepFold("NLReff"), RepFold("fass_ntm_motif_env10", cut_type_rule="_")]
]

COLORS_CYCLE = ["whitesmoke", "lightgray", "blue", "green", "red", "tan", "orange", "gold", "cyan", "navy", "purple", "magenta", "orchid"] # min 13 needed
MARKER_SIZE = 5

TYPE_SEPARATION_LABEL = "type"

#----------------------------------------------------------------------------------------------------

# preprocesser
# aa composition: https://web.expasy.org/docs/relnotes/relstat.html (15.04.2022)
POPULATION = ["A", "Q", "L", "S", "R", "E", "K", "T", "N", "G", "M", "W", "D", "H", "F", "Y", "C", "I", "P", "V"]
WEIGHTS = [.0825, .0393, .0965, .0664, .0553, .0672, .0580, .0535, .0406, .0707, .0241, .0110, .0546, .0227, .0386, .0292, .0138, .0591, .0474, .0686]

# tokenizer
TOKENIZER_PATH = "tokenizer.pickle"

# ploter
PLOT_STYLE = "seaborn"

# csv
SEP = "\t"

PREDICTION_COLUMNS = ["id", "prob", "class", "frag"]
SUMMARY_COLUMNS = ["Model", "#pos", "#neg", "AUROC", "AP"]
RC_FPR_LABEL = "Rc|FPR1e-%d"
