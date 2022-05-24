from util.rep_fold import RepFold

#----------------------------------------------------------------------------------------------------

# model params
FOLDS_QUANTITY = 1
T = 40                      # Input sequence size / max sequence length
V = 26                      # Vocabulary size
EPOCHS = 25
D = 8                       # Embedding dimensionality
M = 6                       # Hidden state dimensionality

LAYER_BEFORE_CLASSIF_NAME = "before-classif"

#----------------------------------------------------------------------------------------------------

# dirs tree
DIRS_TREE = [
    "out/%s/data/history",
    "out/%s/data/prediction",
    "out/%s/data/summary",
    "out/%s/plot/history",
    "out/%s/plot/representation"
]

# data_loader paths
NEGATIVE_DIR = "data/negative/PB40"
POSITIVE_DIR = "data/positive/bass_motif"

NEGATIVE_FOLD_PATH = "data/negative/PB40/PB40_1z20_clu50_%s.fa"
POSITIVE_FOLD_PATH = "data/positive/bass_motif/bass_ctm_motif_%s.fa"

# save paths
MODEL_ARCHITECTURE_PATH = "out/%s/architecture.txt"
MODEL_CONFIG_PATH = "out/%s/config.json"

MODEL_NAME = "bass-model"
MODEL_PATH = "out/%s/model/" + MODEL_NAME + "%d-lstm"

DATA_HISTORY_PATH = "out/%s/data/history/" + MODEL_NAME + "%d.npy"
DATA_PREDICTION_PATH = "out/%s/data/prediction/%s.csv"

PLOT_HISTORY_PATH = "out/%s/plot/history/%s.png"

SUMMARY_PATH = "out/%s/data/summary/%s.csv"
REPRESENTATION_PATH = "out/%s/plot/representation/%s.png"

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
WHITE_SMOKE = "whitesmoke"
LIGHT_GRAY = "lightgray"
DARK_BLUE = "darkblue"
DARK_GREEN = "darkgreen"
DARK_GOLDEN_ROD = "darkgoldenrod"

REP_COMBS = [
    [RepFold("PB40_1z20_clu50_sampled10000", WHITE_SMOKE), RepFold("NLReff", LIGHT_GRAY), RepFold("bass_ntm_domain", DARK_BLUE), RepFold("fass_ntm_domain", DARK_GREEN), RepFold("fass_ctm_domain", DARK_GOLDEN_ROD)],
    [RepFold("PB40_1z20_clu50_sampled10000", WHITE_SMOKE), RepFold("NLReff", LIGHT_GRAY), RepFold("bass_ntm_motif", DARK_BLUE), RepFold("fass_ntm_motif", DARK_GREEN), RepFold("fass_ctm_motif", DARK_GOLDEN_ROD)],
    [RepFold("PB40_1z20_clu50_sampled10000", WHITE_SMOKE), RepFold("NLReff", LIGHT_GRAY), RepFold("bass_ntm_motif", DARK_BLUE), RepFold("bass_ntm_motif_env5", DARK_GREEN), RepFold("bass_ntm_motif_env10", DARK_GOLDEN_ROD)],
    [RepFold("PB40_1z20_clu50_sampled10000", WHITE_SMOKE), RepFold("NLReff", LIGHT_GRAY), RepFold("fass_ntm_motif", DARK_BLUE), RepFold("fass_ntm_motif_env5", DARK_GREEN), RepFold("fass_ntm_motif_env10", DARK_GOLDEN_ROD)],
    [RepFold("PB40_1z20_clu50_sampled10000", WHITE_SMOKE), RepFold("NLReff", LIGHT_GRAY), RepFold("fass_ctm_motif", DARK_BLUE), RepFold("fass_ctm_motif_env5", DARK_GREEN), RepFold("fass_ctm_motif_env10", DARK_GOLDEN_ROD)],
    # TODO: specific class representation
    # [RepFold("PB40_1z20_clu50_sampled10000", WHITE_SMOKE), RepFold("NLReff", LIGHT_GRAY), RepFold("bass_ntm_domain", sep="_")],
    # [RepFold("PB40_1z20_clu50_sampled10000", WHITE_SMOKE), RepFold("NLReff", LIGHT_GRAY), RepFold("bass_ntm_motif", class_id_len=7)],
    # [RepFold("PB40_1z20_clu50_sampled10000", WHITE_SMOKE), RepFold("NLReff", LIGHT_GRAY), RepFold("bass_ntm_motif_env5", class_id_len=7)],
    # [RepFold("PB40_1z20_clu50_sampled10000", WHITE_SMOKE), RepFold("NLReff", LIGHT_GRAY), RepFold("bass_ntm_motif_env10", class_id_len=7)],
    # [RepFold("PB40_1z20_clu50_sampled10000", WHITE_SMOKE), RepFold("NLReff", LIGHT_GRAY), RepFold("fass_ntm_motif", sep="_")],
    # [RepFold("PB40_1z20_clu50_sampled10000", WHITE_SMOKE), RepFold("NLReff", LIGHT_GRAY), RepFold("fass_ntm_motif_env5", sep="_")],
    # [RepFold("PB40_1z20_clu50_sampled10000", WHITE_SMOKE), RepFold("NLReff", LIGHT_GRAY), RepFold("fass_ntm_motif_env10", sep="_")]
]

REP_MODEL = MODEL_NAME + "comb%s"

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
