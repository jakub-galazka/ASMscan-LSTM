from sklearn.manifold import TSNE

# T-distributed Stochastic Neighbor Embedding (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
# multi_dim_representation: ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
def tsne_2d(multi_dim_representation):
    tsne = TSNE(verbose=2).fit_transform(multi_dim_representation)
    return tsne[:, 0], tsne[:, 1] # x, y

'''
Combinations to do (*with types separation):
   PB40_1z20_clu50_sampled10000, NLReff, bass_ntm_domain, fass_ntm_domain, fass_ctm_domain
   PB40_1z20_clu50_sampled10000, NLReff, bass_ntm_motif, fass_ntm_motif, fass_ctm_motif
   PB40_1z20_clu50_sampled10000, NLReff, bass_ntm_motif, bass_ntm_motif_env5", bass_ntm_motif_env10
   PB40_1z20_clu50_sampled10000, NLReff, fass_ntm_motif, fass_ntm_motif_env5", fass_ntm_motif_env10
   PB40_1z20_clu50_sampled10000, NLReff, fass_ctm_motif, fass_ctm_motif_env5", fass_ctm_motif_env10
-----
   PB40_1z20_clu50_sampled10000, NLReff, bass_ntm_domain*
   PB40_1z20_clu50_sampled10000, NLReff, bass_ntm_motif*
   PB40_1z20_clu50_sampled10000, NLReff, bass_ntm_motif_env5*
   PB40_1z20_clu50_sampled10000, NLReff, bass_ntm_motif_env10*
   PB40_1z20_clu50_sampled10000, NLReff, fass_ntm_motif*
   PB40_1z20_clu50_sampled10000, NLReff, fass_ntm_motif_env5*
   PB40_1z20_clu50_sampled10000, NLReff, fass_ntm_motif_env10*

Matplotlib scatter plot for tsne_2d config:
    from cycler import cycler
    plt.rc("axes", prop_cycle=cycler(color=COLORS_CYCLE))
    plt.axis("off")
constans:
    COLORS_CYCLE = ["whitesmoke", "lightgray", "blue", "green", "red", "tan", "orange", "gold", "cyan", "navy", "purple", "magenta", "orchid"] # min 13 needed
    MARKER_SIZE = 5;
'''
