from sklearn.mixture import GaussianMixture



def opt_GMM(X, n_sampling, n_components, means_init):

    for i in range(n_sampling):
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type='full',
                              means_init=means_init, # 2
        #weights_init = [0.1, 0.33, 0.26, 0.1]
        # init_params='random'
                              )
        # np.random.shuffle(X)
        gmm.fit(X=X)
        this_AIC = gmm.aic(X=X)
        this_BIC = gmm.bic(X=X)

        if i == 0:
            best_AIC = this_AIC
            best_gmm = gmm
        elif this_AIC < best_AIC:
                best_AIC = this_AIC
                best_gmm = gmm

    return best_gmm
