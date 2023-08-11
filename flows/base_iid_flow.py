import normflowpy as nfp


class IIDNormalizingFlowModel(nfp.NormalizingFlowModel):
    def nll(self, x, **kwargs):
        zs, prior_logprob, log_det = self(x, **kwargs)
        logprob = prior_logprob + log_det  # Log-likelihood (LL)
        return -logprob  # Negative LL
