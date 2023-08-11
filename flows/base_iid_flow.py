import normflowpy as nfp


class IIDNormalizingFlowModel(nfp.NormalizingFlowModel):
    def nll(self, x, **kwargs):
        raise NotImplementedError
