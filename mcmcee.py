import numpy as np
import time
from multiprocess import Pool
import emcee

def run_mcmc(ndim, nwalkers, nburn, nmcmc, dres, llprob, p0):
    with Pool() as pool:
        start = time.time()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, llprob, pool=pool)

        pos0 = [p0 + dres * np.random.randn(ndim) for i in range(nwalkers)]

        print("Running MCMC...")
        pos1 = sampler.run_mcmc(pos0, nburn, rstate0=np.random.get_state())
        sampler.reset()
        pos2 = pos1
        sampler.run_mcmc(pos2, nmcmc, rstate0=np.random.get_state(), progress=True)
        print("Done.")

        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
        print("Total time:", time.time() - start)

        samps = sampler.chain[:, nburn:, :].reshape((-1, ndim))

        # Compute percentiles for each parameter
        param_percentiles = np.percentile(samps, [16, 50, 84], axis=0)

        # Print MCMC results
        print("MCMC result:")
        for i, (lower, median, upper) in enumerate(param_percentiles.T):
            lower_err = median - lower
            upper_err = upper - median
            print(f"    x[{i}] = {median} + {upper_err} - {lower_err}")

        return samps
