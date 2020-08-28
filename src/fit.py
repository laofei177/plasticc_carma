import numpy as np
import pandas as pd
import os, sys
from agntk.carma.CARMATerm import *
from agntk.viz.mpl_viz import *
from scipy.optimize import differential_evolution, minimize
from celerite import GP
import celerite

bands = ["u", "g", "r", "i", "z", "y"]

# define log posterior function
def neg_ll(params, y, yerr, gp):
    """This method will catch 'overflow/underflow' runtimeWarning and 
    return -inf as probablility
    """
    # change few runtimewarning action setting
    notify_method = "raise"
    np.seterr(over=notify_method)
    np.seterr(under=notify_method)

    params = np.array(params)
    run = True
    lap = 0

    while run:
        if lap > 50:
            return -np.inf

        lap += 1
        try:
            gp.set_parameter_vector(params)
            neg_ll = -gp.log_likelihood(y)
            run = False
        except celerite.solver.LinAlgError:
            params += 1e-6 * np.random.randn(4)
            continue
        except np.linalg.LinAlgError:
            params += 1e-6 * np.random.randn(4)
            continue
        except FloatingPointError:
            return -np.inf

    return neg_ll


# vectorized neg_ll
vec_neg_ll = np.vectorize(neg_ll, excluded=[1, 2, 3], signature="(n)->()")


def lc_fit_data(lc_df, de=True, debug=False, plot=False, bounds=None):

    best_amp = np.zeros(6)
    best_tau = np.zeros(6)

    # fail_num = 0
    lc_df = lc_df.copy()
    std = np.std(lc_df.flux.values)

    if bounds is not None:
        bounds = bounds
    else:
        bounds = [(-4, np.log(4 * std)), (-4, 10)]

    # initialize parameter and kernel
    init_tau = np.exp(np.random.uniform(0, 6, 1)[0])
    init_amp = np.random.uniform(0, 4 * std, 1)[0]
    kernel = DRW_term(np.log(init_amp), np.log(init_tau))

    # loop through lc in each passband
    for band in range(6):

        try:
            lc_band = lc_df[lc_df.passband == band].copy()
            t = lc_band.mjd.values - lc_band.mjd.min()
            y = lc_band.flux.values
            yerr = lc_band.flux_err.values

            rerun = True  # dynamic control of bounds
            counter = -1
            gp = GP(kernel, mean=np.mean(y))
            gp.compute(t, yerr)

            if de:
                # set bound based on LC std for amp
                while rerun and (counter < 10):
                    counter += 1
                    r = differential_evolution(
                        neg_ll, bounds=bounds, args=(y, yerr, gp)
                    )
                    if 'jac' in r.keys():
                        jac_log = np.log10(np.dot(r.jac, r.jac)+1e-8)
                        
                        if jac_log > 0:
                            bounds = [(x[0] * 1.5, x[1] * 1.5) for x in bounds]
                        else:
                            rerun = False
                    else:
                        rerun = False
                        

            else:
                initial_params = gp.get_parameter_vector()

                while rerun and (counter < 10):
                    print(counter)
                    counter += 1
                    r = minimize(
                        neg_ll,
                        initial_params,
                        method="L-BFGS-B",
                        bounds=bounds,
                        args=(y, yerr, gp),
                    )
                    if 'jac' in r.keys():
                        jac_log = np.log10(np.dot(r.jac, r.jac)+1e-8)
                        if jac_log > 0:
                            bounds = [(x[0] * 1.5, x[1] * 1.5) for x in bounds]
                        else:
                            rerun = False
                    else:
                        rerun = False

            best_amp[band] = np.exp(r.x)[0]
            best_tau[band] = np.exp(r.x)[1]

        except Exception as e:
            print(r)
            print(e)
            print(
                f"Exception at object_id: {lc_df.object_id.values[0]}, passband: {band}"
            )
            best_amp[band] = np.nan
            best_tau[band] = np.nan
            # fail_num += 1

        # Below code is used to visualize if stuck in local minima
        if debug:
            print(r)

        if plot:
            plot_drw_ll(t, y, yerr, np.exp(r.x), gp, vec_neg_ll)

    return np.concatenate([[lc_df.object_id.values[0]], best_amp, best_tau])


def plot_lc(lc_df, log=False, meta=None, **kwargs):

    lc = lc_df.copy()
    lc["flux"] -= lc["flux"].min()

    fig = plt.figure(figsize=(10, 5))
    for band in range(6):
        lc_per_f = lc[lc.passband == band]

        if log and meta is not None:
            scale_y = np.log(lc_per_f.flux + meta[f"tflux_{bands[band]}"].values[0])
            plt.scatter(
                lc_per_f.mjd, scale_y - np.mean(scale_y), s=2, label=f"passband: {band}"
            )
        else:
            plt.errorbar(
                lc_per_f.mjd,
                lc_per_f.flux,
                lc_per_f.flux_err,
                fmt=".",
                label=f"passband: {band}",
            )

    plt.legend(fontsize=12)
    plt.xlabel("Time (day)")
    plt.ylabel(f"Flux (arb. unit)")
    plt.title(f"LC for object: {lc_df.object_id[0]}; Log: {log}", fontsize=18)

