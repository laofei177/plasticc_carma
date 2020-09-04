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
    dim = len(params)
    run = True
    lap = 0

    while run:
        if lap > 10:
            return -np.inf

        lap += 1
        try:
            gp.set_parameter_vector(params)
            neg_ll = -gp.log_likelihood(y)
            run = False
        except celerite.solver.LinAlgError:
            params += 1e-6 * np.random.randn(dim)
            continue
        except np.linalg.LinAlgError:
            params += 1e-6 * np.random.randn(dim)
            continue
        except FloatingPointError:
            return -np.inf

    return neg_ll


# vectorized neg_ll
vec_neg_ll = np.vectorize(neg_ll, excluded=[1, 2, 3], signature="(n)->()")


def drw_log_param_init(std):
    """
    Randomly generate DRW parameters

    Args:
        std (float): The std of the LC to fit.

    Returns:
        list: The generated DRW parameters in natural log.
    """

    init_tau = np.exp(np.random.uniform(0, 6, 1)[0])
    init_amp = np.random.uniform(0, 4 * std, 1)[0]

    return np.log([init_amp, init_tau])


def drw_fit(lc_df, de=True, debug=False, plot=False, bounds=None):

    best_fit = np.zeros((2, 6))

    # fail_num = 0
    lc_df = lc_df.copy()
    std = np.std(lc_df.flux.values)

    if bounds is not None:
        first_bounds = bounds
    else:
        first_bounds = [(-4, np.log(4 * std)), (-4, 10)]

    # loop through lc in each passband
    for band in range(6):

        try:
            lc_band = lc_df[lc_df.passband == band].copy()
            t = lc_band.mjd.values - lc_band.mjd.min()
            y = lc_band.flux.values
            yerr = lc_band.flux_err.values

            rerun = True  # dynamic control of bounds
            counter = 0
            bounds = first_bounds
            jac_log_rec = 10

            # initialize parameter and kernel
            kernel = DRW_term(*drw_log_param_init(std))
            gp = GP(kernel, mean=np.mean(y))
            gp.compute(t, yerr)

            if de:
                # set bound based on LC std for amp
                while rerun and (counter < 5):
                    counter += 1
                    r = differential_evolution(
                        neg_ll, bounds=bounds, args=(y, yerr, gp), maxiter=200
                    )

                    if r.success:
                        best_fit[:, band] = np.exp(r.x)

                        if "jac" not in r.keys():
                            rerun = False
                        else:
                            jac_log = np.log10(np.dot(r.jac, r.jac) + 1e-8)

                            # if positive jac, then increase bounds
                            if jac_log > 0:
                                bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                            else:
                                rerun = False

                            # update best-fit if smaller jac found
                            if jac_log < jac_log_rec:
                                jac_log_rec = jac_log
                                best_fit[:, band] = np.exp(r.x)
                    else:
                        bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                        gp.set_parameter_vector(drw_log_param_init(std))

            else:
                initial_params = gp.get_parameter_vector()

                while rerun and (counter < 5):
                    counter += 1
                    r = minimize(
                        neg_ll,
                        initial_params,
                        method="L-BFGS-B",
                        bounds=bounds,
                        args=(y, yerr, gp),
                    )
                    if r.success:
                        best_fit[:, band] = np.exp(r.x)

                        if "jac" not in r.keys():
                            rerun = False
                        else:
                            jac_log = np.log10(np.dot(r.jac, r.jac) + 1e-8)

                            # if positive jac, then increase bounds
                            if jac_log > 0:
                                bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                            else:
                                rerun = False

                            # update best-fit if smaller jac found
                            if jac_log < jac_log_rec:
                                jac_log_rec = jac_log
                                best_fit[:, band] = np.exp(r.x)
                    else:
                        bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                        gp.set_parameter_vector(drw_log_param_init(std))

            if not r.success:
                best_fit[:, band] = np.nan

        except Exception as e:
            print(r)
            print(e)
            print(
                f"Exception at object_id: {lc_df.object_id.values[0]}, passband: {band}"
            )
            best_fit[:, band] = np.nan
            # fail_num += 1

        # Below code is used to visualize if stuck in local minima
        if debug:
            print(r)

        if plot:
            plot_drw_ll(t, y, yerr, np.exp(r.x), gp, vec_neg_ll)

    return np.concatenate([[lc_df.object_id.values[0]], best_fit.flatten()])


def dho_log_param_init():
    """
    Randomly generate DHO parameters

    Returns:
        list: The generated DHO parameters in natural log.
    """

    log_a1 = np.random.uniform(-10, 1, 1)[0]
    log_a2 = np.random.uniform(-14, -3, 1)[0]
    log_b0 = np.random.uniform(-10, -5, 1)[0]
    log_b1 = np.random.uniform(-10, -5, 1)[0]

    return [log_a1, log_a2, log_b0, log_b1]


def dho_fit(lc_df, debug=False, plot=False, bounds=None):

    best_fit = np.zeros((4, 6))
    lc_df = lc_df.copy()

    if bounds is not None:
        first_bounds = bounds
    else:
        first_bounds = [(-10, 7), (-14, 7), (-12, -2), (-11, -2)]

    # loop through lc in each passband
    for band in range(6):

        try:
            lc_band = lc_df[lc_df.passband == band].copy()
            t = lc_band.mjd.values - lc_band.mjd.min()
            y = lc_band.flux.values
            yerr = lc_band.flux_err.values

            rerun = True  # dynamic control of bounds
            compute = True  # handle can't factorize in gp.compute()
            succeded = False  # ever succeded
            compute_ct = 0
            counter = 0
            bounds = first_bounds
            jac_log_rec = 10

            # initialize parameter, kernel and GP
            kernel = DHO_term(*dho_log_param_init())
            gp = GP(kernel, mean=np.mean(y))

            # compute can't factorize, try 4 more times
            while compute & (compute_ct < 5):
                compute_ct += 1
                try:
                    gp.compute(t, yerr)
                    compute = False
                except celerite.solver.LinAlgError:
                    gp.set_parameter_vector(dho_log_param_init())

            # set bound based on LC std for amp
            while rerun and (counter < 5):
                counter += 1
                r = differential_evolution(
                    neg_ll, bounds=bounds, args=(y, yerr, gp), maxiter=200
                )

                if r.success:
                    succeded = True
                    best_fit[:, band] = np.exp(r.x)

                    if "jac" not in r.keys():
                        rerun = False
                    else:
                        jac_log = np.log10(np.dot(r.jac, r.jac) + 1e-8)

                        # if positive jac, then increase bounds
                        if jac_log > 0:
                            bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                        else:
                            rerun = False

                        # update best-fit if smaller jac found
                        if jac_log < jac_log_rec:
                            jac_log_rec = jac_log
                            best_fit[:, band] = np.exp(r.x)
                else:
                    bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                    gp.set_parameter_vector(dho_log_param_init())

            # if no success found, set best to nan
            if not succeded:
                best_fit[:, band] = np.nan

        except Exception as e:
            print(r)
            print(e)
            print(
                f"Exception at object_id: {lc_df.object_id.values[0]}, passband: {band}"
            )
            best_fit[:, band] = np.nan

        # Below code is used to visualize if stuck in local minima
        if debug:
            print(r)

        if plot:
            plot_dho_ll(t, y, yerr, np.exp(r.x), gp, vec_neg_ll)

    return np.concatenate([[lc_df.object_id.values[0]], best_fit.flatten()])


def carma_param_init(dim):
    """
    Randomly generate DHO parameters from [-8, 1] in log.

    Args:
        dim (int): For a CARMA(p,q_ model, dim=p+q+1

    Returns:
        list: The generated CAMRA parameters in natural log.
    """

    log_param = np.random.uniform(-8, 1, dim)

    return log_param


def carma_fit(lc_df, p, q, de=True, debug=False, plot=False, bounds=None):

    best_fit = np.zeros((int(p + q + 1), 6))
    lc_df = lc_df.copy()

    if bounds is not None:
        first_bounds = bounds
    else:
        first_bounds = [(-10, 5)] * int(p + q + 1)

    # loop through lc in each passband
    for band in range(6):

        try:
            lc_band = lc_df[lc_df.passband == band].copy()
            t = lc_band.mjd.values - lc_band.mjd.min()
            y = lc_band.flux.values
            yerr = lc_band.flux_err.values

            rerun = True  # dynamic control of bounds
            compute = True  # handle can't factorize in gp.compute()
            succeded = False  # ever succeded
            compute_ct = 0
            counter = 0
            bounds = first_bounds
            jac_log_rec = 10

            # initialize parameter, kernel and GP
            log_params = carma_param_init(int(p + q + 1))
            kernel = CARMA_term(log_params[:p], log_params[p:])
            gp = GP(kernel, mean=np.mean(y))

            # compute can't factorize, try 4 more times
            while compute & (compute_ct < 5):
                compute_ct += 1
                try:
                    gp.compute(t, yerr)
                    compute = False
                except celerite.solver.LinAlgError:
                    gp.set_parameter_vector(carma_param_init(int(p + q + 1)))

            if de:
                # set bound based on LC std for amp
                while rerun and (counter < 5):
                    counter += 1
                    r = differential_evolution(
                        neg_ll, bounds=bounds, args=(y, yerr, gp), maxiter=200
                    )

                    if r.success:
                        succeded = True
                        best_fit[:, band] = np.exp(r.x)

                        if "jac" not in r.keys():
                            rerun = False
                        else:
                            jac_log = np.log10(np.dot(r.jac, r.jac) + 1e-8)

                            # if positive jac, then increase bounds
                            if jac_log > 0:
                                bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                            else:
                                rerun = False

                            # update best-fit if smaller jac found
                            if jac_log < jac_log_rec:
                                jac_log_rec = jac_log
                                best_fit[:, band] = np.exp(r.x)
                    else:
                        bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                        gp.set_parameter_vector(carma_param_init(int(p + q + 1)))

            else:
                initial_params = gp.get_parameter_vector()

                while rerun and (counter < 5):
                    counter += 1
                    r = minimize(
                        neg_ll,
                        initial_params,
                        method="L-BFGS-B",
                        bounds=bounds,
                        args=(y, yerr, gp),
                    )
                    if r.success:
                        succeded = True
                        best_fit[:, band] = np.exp(r.x)
                        jac_log = np.log10(np.dot(r.jac, r.jac) + 1e-8)

                        # if positive jac, then increase bounds
                        if jac_log > 0:
                            bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                        else:
                            rerun = False

                        # update best-fit if smaller jac found
                        if jac_log < jac_log_rec:
                            jac_log_rec = jac_log
                            best_fit[:, band] = np.exp(r.x)
                    else:
                        bounds = [(x[0] - 1, x[1] + 1) for x in bounds]
                        gp.set_parameter_vector(carma_param_init(int(p + q + 1)))

            if not succeded:
                best_fit[:, band] = np.nan

        except Exception as e:
            print(e)
            print(
                f"Exception at object_id: {lc_df.object_id.values[0]}, passband: {band}"
            )
            best_fit[:, band] = np.nan

        # Below code is used to visualize if stuck in local minima
        if debug:
            print(r)

    #         if plot:
    #             plot_dho_ll(t, y, yerr, np.exp(r.x), gp, vec_neg_ll)

    return np.concatenate([[lc_df.object_id.values[0]], best_fit.flatten()])


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
    plt.title(f"LC for object: {lc_df.object_id.values[0]}; Log: {log}", fontsize=18)

