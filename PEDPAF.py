"""
Author: Killian Lebreton, University of Orleans / LPC2E
Date: 2023-03 to 2023-08, Master 2 Internship
Description: PEDPAF plots and fits pulsar energy distributions
using energy files made with PSRSALSA
(PEDPAF: Pulsar Energy Distribution Plotting And Fitting)
"""

### Libraries ###
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sp
from scipy.stats import chi2, ks_2samp

plt.rcParams.update({"font.family": "Arial", "font.size": 20, "figure.figsize": (8, 6)})

proper_headers = [
    "POLN",
    "FREQ",
    "SUBINT",
    "ON Peak intensity",
    "OFF Peak intensity",
    "ON Integrated energy",
    "OFF Integrated energy",
    "ON RMS",
    "OFF RMS",
    "S/N",
]


### Functions ###
## Command line arguments ##
def parse_args():
    """
    Parse the commandline arguments.

    Returns
    -------
    args: populated namespace
        The commandline arguments.
    """

    parser = argparse.ArgumentParser(
        description="Plot and fit energy distributions from PSRSALSA energy files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("file", type=str, help="Path of the file")

    # options that affect the data prepping
    prep = parser.add_argument_group(title="Data prepping")

    prep.add_argument(
        "--pos_snr",
        dest="pos_snr",
        action="store_false",
        default=True,
        help="Disregard rotations for which the S/N is lower than -0.5",
    )

    prep.add_argument(
        "--mean_norm",
        dest="mean_norm",
        action="store_false",
        default=True,
        help="Normalize the on- and off-pulse energies using the mean of the on-pulse energies",
    )

    # options that affect the type of plot
    plot = parser.add_argument_group(title="Plot parameters")

    plot.add_argument(
        "-p",
        "--plot",
        dest="plot",
        type=str,
        choices=["on", "off", "both"],
        default="on",
        help="Define what distribution(s) is/are plotted",
    )

    plot.add_argument(
        "-f",
        "--fit",
        dest="fit",
        type=str,
        choices=["normal", "lognormal", "powerlaw", "chi_squared"],
        default=None,
        help="Define what fit is applied",
    )

    plot.add_argument(
        "-g",
        "--guesses",
        dest="guesses",
        type=float,
        nargs="+",
        default=None,
        help="Initial guesses for the energy distribution fitting",
    )

    plot.add_argument(
        "-b",
        "--bins",
        dest="bins",
        type=int,
        default=30,
        help="Number of bins to use for the distributions",
    )

    plot.add_argument(
        "--log",
        dest="log",
        action="store_false",
        default=True,
        help="Change axes' scale to logarithmic",
    )

    plot.add_argument(
        "--custom_ID",
        dest="ID",
        type=str,
        default=None,
        help="Custom source name, otherwise it is retrieved from the file",
    )

    # options that affect the output formatting
    output = parser.add_argument_group(title="Output formatting")

    output.add_argument(
        "-d",
        "--disp",
        dest="disp",
        action="store_false",
        default=True,
        help="Show the matplotlib window that display plots",
    )

    output.add_argument(
        "-s",
        "--save",
        dest="save",
        action="store_true",
        default=False,
        help="Save plots in pdf format",
    )

    output.add_argument(
        "-r",
        "--results",
        dest="results",
        action="store_false",
        default=True,
        help="Print fitting parameters and their uncertainties inside the terminal",
    )

    args = parser.parse_args()

    return args


def check_args(args):
    if args.guesses is not None and args.fit is None:
        print(
            "Inconsistency: there is no need to input guesses if you do not want the distributions to be fitted"
        )
        sys.exit(1)

    if args.bins < 0:
        print("Error: you cannot use a negative number of bins")
        sys.exit(1)

    if args.guesses is not None and len(args.guesses) != len(fitting_guesses[args.fit]):
        print(
            "The number of guesses you typed is different than what is required, please use "
            + str(len(fitting_guesses[args.fit]))
            + " guesses"
        )
        sys.exit(1)
    return


## Data preparation ##
def data_prepper_penergy(filepath, pos_snr, on_mean_norm, energies_only=True):
    df = pd.read_csv(
        filepath, sep=" ", skiprows=1, comment="#", header=None, names=proper_headers
    )

    subint = df["SUBINT"].to_numpy(dtype=float)

    ener_on = df["ON Integrated energy"].to_numpy(dtype=float)
    ener_off = df["OFF Integrated energy"].to_numpy(dtype=float)

    ener_on_rms = df["ON RMS"].to_numpy(dtype=float)
    ener_off_rms = df["ON RMS"].to_numpy(dtype=float)

    snr = df["S/N"].to_numpy(dtype=float)

    if pos_snr == True:
        mask = snr > -0.5

        subint = subint[mask]
        snr = snr[mask]
        ener_off = ener_off[mask]
        ener_on = ener_on[mask]
        ener_off_rms = ener_off_rms[mask]
        ener_on_rms = ener_on_rms[mask]

    if on_mean_norm == True:
        mean = np.mean(ener_on)
        ener_on = ener_on / mean
        ener_off = ener_off / mean

    if energies_only == True:
        return (ener_on, ener_off)
    else:
        return (ener_on, ener_off, ener_on_rms, ener_off_rms, subint, snr)


def metadata_penergy(filepath):
    df = pd.read_csv(filepath, sep=" ", comment="#")
    psr_name = df.columns[3]
    return psr_name


## Plotting ##
def plot_distrib_on_and_off(
    data, ID, loglog_axes=False, disp=True, save=False, on_bin_nb=150, off_bin_nb=30
):
    on = data[0]
    off = data[1]

    h1 = plt.hist(on, on_bin_nb, histtype="step", linewidth=3, label="On")
    h2 = plt.hist(off, off_bin_nb, histtype="step", linewidth=3, label="Off")
    plt.title(ID + " on and off pulse energy distributions")
    plt.xlabel("E / <E>")
    plt.ylabel("Count")
    plt.legend(ncol=2)
    plt.grid()
    plt.tight_layout()

    if loglog_axes == True:
        plt.xscale("log")
        plt.yscale("log")

    if save == True:
        if loglog_axes == True:
            plt.savefig(ID + "_distrib_on_and_off_loglog.pdf")
        else:
            plt.savefig(ID + "_distrib_on_and_off.pdf")

    if disp == True:
        plt.show()

    plt.close()

    return (h1, h2)


def plot_distrib(
    data,
    data_type,
    ID,
    loglog_axes=False,
    disp=True,
    save=False,
    bin_nb=30,
    close_fig=False,
):
    h1 = plt.hist(data, bin_nb, histtype="step", linewidth=3, label=data_type)
    plt.title(ID + " " + data_type + " energy distribution")
    plt.xlabel("E / <E>")
    plt.ylabel("Count")
    plt.grid()
    plt.tight_layout()

    if loglog_axes == True:
        plt.xscale("log")
        plt.yscale("log")

    if save == True:
        if loglog_axes == True:
            plt.savefig(ID + "_distrib_" + data_type + "_loglog.pdf")
        else:
            plt.savefig(ID + "_distrib_" + data_type + ".pdf")

    if disp == True:
        plt.show()

    if close_fig == True:
        plt.close()

    return h1


def plot_on_off_pulse_regions(phase, intensity, on_pulse_window, x_lims):
    """plots a intensity vs. phase pulse with filled on and off pulse regions

    Args:
        phase (array like): phase values between 0 and 1 or in bins
        intensity (array like): signal intensity
        on_pulse_window (2 value array like): start and end of the on pulse window
        x_lims (2 value array like): x axis limits for the plot
    """
    on_start = on_pulse_window[0]
    on_end = on_pulse_window[1]

    off_pulse_mask0 = phase < on_start
    off_pulse_mask1 = phase > on_end
    on_pulse_mask = (phase > on_start) * (phase < on_end)

    fig, ax = plt.subplots()

    ax.fill_between(
        phase[on_pulse_mask],
        intensity[on_pulse_mask],
        color="C0",
        alpha=0.5,
        label="On-pulse fluence",
    )

    ax.fill_between(
        phase[off_pulse_mask0],
        intensity[off_pulse_mask0],
        color="C1",
        alpha=0.7,
        label="Off-pulse fluence",
    )

    ax.fill_between(
        phase[off_pulse_mask1], intensity[off_pulse_mask1], color="C1", alpha=0.7
    )

    ax.vlines(
        [on_start, on_end],
        np.min(np.hstack((intensity[off_pulse_mask0], (intensity[off_pulse_mask1])))),
        np.max(intensity[on_pulse_mask]),
        colors="red",
        linestyles="--",
        linewidth=3,
        label="On-pulse window",
        zorder=5,
    )

    plt.plot(phase, intensity, color="black")
    plt.grid()
    plt.xlabel("Phase")
    # plt.xlim(x_lims)
    plt.legend()
    plt.ylabel("Intensity [AU]")
    plt.tight_layout()
    plt.show()


def plot_distrib_with_fit(
    data,
    fitting_func,
    guesses,
    data_type,
    ID,
    bin_nb=30,
    loglog_axes=False,
    disp=True,
    save=False,
    close_fig=False,
    fit_params=True,
):
    h = plot_distrib(
        data, data_type=data_type, ID=ID, save=False, disp=False, bin_nb=bin_nb
    )

    print(
        "For "
        + ID
        + " with a "
        + fitting_func.__name__
        + " for the "
        + data_type
        + " pulse"
    )
    x_fit, y_fit = fitting_func(
        data, guesses, bin_nb=bin_nb, disp_fit=fit_params, disp_uncer=fit_params
    )

    f = plt.plot(x_fit, y_fit, label=data_type + " fit", linestyle="--", linewidth=3)
    plt.legend()

    if loglog_axes == True:
        plt.xscale("log")
        plt.yscale("log")

    if save == True:
        if loglog_axes == True:
            plt.savefig(
                ID
                + "_distrib_"
                + data_type
                + "_"
                + fitting_func.__name__
                + "_loglog.pdf"
            )
        else:
            plt.savefig(
                ID + "_distrib_" + data_type + "_" + fitting_func.__name__ + ".pdf"
            )

    if disp == True:
        plt.show()

    if close_fig == True:
        plt.close()

    return (h, f)


def plot_distrib_on_and_off_with_fits(
    data_on,
    data_off,
    fit_func_on,
    fit_func_off,
    guesses_on,
    guesses_off,
    ID,
    bin_nb,
    loglog_axes=False,
    disp=True,
    save=False,
    fit_params=True,
):
    h_on, f_on = plot_distrib_with_fit(
        data_on,
        fit_func_on,
        guesses_on,
        "on",
        ID,
        bin_nb,
        save=False,
        disp=False,
        fit_params=fit_params,
    )
    h_off, f_off = plot_distrib_with_fit(
        data_off,
        fit_func_off,
        guesses_off,
        "off",
        ID,
        bin_nb,
        save=False,
        disp=False,
        fit_params=fit_params,
    )
    plt.title(ID + " on and off energy distributions")
    plt.legend(ncol=2)

    if loglog_axes == True:
        plt.xscale("log")
        plt.yscale("log")

    if save == True:
        if loglog_axes == True:
            plt.savefig(
                ID
                + "_both_distrib_on_"
                + fit_func_on.__name__
                + "_and_off_"
                + fit_func_off.__name__
                + "_loglog.pdf"
            )
        else:
            plt.savefig(
                ID
                + "_both_distrib_on_"
                + fit_func_on.__name__
                + "_and_off_"
                + fit_func_off.__name__
                + ".pdf"
            )

    if disp == True:
        plt.show()

    plt.close()

    return (h_off, f_off, h_on, f_on)


def plot_ccdf(data, data_type, ID, loglog_axes=False, disp=True, save=False, bin_nb=30):
    h1, b1, patches = plt.hist(
        data,
        bin_nb,
        histtype="step",
        linewidth=3,
        label=data_type,
        cumulative=-1,
        density=True,
    )
    patches[0].set_xy(patches[0].get_xy()[1:])  # removes the first bar of the histogram

    plt.title(ID + " " + data_type + " energy ccdf")
    plt.xlabel("E / <E>")
    plt.ylabel("Normalized count")
    plt.grid()
    plt.tight_layout()

    if loglog_axes == True:
        plt.xscale("log")
        plt.yscale("log")

    if save == True:
        if loglog_axes == True:
            plt.savefig(ID + "_ccdf_" + data_type + "_loglog.pdf")
        else:
            plt.savefig(ID + "_ccdf_" + data_type + ".pdf")

    if disp == True:
        plt.show()

    plt.close()

    return h1


## Fitting ##
def normal_distrib(x, mu, sig, A):
    f = 1 / (sig * np.sqrt(2 * np.pi))
    e = np.exp(-((x - mu) ** 2 / (2 * sig**2)))
    return f * e * A


def lognormal_distrib(x, mu, sig, A):
    f = 1 / (x * sig * np.sqrt(2 * np.pi))
    e = np.exp(-((np.log(x) - mu) ** 2 / (2 * sig**2)))
    return f * e * A


def powerlaw_distrib(x, a, b):
    return a * np.power(x, b)


def chi_squared_distrib(x, k, A):
    return A * chi2.pdf(x, df=k)


def normal_fitting(
    data, guesses, bin_nb=30, disp_fit=True, disp_uncer=True, return_coeffs=False
):
    hist, bins = np.histogram(data, bin_nb)
    bins_center = 0.5 * (bins[:-1] + bins[1:])

    fit, cov = sp.curve_fit(normal_distrib, bins_center, hist, p0=guesses)

    if disp_fit == True:
        print("The fitting parameters are:", fit)
    if disp_uncer == True:
        print("The uncertainties on the fitting parameters are:", np.sqrt(np.diag(cov)))

    if return_coeffs == True:
        print("-*-*-*-*-*-*-*-")
        return (fit, cov)
    else:
        x_fit = np.linspace(bins_center[0], bins_center[-1], 50)
        y_fit = normal_distrib(x_fit, fit[0], fit[1], fit[2])
        print("KS test: ", ks_2samp(hist, y_fit))
        print("-*-*-*-*-*-*-*-")
        return (x_fit, y_fit)


def lognormal_fitting(
    data, guesses, bin_nb=30, disp_fit=True, disp_uncer=True, return_coeffs=False
):
    hist, bins = np.histogram(data, bin_nb)
    bins_center = 0.5 * (bins[:-1] + bins[1:])

    pos_mask = bins[:-1] > 0
    bins_center = bins_center[pos_mask]
    hist = hist[pos_mask]

    fit, cov = sp.curve_fit(lognormal_distrib, bins_center, hist, p0=guesses)

    if disp_fit == True:
        print("The fitting parameters are:", fit)
    if disp_uncer == True:
        print("The uncertainties on the fitting parameters are:", np.sqrt(np.diag(cov)))

    if return_coeffs == True:
        return (fit, cov)
    else:
        x_fit = np.linspace(bins_center[0], bins_center[-1], 50)
        y_fit = lognormal_distrib(x_fit, fit[0], fit[1], fit[2])
        print("KS test: ", ks_2samp(hist, y_fit))
        print("-*-*-*-*-*-*-*-")
        return (x_fit, y_fit)


def powerlaw_fitting(
    data,
    guesses,
    fit_start=1,
    bin_nb=30,
    disp_fit=True,
    disp_uncer=True,
    return_coeffs=False,
):
    hist, bins = np.histogram(data, bin_nb)
    bins_center = 0.5 * (bins[:-1] + bins[1:])

    fit_mask = bins_center > fit_start

    fit, cov = sp.curve_fit(
        powerlaw_distrib,
        bins_center[fit_mask],
        hist[fit_mask],
        p0=guesses,
    )

    if disp_fit == True:
        print("The fitting parameters are:", fit)
    if disp_uncer == True:
        print("The uncertainties on the fitting parameters are:", np.sqrt(np.diag(cov)))

    if return_coeffs == True:
        return (fit, cov)
    else:
        x_fit = np.linspace(bins_center[fit_mask][0], bins_center[fit_mask][-1], 50)
        y_fit = powerlaw_distrib(x_fit, fit[0], fit[1])
        print("KS test: ", ks_2samp(hist[fit_mask], y_fit))
        print("-*-*-*-*-*-*-*-")
        return (x_fit, y_fit)


def broken_powerlaw_fitting(
    data,
    guesses,
    separation=1,
    bin_nb=30,
    disp_fit=True,
    disp_uncer=True,
    return_coeffs=False,
):
    hist, bins = np.histogram(data, bin_nb)
    bins_center = 0.5 * (bins[:-1] + bins[1:])

    split_mask = bins_center < separation
    guesses0 = guesses[0:2]
    guesses1 = guesses[2:4]

    fit0, cov0 = sp.curve_fit(
        powerlaw_distrib,
        bins_center[split_mask],
        hist[split_mask],
        p0=guesses0,
    )
    fit1, cov1 = sp.curve_fit(
        powerlaw_distrib,
        bins_center[~split_mask],
        hist[~split_mask],
        p0=guesses1,
    )

    if disp_fit == True:
        print("Up until x=" + str(separation) + " , the fitting parameters are:", fit0)
        print("After x=" + str(separation) + " , the fitting parameters are:", fit1)
    if disp_uncer == True:
        print(
            "Up until x="
            + str(separation)
            + " , the uncertainties on the fitting parameters are:",
            np.sqrt(np.diag(cov0)),
        )
        print(
            "After x="
            + str(separation)
            + " , the uncertainties on the fitting parameters are:",
            np.sqrt(np.diag(cov1)),
        )

    if return_coeffs == True:
        return (fit0, cov0, fit1, cov1)
    else:
        x_fit0 = np.linspace(
            bins_center[split_mask][0], bins_center[split_mask][-1], 50
        )
        y_fit0 = powerlaw_distrib(x_fit0, fit0[0], fit0[1])

        x_fit1 = np.linspace(
            bins_center[~split_mask][0], bins_center[~split_mask][-1], 50
        )
        y_fit1 = powerlaw_distrib(x_fit1, fit1[0], fit1[1])

        print("KS test 1: ", ks_2samp(hist[split_mask], y_fit0))
        print("KS test 2: ", ks_2samp(hist[~split_mask], y_fit1))
        print("-*-*-*-*-*-*-*-")

        if any(np.isnan(np.sqrt(np.diag(cov0)))) == True:
            y_fit0 = y_fit0 * np.nan
            print("First powerlaw removed because of bad fitting")

        x_fit = np.hstack((x_fit0, x_fit1))
        y_fit = np.hstack((y_fit0, y_fit1))

        return (x_fit, y_fit)


def chi_squared_fitting(
    data, guesses, bin_nb=30, disp_fit=True, disp_uncer=True, return_coeffs=False
):
    hist, bins = np.histogram(data, bin_nb)
    bins_center = 0.5 * (bins[:-1] + bins[1:])

    fit, cov = sp.curve_fit(chi_squared_distrib, bins_center, hist, p0=guesses)

    if disp_fit == True:
        print("The fitting parameters are:", fit)
    if disp_uncer == True:
        print("The uncertainties on the fitting parameters are:", np.sqrt(np.diag(cov)))

    if return_coeffs == True:
        return (fit, cov)
    else:
        x_fit = np.linspace(bins_center[0], bins_center[-1], 50)
        y_fit = chi_squared_distrib(x_fit, fit[0], fit[1])
        print("KS test: ", ks_2samp(hist, y_fit))
        print("-*-*-*-*-*-*-*-")
        return (x_fit, y_fit)


fitting_funcs = {
    "normal": normal_fitting,
    "lognormal": lognormal_fitting,
    "chi_squared": chi_squared_fitting,
    "powerlaw": powerlaw_fitting,
}
fitting_guesses = {
    "normal": [0, 0.1, 1],
    "lognormal": [2.5, 0.5, 1],
    "chi_squared": [1, 10],
    "powerlaw": [1, -0.5],
}


### Main ###


def main():
    args = parse_args()
    print(args)

    check_args(args)

    data = data_prepper_penergy(args.file, args.pos_snr, args.mean_norm)

    if args.ID is None:
        ID = metadata_penergy(args.file)
    else:
        ID = args.ID

    if args.guesses is None and args.fit is not None:
        guesses = fitting_guesses[args.fit]
    else:
        guesses = args.guesses

    if args.fit is None:
        if args.plot == "on":
            plot_distrib(
                data[0],
                "on-pulse",
                ID,
                loglog_axes=args.log,
                disp=args.disp,
                save=args.save,
                bin_nb=args.bins,
            )

        elif args.plot == "off":
            plot_distrib(
                data[1],
                "off-pulse",
                ID,
                loglog_axes=args.log,
                disp=args.disp,
                save=args.save,
                bin_nb=args.bins,
            )

        else:
            plot_distrib_on_and_off(
                data,
                ID,
                loglog_axes=args.log,
                disp=args.disp,
                save=args.save,
                bin_nb=args.bins,
            )

    else:
        if args.plot == "on":
            plot_distrib_with_fit(
                data[0],
                fitting_funcs[args.fit],
                guesses,
                "on-pulse",
                ID,
                loglog_axes=args.log,
                disp=args.disp,
                save=args.save,
                bin_nb=args.bins,
                fit_params=args.results,
            )

        elif args.plot == "off":
            plot_distrib_with_fit(
                data[1],
                fitting_funcs[args.fit],
                guesses,
                "off-pulse",
                ID,
                loglog_axes=args.log,
                disp=args.disp,
                save=args.save,
                bin_nb=args.bins,
                fit_params=args.results,
            )

        else:
            plot_distrib_on_and_off_with_fits(
                data[0],
                data[1],
                fitting_funcs[args.fit],
                normal_fitting,
                guesses,
                [0, 0.1, 1],
                ID,
                loglog_axes=args.log,
                disp=args.disp,
                save=args.save,
                bin_nb=args.bins,
                fit_params=args.results,
            )

    return


if __name__ == "__main__":
    main()


### Legacy Code ###
# observed_psr = [
#     "0329+54",
#     "0809+74",
#     "0834+06",
#     "0919+06",
#     "0950+08",
#     "1112+50",
#     "1133+16",
#     "1237+25",
# ]  # "0823+26" not included bc of MP and IP

# for psr in observed_psr:

#     path = "energy-files\B"+psr+"total.debase.gg.on.en"
#     on,off = data_prepper_penergy(path)

#     #plot_distrib(on,'On-pulse','B'+psr,loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib(off,'Off-pulse','B'+psr,loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib_with_fit(off,normal_fitting,[1.,7.,1],'Off-pulse','B'+psr,loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib_with_fit(on,lognormal_fitting,[2.75,0.5,1],'On-pulse','B'+psr,loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib_with_fit(on,chi_squared_fitting,[1,10],'On-pulse','B'+psr,loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib_with_fit(on,normal_fitting,[1.,7.,1],'On-pulse','B'+psr,loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib_with_fit(on,powerlaw_fitting,[1,-0.5],'On-pulse','B'+psr,loglog_axes=True,disp=False,save=True,close_fig=True)


# special = ['-IP.en','-MP.en']
# for s in special:
#     path = 'energy-files\B0823+26total.debase.gg.on'+ s
#     on,off = data_prepper_penergy(path)

#     #plot_distrib(on,'On-pulse','B0823+26 '+s[1:3],loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib(off,'Off-pulse','B0823+26 '+s[1:3],loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib_with_fit(off,normal_fitting,[1.,7.,1],'Off','B0823+26 '+s[1:3],loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib_with_fit(on,lognormal_fitting,[2.75,0.5,1],'On','B0823+26 '+s[1:3],loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib_with_fit(on,normal_fitting,[1.,7.,1],'On','B0823+26 '+s[1:3],loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib_with_fit(on,powerlaw_fitting,[1,-0.5],'On','B0823+26 '+s[1:3],loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib_with_fit(on,chi_squared_fitting,[1,10],'On','B0823+26 '+s[1:3],loglog_axes=True,disp=False,save=True,close_fig=True)
#     #plot_distrib(off,'Off','B0823+26 '+s[1:3],loglog_axes=True,disp=False,save=True,close_fig=True)
