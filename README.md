# PEDPAF (Pulsar Energy Distribution Plotting And Fitting)  

This program plots and fits pulsar energy distributions using energy files from PSRSALSA.
## Requirements
This program runs on Python 3 and requires the following packages.
* argparse
* sys
* matplotlib
* numpy
* pandas
* scipy

## Command line options

* positional arguments:
  * file:                  Path of the file

* options:
  * -h, --help:           show this help message and exit

* Data prepping:
  * --pos_snr:             Disregard rotations for which the S/N is lower than -0.5 (default: True)
  * --mean_norm:          Normalize the on- and off-pulse energies using the mean of the on-pulse energies (default: True)

* Plot parameters:
  * -p, --plot {on,off,both}: Define what distribution(s) is/are plotted (default: on)
  * -f, --fit {normal,lognormal,powerlaw,chi_squared}: Define what fit is applied (default: None)
  * -g, --guesses GUESSES [GUESSES ...]: Initial guesses for the energy distribution fitting (default: None)
  * -b, --bins BINS:  Number of bins to use for the distributions (default: 30)
  * --log:                 Change axes' scale to logarithmic (default: True)
  * --custom_ID ID:       Custom source name, otherwise it is retrieved from the file (default: None)

* Output formatting:
  * -d, --disp:            Show the matplotlib window that display plots (default: True)
  * -s, --save:            Save plots in pdf format (default: False)
  * -r, --results:         Print fitting parameters and their uncertainties inside the terminal (default: True)
