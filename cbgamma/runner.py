import cbgamma.visualize.dynplot as dplt
import numpy as np


###
### Main entry point for the application.
###


def main():
    """The main function is called automatically when this file is executed."""

    plot = dplt.dyn_plot(nrows=1)
    
    for x in np.arange(0,10,0.1):
        plot.append(x, np.exp(-x**2)+10*np.exp(-(x-7)**2), index=0)
        #plot.append(x, np.exp(-x**3)+4*np.exp(-(x-3)**1), index=1)
        plot.update()

if __name__ == '__main__':
    main()
