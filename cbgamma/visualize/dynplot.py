from collections.abc import Iterable
import matplotlib.pyplot as plt
import time

class DynamicPlot():
    """Wrapper for the figure object that can be dynamically changed in realtime."""

    def __init__(self, nrows, ncols, sharex, sharey, sleep):
        plt.ion()
        self.figure, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey)
        self.axes = []
        self.sleep = sleep

        if isinstance(axs, Iterable):
            for ax in axs:
                self.axes.append(DynamicAxis(ax))
                ax.grid()
        else:
            self.axes.append(DynamicAxis(axs))
            axs.grid()

    def append(self, x, y, index=0):
        self.axes[index].append(x, y)

    def ax(self, index=0):
        self.axes[index]

    def update(self):
        for ax in self.axes:
            ax.update_lines()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        time.sleep(self.sleep)

class DynamicAxis():
    """Wrapper for the axis class to provide data that can be easily changed."""
    xdata = []
    ydata = []

    min_x = 0
    min_y = 0
    
    max_x = 10
    max_y = 10

    def __init__(self, ax):
        self.ax = ax
        self.lines, = ax.plot([], []);
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)

    def set_autoscalex_on(enable):
        self.ax.set_autoscalex_on(enable)

    def set_autoscaley_on(enable):
        self.ax.set_autoscaley_on(enable)

    def update_lines(self):
        self.lines.set_xdata(self.xdata)
        self.lines.set_ydata(self.ydata)
        self.ax.relim()
        self.ax.autoscale_view()

    def append(self, x, y):
        self.xdata.append(x)
        self.ydata.append(y)
        

def dyn_plot(nrows=1, ncols=1, sharex=False, sharey=False, sleep=0.05):
    return DynamicPlot(nrows, ncols, sharex, sharey, sleep)
    

             
