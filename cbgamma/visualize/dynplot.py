from matplotlib import figure
from matplotlib import _pylab_helpers
import matplotlib.pyplot as plt
import time


class DynamicFigure(Figure):
    """Wrapper for matplotlib.figure Figure class for creating
    dynamic plots that updates in realtime."""
    def __str__():
        return "Dynamic" + super().__str__()

        def __init__(self,
                     figsize=None,
                     dpi=None,
                     facecolor=None,
                     edgecolor=None,
                     linewidth=0.0,
                     frameon=None,
                     subplotpars=None,
                     tight_layout=None,
                     constrained_layout=None,
                     sleep=0.05):
            super().__init__(
                     figsize=figsize,
                     dpi=dpi,
                     facecolor=facecolor,
                     edgecolor=edgecolor,
                     linewidth=linewidth,
                     frameon=frameon,
                     subplotpars=subplotpars,
                     tight_layout=tight_layout,
                     constrained_layout=constrained_layout)

    def gca():
        
            
    
def gcdf():
    figManager = _pylab_helpers.Gcf.get_active()
    if figManager is not None:
        return figManager.canvas.figure
    else:
        return dyn_figure()

def dyn_figure(num=None,
               figsize=None,
               dpi=None,
               facecolor=None,
               edgecolor=None,
               frameon=True,
               FigureClass=Figure,
               clear=False,
               **kwargs
):
    
                     

def plot(*args, scalex=True, scaley=True, data=None, **kwargs):
    gca().plt.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs);

    
def update():
    
    
