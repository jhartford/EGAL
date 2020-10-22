from matplotlib import pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os, tempfile, subprocess

TEXT_WIDTH = 13.96
COL_WIDTH = 8.255

def get_fig_size(fig_width_cm, fig_height_cm=None):
    """Convert dimensions in centimeters to inches.
    If no height is given, it is computed using the golden ratio.
    """
    if not fig_height_cm:
        golden_ratio = (1 + np.sqrt(5))/2
        fig_height_cm = fig_width_cm / golden_ratio

    size_cm = (fig_width_cm, fig_height_cm)
    return tuple([x/2.54 for x in size_cm])

def cleanup(ax, xlabel, ylabel, title=None):
    ax.legend(frameon=False, bbox_to_anchor=(0.5, -0.22), ncol=3, loc=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
font_size = 10

def save_fig(fig, file_name, fmt=None, dpi=300, tight=True):
    """Save a Matplotlib figure as EPS/PNG/PDF to the given path and trim it.
    """

    if not fmt:
        fmt = file_name.strip().split('.')[-1]

    if fmt not in ['eps', 'png', 'pdf']:
        raise ValueError('unsupported format: %s' % (fmt,))

    extension = '.%s' % (fmt,)
    if not file_name.endswith(extension):
        file_name += extension

    file_name = os.path.abspath(file_name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_name = tmp_file.name + extension

    # save figure
    if tight:
        fig.savefig(tmp_name, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(tmp_name, dpi=dpi)

    # trim it
    if fmt == 'eps':
        subprocess.call('epstool --bbox --copy %s %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'png':
        subprocess.call('convert %s -trim %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'pdf':
        subprocess.call('pdfcrop %s %s' % (tmp_name, file_name), shell=True)

def get_fig_size(fig_width_cm, fig_height_cm=None):
    """Convert dimensions in centimeters to inches.
    If no height is given, it is computed using the golden ratio.
    """
    if not fig_height_cm:
        golden_ratio = (1 + np.sqrt(5))/2
        fig_height_cm = fig_width_cm / golden_ratio

    size_cm = (fig_width_cm, fig_height_cm)
    return tuple([x/2.54 for x in size_cm])

def cleanup(ax, xlabel, ylabel, title=None):
    ax.legend(frameon=False, bbox_to_anchor=(0.5, -0.22), ncol=3, loc=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
font_size = 10

def figure_setup(usetex=False):
    """Set all the sizes to the correct values and use
    tex fonts for all texts.
    """
    params = {'text.usetex': usetex,
              'figure.dpi': 200,
              'font.size': font_size,
              'font.serif': [],
              'font.sans-serif': [],
              'font.monospace': [],
              'axes.labelsize': font_size-3,
              'axes.titlesize': font_size-3,
              'axes.linewidth': 0.6,
              'axes.spines.top': False,
              'axes.spines.bottom': False,
              'axes.spines.right': False,
              'axes.spines.left': False,
              'legend.frameon': False,
              #'legend.fontsize': 'smaller',
              #'text.fontsize': font_size(),
              'legend.fontsize': font_size-2,
              'xtick.labelsize': font_size-4,
              'ytick.labelsize':font_size-4,
              'font.family': 'serif'}
    plt.rcParams.update(params)

figure_setup()

def save_fig(fig, file_name, fmt=None, dpi=300, tight=True):
    """Save a Matplotlib figure as EPS/PNG/PDF to the given path and trim it.
    """

    if not fmt:
        fmt = file_name.strip().split('.')[-1]

    if fmt not in ['eps', 'png', 'pdf']:
        raise ValueError('unsupported format: %s' % (fmt,))

    extension = '.%s' % (fmt,)
    if not file_name.endswith(extension):
        file_name += extension

    file_name = os.path.abspath(file_name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_name = tmp_file.name + extension

    # save figure
    if tight:
        fig.savefig(tmp_name, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(tmp_name, dpi=dpi)

    # trim it
    if fmt == 'eps':
        subprocess.call('epstool --bbox --copy %s %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'png':
        subprocess.call('convert %s -trim %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'pdf':
        subprocess.call('pdfcrop %s %s' % (tmp_name, file_name), shell=True)

def lerp(x, a):
    return a / 2 + x * (1 - a /2 )

cmap = plt.get_cmap('magma')

get_ls = { 'EGAL hybrid':'solid', 
            'EGAL importance':'solid', 
            'EGAL eps-greedy':'solid',
            "Entropy Search":'dotted',
            "Random Search":'dashdot',
             "Least Confidence":'dotted',
             "Guided Learner":"dashed"}

def plotperf(experiments, models, names, exp_name="test", pdf=False, use_cmap=True):
    '''
    Plotting code to produce training figures
    '''
    if pdf:
        figure_setup(True)
    else:
        figure_setup(False)
    for j, metric in enumerate(["imbalance", "accuracy", "class_coverage"]):
        fig, ax = plt.subplots(figsize=get_fig_size(COL_WIDTH, COL_WIDTH / 2.))
        for i, (m, name) in enumerate(zip(models, names)):
            ls = get_ls[name]
            c = cmap(lerp(i / len(names), 0.2)) if use_cmap else f"C{i}"
            x = np.array([e[i]["queries"] for e in experiments]).mean(axis=0)
            mean = np.array([e[i][metric] for e in experiments]).mean(axis=0)
            if metric == "accuracy":
                print(name, mean[-1])
            sd = np.array([e[i][metric] for e in experiments]).std(axis=0) * 1.96 / np.sqrt(len(experiments))
            ax.plot(x, mean, linestyle=ls, color=c, label=name)
            ax.fill_between(x, mean + sd, mean - sd, color=c, alpha=0.1)
        ax.set_ylabel(metric.replace("_"," ").title())
        ax.set_xlabel("Num Samples")
        #ax.semilogx()
        ax.legend()

        ax.legend(frameon=False, bbox_to_anchor=(0.5, -0.22), ncol=2, loc=9, fontsize=font_size-4)


        for i in range(3):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
        fig.savefig(f"results/plots/{metric}-{exp_name}.png", bbox_inches="tight")
        if pdf:
            fig.savefig(f"results/plots/paper/{metric}-{exp_name}.pdf", bbox_inches="tight")
        plt.close(fig)
    
