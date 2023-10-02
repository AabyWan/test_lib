"""

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay


def hist_fig(data, le, transform, figsize=(5,5)):
    _m = le['m'].classes_
    _a = le['a'].classes_

    n_cols = len(_m)
    n_rows = len(_a)

    #                                                   (width, height)
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=figsize, 
                            sharex=True, sharey=True, constrained_layout=True)
    
    for col_i, metric in enumerate(_m):
        for row_i, algo in enumerate(_a):
            # Transform strings to labels
            a_label = le['a'].transform(np.array(algo).ravel())
            m_label = le['m'].transform(np.array(metric).ravel())

            # Subset data and get the distances for the chosen transformation
            _X = data.query(f"algo=={a_label} and metric == {m_label}")[transform].values

            sns.histplot(
                _X, 
                kde=True, 
                stat='proportion', 
                bins=25, #type:ignore 
                ax=axes[row_i,col_i])
            axes[row_i,col_i].set(title=f"{algo.capitalize()} - {metric.capitalize()}", xlim=(-0.01,1))
    
    # Close the figure for memory management and to avoid it showing on return
    _ = plt.suptitle(f"Transformation = {transform.capitalize()}")
    plt.close()
    
    return fig

def bit_weights_ax(bits, title="", ax=None):
    if ax == None: ax = plt.gca()
    _ = sns.heatmap(
        data=bits.T, 
        cmap='Greys', 
        vmin=0, 
        vmax=1,
        cbar_kws={'label': 'Freq.'}, 
        ax=ax)
    ax.set(title=title)

    return ax

def kde_ax(data, transform, le, annotate=True, fill=False, threshold=None, title='', ax=None):
    # Create an axis if none is provided
    if ax == None : ax = plt.gca()
    
    # Copy to avoid overwriting original numeric class in provided data
    _data = data.copy()

    # Convert numeric class label to strings to overcome numeric label bug in SNS
    _data['class'] = le['c'].inverse_transform(_data['class'])

    # Plot 2d-lines using normalised KDE density.
    _=sns.kdeplot(_data, x=transform, hue='class',ax=ax)

    # Add TP and TN labels.
    if annotate:
        # Set y-axis to zero at first
        ax_max_y = 0

        for line, class_label in zip(ax.get_lines(), le['c'].classes_):
            x = np.array(line.get_data()[0])
            y = np.array(line.get_data()[1])

            # Add space between 2d-line and text
            max_y = y.max()+0.02 # #type:ignore 
            max_x = x[y.argmax()]
            
            # Set label text
            label = 'TP' if class_label == 'Intra (1)' else 'TN' 
            _=ax.text(max_x, max_y+0.02, f'{label}', horizontalalignment='center') 
            
            # Increase max y-axis according to annotation    
            ax_max_y = max_y+0.25 if max_y > ax_max_y else ax_max_y
        _=ax.set(ylim=(0,ax_max_y))
        
        if fill:
            # Re-plot KDE with fill. Fill does not have 2d-lines.
            sns.kdeplot(_data, x=transform, fill=True, hue='class',ax=ax)

        if threshold:
            # Get custom SNS legend handles from KDE plot
            handles = ax.legend_.legend_handles #type:ignore
            
            for handle, txt in zip(handles, ax.legend_.texts): #type:ignore
                # assign the legend labels to the handles
                handle.set_label(txt.get_text()) #type:ignore
            
            # Draw the decision threshold and add a label with the value
            dt = plt.axvline(threshold, label=f'Threshold@{threshold:.2f}', linestyle='--')
            
            # Update custom SNS legend with the added line.
            ax.legend(handles=handles + [dt], loc="upper left", title='Class')
        
        else:
            ax.legend(labels=le['c'].classes_, loc="upper left", title='Class')

        ax.set(title=title, xticks=np.arange(0.0, 1.01, 0.1), xlim=(0,1), xlabel='Similarity')

    return ax

def cm_ax(cm, class_labels=None, values_format='.0f', ax=None):
    # Wrapper for plotting the Confusion Matrix
    #cm : confusion matrix from sklearn.metrics.confusion_matrix
    #class_labels : (default=None) list of class labels
    #values_format : (default='.0f) f-string format of values
    #ax : (default=None) plt.gca() axis to plot on, or return a new ax
    ax = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
        ).plot(
            values_format=values_format,
            colorbar=False, cmap='Blues',ax=ax).ax_
    
    return ax

def eer_ax(fpr, tpr, thresholds, plot_circles=True, threshold=None, legend='', ax=None):
    # Create an axis if none is provided
    if ax == None : ax = plt.gca()

    # False Positive Rate (fpr) == False Accept Rate (FAR)
    # False Negative Rate (fnr) == False Rejection Rate (FRR)
    frr = 1-tpr 
    fpr_ax = ax.plot(thresholds, fpr, label=f'FPR {legend}')
    frr_ax = ax.plot(thresholds, frr, label=f'FRR {legend}')
    
    if plot_circles:
        ax.plot(thresholds, fpr, '.', mfc='none', color=fpr_ax[0].get_color())
        ax.plot(thresholds, frr, '.', mfc='none', color=frr_ax[0].get_color())

    if threshold:
        ax.axvline(threshold, label=f'Threshold {legend}', linestyle='--')
    
    ax.set(
        xlabel="Similarity",
        ylabel="Rate",
        xticks=(np.arange(0.0, 1.1, 0.1)),
        #xlim=(0,1.01),
        yticks=(np.arange(0.0, 1.1, 0.1)),
        #ylim=(0,1.01)
        )
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    
    # TODO add zoom insert of EER threshold using below
    # https://matplotlib.org/stable/gallery/axes_grid1/inset_locator_demo2.html
    ax.legend(loc="center left")

    return ax

def roc_ax(fpr, tpr, roc_auc, legend="", ax=None):
    # Create an axis if none is provided
    if ax == None : ax = plt.gca()
    
    if roc_auc == None: 
        label = f"{legend}"
    else:
        label = f"{legend} AUC={roc_auc:.4f}"

    ax.plot(fpr,tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", label="50/50 chance")    
    
    ax.set(
        xticks=(np.arange(0,1.1,0.1)),
        xlim=(0,1),
        yticks=(np.arange(0,1.1,0.1)),
        ylim=(0,1),
        xlabel="False Positive Rate", 
        ylabel="True Positive Rate")

    ax.legend(loc='lower right')
    return ax