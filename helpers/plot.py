import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.subplots as sp
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from snorkel.labeling import LabelingFunction

# Constant representing abstaining from labeling
ABSTAIN = -1 

def plot_label_distribution(
    label_matrix: np.ndarray,
    title: str = 'Distribution of Number of Labels per Example',
    color: str = 'royalblue',
    show_stats: bool = True,
    normalize: bool = True,
    nbins: int = None
) -> go.Figure:
    """
    Plot the distribution of the number of labels per example in the dataset using Plotly.

    Args:
        label_matrix (np.ndarray): A 2D numpy array where each row represents an example and 
                                   each column represents a label.
        title (str): Title for the plot.
        color (str): Color of the histogram bars.
        show_stats (bool): Whether to show statistics (mean, median) on the plot.
        normalize (bool): Whether to normalize the histogram to show fractions instead of counts.
        nbins (Optional[int]): Number of bins for histogram. If None, uses number of possible labels.

    Returns:
        go.Figure: Plotly figure object that can be further customized or displayed.
    """

    # Validate the input matrix is 2D
    if label_matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {label_matrix.shape}")

    # Count the number of non-abstaining labels per example (each row)
    num_labels_per_example = (label_matrix != ABSTAIN).sum(axis=1)
    
    # Compute basic statistics: mean and median number of labels per example
    mean_labels = np.mean(num_labels_per_example)
    median_labels = np.median(num_labels_per_example)
    
    # Determine number of bins for the histogram if not provided
    if nbins is None:
        nbins = min(label_matrix.shape[1] + 1, 20)  # Add one for 0 labels and cap at 20 bins
    
    # Create the histogram trace for the plot
    histogram = go.Histogram(
        x=num_labels_per_example,
        histnorm='probability' if normalize else None,  # Normalize when needed
        nbinsx=nbins,
        marker=dict(color=color),
        name='Label Distribution'
    )
    
    # Define the layout of the plot with axis titles and overall styling
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title='Number of Labels per Example',
            dtick=1  # Force x-axis ticks to be integers
        ),
        yaxis=dict(
            title='Fraction of Dataset' if normalize else 'Count'
        ),
        bargap=0.2,
        template='plotly_white'  # Use a clean Plotly template
    )
    
    # Create the figure by combining the histogram trace and layout
    fig = go.Figure(data=[histogram], layout=layout)
    
    # If requested, add vertical lines and annotations to show mean and median statistics
    if show_stats:
        fig.add_vline(x=mean_labels, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_labels:.2f}")
        fig.add_vline(x=median_labels, line_dash="dot", line_color="green",
                     annotation_text=f"Median: {median_labels:.2f}")
        
        # Annotation block: shows total examples, max labels, and number of unlabeled examples
        fig.add_annotation(
            x=0.98, y=0.98,
            xref="paper", yref="paper",
            text=f"Total examples: {len(num_labels_per_example)}<br>"
                 f"Max labels: {num_labels_per_example.max()}<br>"
                 f"Unlabeled: {np.sum(num_labels_per_example == 0)}",
            showarrow=False,
            align="right",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    
    return fig

def plot_conflict_matrix(conflict_matrix_df):
    """
    Create an interactive visualization of the pairwise conflict matrix using Plotly,
    showing a smaller top-left section (approximately "one quarter" of dimensions).

    Args:
        conflict_matrix_df (pd.DataFrame): DataFrame containing the conflict matrix
        
    Returns:
        tuple: A tuple containing (heatmap_fig, summary_fig), the two Plotly figures generated.
    """
    # Determine sizes to extract a representative subset of the entire conflict matrix
    size1 = math.ceil(conflict_matrix_df.shape[0] / 2)
    size2 = math.floor(conflict_matrix_df.shape[0] / 2)

    # Extract a subsection (quarter) of the conflict matrix for visualization
    conflict_matrix_df_quarter = conflict_matrix_df.iloc[:size2, size2 :]

    # Create a heatmap using the extracted portion of the matrix
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=conflict_matrix_df_quarter.values,
        x=conflict_matrix_df_quarter.columns,
        y=conflict_matrix_df_quarter.index,  
        colorscale='Reds',
        hovertemplate='LF1: %{y}<br>LF2: %{x}<br>Conflict Rate: %{z:.3f}<extra></extra>'
    ))
    # Update layout to include titles and adjust axis properties
    heatmap_fig.update_layout(
        title="Pairwise Conflicts Between Labeling Functions", 
        xaxis=dict(title="Labeling Functions", tickangle=45),
        yaxis=dict(title="Labeling Functions")
    )

    # Build a summary bar chart that aggregates total conflict scores per labeling function (LF)
    total_conflicts = []
    # Use the DataFrame index (LF names) to aggregate conflict counts per LF
    lf_names = conflict_matrix_df.index
    for lf in lf_names:
        # Sum conflicts across the row, subtracting the self-conflict entry
        conflict_sum = conflict_matrix_df.loc[lf].sum() - conflict_matrix_df.loc[lf, lf]
        total_conflicts.append(conflict_sum)

    # Create a summary DataFrame for easy plotting
    summary_df = pd.DataFrame({
        'Labeling Function': lf_names,
        'Total Conflicts': total_conflicts
    })

    # Sort the DataFrame in descending order of total conflicts
    summary_df = summary_df.sort_values('Total Conflicts', ascending=False)

    # Generate a bar chart from the summary DataFrame using Plotly Express
    summary_fig = px.bar(
        summary_df,
        x='Labeling Function',
        y='Total Conflicts',
        title='Total Conflicts by Labeling Function',
        color='Total Conflicts',
        color_continuous_scale='Reds'
    )
    # Update layout with x-axis rotation and y-axis title styling
    summary_fig.update_layout(
        xaxis=dict(title="Labeling Functions", tickangle=45),
        yaxis=dict(title="Total Conflict Score")
    )

    # Return both the heatmap and bar chart figures
    return heatmap_fig, summary_fig

def plot_coverage_overlap(labeling_functions: list[LabelingFunction], 
                          label_matrix: np.ndarray,
                          colorscale='Reds',
                          show_values=True,
                          sort_by_coverage=True) -> dict:
    """
    Visualize coverage and overlap between labeling functions.
    
    Args:
        labeling_functions (list[LabelingFunction]): A list of labeling functions.
        label_matrix (np.ndarray): A numpy array of shape (num_examples, num_lfs) with labels.
        colorscale (str or list): Colorscale for the heatmap (default: 'Reds').
        show_values (bool): Whether to show coverage values on the heatmap (default: True).
        sort_by_coverage (bool): Whether to sort LFs by their overall coverage (default: True).
        
    Returns:
        dict: A dictionary with overlap and coverage information, and the Plotly figure.
    """
    # Extract LF names from the list of labeling functions
    lf_names = [lf.name for lf in labeling_functions]
    n_lfs = len(labeling_functions)
    
    # Build a boolean mask to indicate coverage (non-ABSTAIN) per LF across all examples
    coverage_masks = [label_matrix[:, i] != ABSTAIN for i in range(n_lfs)]
    
    # Calculate the coverage rate (fraction of examples labeled) for each LF
    coverages = [mask.mean() for mask in coverage_masks]
    
    # Prepare indices for sorting, if needed
    indices = list(range(n_lfs))
    
    # If sorting by coverage is enabled, sort labeling functions in descending order based on their coverage
    if sort_by_coverage:
        sorted_indices = np.argsort(coverages)[::-1]  # Descending sort order
        indices = sorted_indices
    
    # Reorder LF names, masks, and coverage values using the sorted indices
    sorted_lf_names = [lf_names[i] for i in indices]
    sorted_masks = [coverage_masks[i] for i in indices]
    sorted_coverages = [coverages[i] for i in indices]
    
    # Initialize an overlap matrix to store pairwise overlap statistics between LFs
    overlap_matrix = np.zeros((n_lfs, n_lfs))
    
    # Calculate overlap rates: diagonal holds coverage rate, others hold pairwise overlap rates
    for i in range(n_lfs):
        for j in range(n_lfs):
            if i == j:
                # Diagonal: store individual LF coverage
                overlap_matrix[i, j] = sorted_coverages[i]
            else:
                # Compute overlap as the fraction of examples labeled by both LF i and LF j
                overlap_matrix[i, j] = (sorted_masks[i] & sorted_masks[j]).mean()
    
    # Convert the overlap matrix into a DataFrame for anyone needing tabular data
    overlap_df = pd.DataFrame(overlap_matrix, 
                              columns=sorted_lf_names, 
                              index=sorted_lf_names)
    
    # Create a DataFrame for LF-wise coverage for bar chart visualization later
    coverage_df = pd.DataFrame({
        'Labeling Function': sorted_lf_names,
        'Coverage': sorted_coverages
    })
    
    # Create a subplot layout with two columns: one for the heatmap, one for the coverage bar chart
    fig = sp.make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=["Labeling Function Overlap", "Coverage by LF"],
        specs=[[{"type": "heatmap"}, {"type": "bar"}]]
    )
    
    # Prepare custom hover text for each cell in the heatmap with percentage formatting
    hover_text = []
    for i in range(n_lfs):
        row_texts = []
        for j in range(n_lfs):
            if i == j:
                # Diagonal cell: show individual coverage information
                row_texts.append(f"{sorted_lf_names[i]}<br>Coverage: {sorted_coverages[i]:.2%}")
            else:
                # Off-diagonal cell: show pairwise overlap info
                row_texts.append(f"Overlap: {overlap_matrix[i, j]:.2%}<br>{sorted_lf_names[i]} ∩ {sorted_lf_names[j]}")
        hover_text.append(row_texts)
    
    # Optionally create a text annotation matrix to display values on the heatmap
    text = None
    if show_values:
        text = []
        for i in range(n_lfs):
            row_text = []
            for j in range(n_lfs):
                row_text.append(f"{overlap_matrix[i, j]:.2f}")
            text.append(row_text)
    
    # Create a version of the overlap matrix for visualization and set the diagonal to None
    # so that the heatmap background remains transparent on the diagonal
    viz_matrix = overlap_matrix.copy()
    for i in range(n_lfs):
        viz_matrix[i, i] = None
    
    # Add the heatmap trace to the figure using the prepared matrices and hover texts
    heatmap = go.Heatmap(
        z=viz_matrix,
        x=sorted_lf_names,
        y=sorted_lf_names,
        colorscale=colorscale,
        text=text,
        hoverinfo='text',
        hovertext=hover_text,
        showscale=True,
        colorbar=dict(
            title='Overlap',
            thickness=15,
            tickformat='.0%'
        )
    )
    fig.add_trace(heatmap, row=1, col=1)
    
    # Add a bar chart trace for the LF coverage statistics
    bar = go.Bar(
        x=coverage_df['Coverage'],
        y=coverage_df['Labeling Function'],
        orientation='h',
        marker=dict(color='rgba(58, 71, 80, 0.8)'),
        text=[f"{v:.1%}" for v in coverage_df['Coverage']],
        textposition='auto',
        name='Coverage'
    )
    fig.add_trace(bar, row=1, col=2)
    
    # Update the overall layout for the subplots: titles, axis formatting, and legend placement
    fig.update_layout(
        title='Labeling Function Coverage and Overlap Analysis',
        xaxis=dict(tickangle=-45),
        xaxis2=dict(
            title='Coverage',
            tickformat='.0%',
            range=[0, max(coverages) * 1.1]  # Create a margin of 10% above max coverage
        )
    )
    
    # Reverse the y-axis order on the bar chart to match the order in the heatmap
    fig.update_yaxes(
        autorange="reversed",
        row=1, col=2
    )
    
    # Return a dictionary containing the overlap DataFrame, coverage DataFrame, and the generated figure
    return {
        'overlap_matrix': overlap_df,
        'coverage': coverage_df,
        'figure': fig
    }

def plot_class_examples(dataset, num_classes=10, figsize=(15, 6), img_feature_name: str = 'img', label_feature_name: str = 'label', cmap=None):
    """
    Plot one example image for each class in the dataset.

    Args:
        dataset: The dataset containing images and labels.
        num_classes (int): Number of classes to display.
        figsize (tuple): Size of the figure as (width, height).
        img_feature_name (str): Name of the feature containing the images in the dataset.
        label_feature_name (str): Name of the feature containing the class labels in the dataset.
        cmap: Optional colormap for displaying images (e.g., 'gray' for grayscale images).

    Returns:
        tuple: Figure and axes objects for further customization if needed.
    """
    # Handle empty dataset with a simple message
    if len(dataset) == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "Empty dataset", ha='center', va='center')
        ax.axis("off")
        return fig, ax
    
    # Determine subplot layout: maximum number of columns is 5
    n_columns = min(num_classes, 5)
    n_rows = math.ceil(num_classes / n_columns)
    fig, axs = plt.subplots(n_rows, n_columns, figsize=figsize)
    fig.suptitle('Example Images by Class', fontsize=14)
    
    # Ensure axs is a 2D array even when there is only one row or one column
    if n_rows == 1:
        axs = np.array([axs])
    if n_columns == 1:
        axs = np.array([axs]).T
    
    try:
        # Retrieve class names from the dataset features
        class_names = dataset.features[label_feature_name].names
        
        # Loop over the specified number of classes and plot one example image per class
        for i in range(num_classes):
            row, col = i // n_columns, i % n_columns
            
            # Check if subplot space is available
            if row >= n_rows or col >= n_columns:
                continue
                
            # Use dataset filtering to select one sample matching the class label
            sample = dataset.filter(lambda x: x[label_feature_name] == i).select(range(1))
            
            # If an image is found for the current class, display it using imshow
            if len(sample[img_feature_name]) > 0:
                img = sample[img_feature_name][0]
                axs[row, col].imshow(img, cmap=cmap)
                axs[row, col].set_title(f"{class_names[i]} (Class {i})")
                axs[row, col].axis("off")
            else:
                # If no example exists, display a fallback text message
                axs[row, col].text(0.5, 0.5, f"No examples for class {class_names[i]}", 
                                  ha='center', va='center')
                axs[row, col].axis("off")
    
    except Exception as e:
        plt.suptitle(f"Error displaying class examples: {str(e)}")
        print(f"Error in plot_class_examples: {str(e)}")
    
    # Hide any unused subplots (in cases where num_classes < total subplots available)
    for i in range(num_classes, n_rows * n_columns):
        row, col = i // n_columns, i % n_columns
        if row < n_rows and col < n_columns:
            axs[row, col].axis("off")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust layout to reserve space for the title
    
    return fig, axs

def plot_confidence_distribution(
    probabilities: np.ndarray,
    positive_label: str = "POSITIVE",
    negative_label: str = "NEGATIVE",
    n_bins: int = 10,
    colors: tuple = ("royalblue", "crimson"),
    show_statistics: bool = True,
    height: int = 500,
    width: int = 800,
) -> go.Figure:
    """
    Plot histograms of both positive and negative class probability distributions.

    Args:
        probabilities: 2D numpy array with shape (n_samples, n_classes) containing probabilities
                       (can pass a 1D array if only plotting one class).
        positive_label: Name for the positive class (default: "POSITIVE").
        negative_label: Name for the negative class (default: "NEGATIVE").
        n_bins: Number of histogram bins (default: 10).
        colors: Tuple of (positive_color, negative_color) for histograms.
        show_statistics: Whether to show mean and median lines (default: True).
        height: Height of the plot in pixels (default: 500).
        width: Width of the plot in pixels (default: 800).

    Returns:
        fig: Plotly figure object that can be further customized or displayed.
    """
    # Ensure the probabilities are in numpy array format regardless of input type
    if isinstance(probabilities, list):
        probabilities = np.array(probabilities)

    # Process input probabilities based on its dimensionality.
    # For 1D arrays, infer positive probabilities and calculate negatives.
    if len(probabilities.shape) == 1:
        pos_probs = probabilities
        neg_probs = 1 - probabilities
    # For 2D arrays with at least two columns, assume first two columns are the classes
    elif len(probabilities.shape) == 2 and probabilities.shape[1] >= 2:
        pos_probs = probabilities[:, 1]  # Assume positive class probabilities are in column 1
        neg_probs = 1 - probabilities[:, 1] 
    else:
        raise ValueError(
            "Expected either 1D array of positive probs or 2D array with shape (n_samples, n_classes)"
        )

    pos_probs = pos_probs[pos_probs >= 0.5]  # Filter positive probabilities
    neg_probs = neg_probs[neg_probs < 0.5]  # Filter negative probabilities

    # Create histogram traces for both positive and negative class probabilities
    pos_histogram = go.Histogram(
        x=pos_probs,
        nbinsx=n_bins,
        marker=dict(color=colors[0], opacity=0.7),
        name=f"{positive_label} Class",
        histnorm="probability density",
    )
    neg_histogram = go.Histogram(
        x=neg_probs,
        nbinsx=n_bins,
        marker=dict(color=colors[1], opacity=0.7),
        name=f"{negative_label} Class",
        histnorm="probability density",
    )

    # Combine histogram traces into a single figure
    fig = go.Figure(data=[pos_histogram, neg_histogram])

    # If enabled, compute and add mean lines along with annotation texts for both classes
    if show_statistics:
        pos_mean = np.mean(pos_probs)
        neg_mean = np.mean(neg_probs)

        # Draw vertical mean line for positive class
        fig.add_shape(
            type="line",
            x0=pos_mean,
            y0=0,
            x1=pos_mean,
            y1=1,
            yref="paper",
            line=dict(color=colors[0], width=2, dash="dash"),
            label=dict(
                text=f"Positive Mean: {pos_mean:.2f}",
            ),
        )
        # Draw vertical mean line for negative class
        fig.add_shape(
            type="line",
            x0=neg_mean,
            y0=0,
            x1=neg_mean,
            y1=1,
            yref="paper",
            line=dict(color=colors[1], width=2, dash="dash"),
            label=dict(
                text=f"Negative Mean: {neg_mean:.2f}",
            ),
        )

    # Update figure layout with titles, axis ranges/formats, and styling details
    fig.update_layout(
        title=f"Confidence Distribution: {positive_label} vs. {negative_label}",
        xaxis=dict(
            title="Confidence", range=[0, 1], tickformat=".1f", gridcolor="lightgray"
        ),
        yaxis=dict(title="Density", gridcolor="lightgray"),
        bargap=0.1,
        barmode="overlay",  # Overlay histograms to facilitate comparison
        template="plotly_white",
        height=height,
        width=width,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )

    return fig