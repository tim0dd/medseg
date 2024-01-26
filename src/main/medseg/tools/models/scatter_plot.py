from matplotlib import pyplot as plt

data_1 = [
    {"Model": "U-Net", "GMACs": 270.78, "mIoU (CFU)": 75.77, "mIoU (UKW)": 69.56, "mIoU (MV)": 71.52},
    {"Model": "Segformer-B3", "GMACs": 77.45, "mIoU (CFU)": 78.85, "mIoU (UKW)": 76.52, "mIoU (MV)": 77.88},
    {"Model": "FCBFormer", "GMACs": 157.24, "mIoU (CFU)": 78.85, "mIoU (UKW)": 73.02, "mIoU (MV)": 75.13},
    {"Model": "HarDNet-DFUS", "GMACs": 139.37, "mIoU (CFU)": 77.51, "mIoU (UKW)": 73.11, "mIoU (MV)": 75.76},
    {"Model": "SegNeXt-B", "GMACs": 44.74, "mIoU (CFU)": 78.47, "mIoU (UKW)": 68.47, "mIoU (MV)": 70.76},
]

data_2 = {
    'Model': ['U-Net', 'Segformer-B3', 'FCBFormer', 'HarDNet-DFUS', 'SegNeXt-B']*3,
    'Size': [124.39, 47.22, 52.96, 51.07, 29.63]*3,
    'GMACs': [270.78, 77.45, 157.24, 139.37, 44.74]*3,
    'Type': ['Avg. CFU']*5 + ['Avg. UKW']*5 + ['Maj. vote UKW']*5,
    'mIoU': [75.77, 78.85, 78.85, 77.51, 78.47, 69.56, 76.52, 73.02, 73.11, 68.47, 71.52, 77.88, 75.13, 75.76, 70.76]
}

import pandas as pd
import seaborn as sns
from adjustText import adjust_text

def create_scatter_plot():
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data_1)

    # Reshape the DataFrame into a long format
    df = df.melt(id_vars=['Model', 'GMACs'], var_name='Type', value_name='mIoU')

    # Sort the DataFrame by GMACs for proper line plotting
    df = df.sort_values('GMACs')

    # Create a scatter plot with seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plot = sns.scatterplot(data=df, x='GMACs', y='mIoU', hue='Type')

    # Add lines connecting the dots for each type of result
    for type in df.Type.unique():
        df_type = df[df.Type == type]
        plt.plot(df_type.GMACs, df_type.mIoU, marker="None", linestyle='-',
                 color=sns.color_palette()[list(df.Type.unique()).index(type)])

    # Add labels for each point
    for line in range(0, df.shape[0]):
        plot.text(df.GMACs[line] + 0.2, df.mIoU[line], df.Model[line], horizontalalignment='left', size='small',
                  color='black')

    # Add a title
    # plt.title('Comparison of Neural Networks')

    # Show the plot
    plt.savefig('/home/timo/Code/Masterarbeit/out/plots/scatter_plot.png')


def create_bar_plot():
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data_1)

    # Reshape the DataFrame into a long format
    df = df.melt(id_vars=['Model', 'GMACs'], var_name='Type', value_name='mIoU')

    # Create a grouped bar plot
    plt.figure(figsize=(12, 8))
    plot = sns.barplot(data=df, x='Model', y='mIoU', hue='Type')

    # Add a title
    #plt.title('Comparison of Neural Networks')

    # Show the plot
    plt.savefig("/home/timo/Code/Masterarbeit/out/plots/bar_plot.png")


from pandas.plotting import parallel_coordinates


def parallel_coordinates_plot():
    # Convert the data to a pandas DataFrame
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data_1)

    # Normalize the GMACs values to a 0-100 scale to match the mIoU scale
    df['GMACs'] = (df['GMACs'] - df['GMACs'].min()) / (df['GMACs'].max() - df['GMACs'].min()) * 100

    # Create a parallel coordinates plot
    plt.figure(figsize=(10, 6))
    parallel_coordinates(df, 'Model', color=sns.color_palette())

    # Add a title
    plt.title('Comparison of Neural Networks')

    # Show the plot
    plt.savefig('/home/timo/Code/Masterarbeit/out/plots/parallel_coordinates_plot.png')

def create_scatter_plot_with_bubbles():
    # Convert the data to a pandas DataFrame
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data_2)
    # Normalize size for better visualization
    df['Size'] = (df['Size'] - df['Size'].min()) / (
                df['Size'].max() - df['Size'].min()) * 2000  # Increase multiplier for larger bubbles

    # Create a scatter plot (bubble chart)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create color palette
    palette = sns.husl_palette(len(df['Type'].unique()),h=0.7, l=.8)  # Adjust l (lightness) parameter as needed, .7 should be lighter
    scatter = sns.scatterplot(data=df, x='GMACs', y='mIoU', hue='Type', size='Size', sizes=(100, 2000), palette=palette,
                              alpha=0.5)

    # Add labels for each point
    texts = []
    for line in range(0, df.shape[0]):
        texts.append(
            plt.text(df.GMACs[line], df['mIoU'][line], df.Model[line], horizontalalignment='center', size='medium',
                     color='black'))



    # Connect the bubbles along each mIoU type, sorted by GMACs
    for mIoU_type, color in zip(df['Type'].unique(), palette):
        df_type = df[df['Type'] == mIoU_type]
        df_type = df_type.sort_values(by='GMACs')
        plt.plot(df_type['GMACs'], df_type['mIoU'], linestyle='--', color=color)

    # Flip GMACs axis
    plt.xlim(plt.xlim()[::-1])
    adjust_text(texts)
    # Add title and labels
    #plt.title('Comparison of Neural Networks')
    plt.xlabel('GMACs (inverted)')
    #ax.annotate('GMACs Inversed', xy=(0.5, -0.10), xycoords='axes fraction', fontsize=16,
    #            arrowprops=dict(arrowstyle='->', lw=1.5), xytext=(0.5, 0.10))
    plt.ylabel('mIoU')

    # Create a legend for the colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10) for i in
                       range(len(df['Type'].unique()))]
    # legend_elements += [
    #     patches.Patch(color='none', label="Small Model", marker="o", markersize=np.sqrt(100), markerfacecolor='black'),
    #     patches.Patch(color='none', label="Medium Model", marker="o", markersize=np.sqrt(1000),
    #                    markerfacecolor='black'),
    #     patches.Patch(color='none', label="Large Model", marker="o", markersize=np.sqrt(2000), markerfacecolor='black')]
    ax.legend(legend_elements, df['Type'].unique(), title="Evaluation modality", bbox_to_anchor=(1, 1), loc='upper left')
   # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Adjust the layout to fit everything

    plt.tight_layout()

    #plt.show()
    plt.savefig('/home/timo/Code/Masterarbeit/out/plots/bubbles.png')
def create_scatter_plot_rel_work():
    data = {
        'Model': ['Segformer-B3', 'Segformer-B4', 'SegNeXt-B', 'SegNeXt-L', 'Mask2Former Swin-T', 'Mask2Former R50'] * 2,
        'Size': [48, 64, 28, 49, 44, 44] * 2,
        'Dataset': ['ADE20K'] * 6 + ['Cityscapes'] * 6,
        'mIoU': [50.0, 51.8, 49.9, 52.1, 49.6, 49.2, 83.3, 83.8, 83.8, 83.9, None, 82.2]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.husl_palette(len(df['Dataset'].unique()), h=0.7, l=.8)  # Adjust l (lightness) parameter as needed
    scatter = sns.scatterplot(data=df, x='Size', y='mIoU', hue='Dataset', palette=palette)

    texts = []
    for line in range(0, df.shape[0]):
        if df.mIoU[line] is not None:  # Exclude missing data
            texts.append(
                plt.text(df.Size[line], df.mIoU[line], df.Model[line], horizontalalignment='center', size='medium',
                         color='black'))

    plt.xlabel('Size')
    plt.ylabel('mIoU')

    plt.tight_layout()
    plt.savefig('/home/timo/scatter_plot_rel.png')  # Change output path as needed

def create_separate_scatter_plots():
    data = {
        'Model': ['Segformer-B3', 'Segformer-B4', 'SegNeXt-B', 'SegNeXt-L', 'Mask2Former Swin-T', 'Mask2Former R50'] * 2,
        'Size': [48, 64, 28, 49, 44, 44] * 2,
        'Dataset': ['ADE20K'] * 6 + ['Cityscapes'] * 6,
        'mIoU': [50.0, 51.8, 49.9, 52.1, 49.6, 49.2, 83.3, 83.8, 83.8, 83.9, None, 82.2]
    }
    df = pd.DataFrame(data)

    # Create separate DataFrames for each dataset
    df_ADE20K = df[df['Dataset'] == 'ADE20K'].dropna(subset=['mIoU'])
    df_Cityscapes = df[df['Dataset'] == 'Cityscapes'].dropna(subset=['mIoU'])

    fig, axs = plt.subplots(2, figsize=(12, 12))

    for ax, df, dataset in zip(axs, [df_ADE20K, df_Cityscapes], ['ADE20K', 'Cityscapes']):
        scatter = sns.scatterplot(data=df, x='Size', y='mIoU', ax=ax)
        ax.set_title(dataset)

        texts = []
        for line in range(0, df.shape[0]):
            texts.append(
                ax.text(df.Size.iloc[line], df.mIoU.iloc[line], df.Model.iloc[line], horizontalalignment='center', size='medium',
                         color='black'))

        ax.set_xlabel('Size')
        ax.set_ylabel('mIoU')

    plt.tight_layout()
    plt.savefig('/home/timo/scatter_plot_rel.png')  # Change output path as needed

# Test the function


# def create_scatter_plot_with_bubbles():
#     data_2 = {
#         'Model': ['U-Net', 'Segformer-B3', 'FCBFormer', 'HarDNet-DFUS', 'SegNeXt-B'] * 3,
#         'Size': [124.39, 47.22, 52.96, 51.07, 29.63] * 3,
#         'GMACs': [270.78, 77.45, 157.24, 139.37, 44.74] * 3,
#         'Type': ['Avg. CFU'] * 5 + ['Avg. UKW'] * 5 + ['Maj. vote UKW'] * 5,
#         'mIoU': [75.77, 78.85, 78.85, 77.51, 78.47, 69.56, 76.52, 73.02, 73.11, 68.47, 71.52, 77.88, 75.13, 75.76,
#                  70.76]
#     }
#     df = pd.DataFrame(data_2)
#     df['Size'] = (df['Size'] - df['Size'].min()) / (
#                 df['Size'].max() - df['Size'].min()) * 2000  # Increase multiplier for larger bubbles
#
#     fig, ax = plt.subplots(figsize=(12, 6))
#     palette = sns.husl_palette(len(df['Type'].unique()),h=0.7, l=.8)  # Adjust l (lightness) parameter as needed, .7 should be lighter
#     scatter = sns.scatterplot(data=df, x='GMACs', y='mIoU', hue='Type', size='Size', sizes=(100, 2000), palette=palette,
#                               alpha=0.5)
#     texts = []
#     for line in range(0, df.shape[0]):
#         texts.append(
#             plt.text(df.GMACs[line], df['mIoU'][line], df.Model[line], horizontalalignment='center', size='medium',
#                      color='black'))
#     # Connect the bubbles along each mIoU type, sorted by GMACs
#     for mIoU_type, color in zip(df['Type'].unique(), palette):
#         df_type = df[df['Type'] == mIoU_type]
#         df_type = df_type.sort_values(by='GMACs')
#         plt.plot(df_type['GMACs'], df_type['mIoU'], linestyle='--', color=color)
#
#     plt.xlim(plt.xlim()[::-1])
#     adjust_text(texts)
#     plt.xlabel('GMACs (inverted)')
#     plt.ylabel('mIoU')
#     legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10) for i in
#                        range(len(df['Type'].unique()))]
#     ax.legend(legend_elements, df['Type'].unique(), title="Evaluation modality", bbox_to_anchor=(1, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig('/home/timo/Code/Masterarbeit/out/plots/bubbles.png')


if __name__ == '__main__':
    # create_scatter_plot()
    # create_bar_plot()
    # parallel_coordinates_plot()
    create_scatter_plot_with_bubbles()
    #create_separate_scatter_plots()
    #create_scatter_plot_rel_work()