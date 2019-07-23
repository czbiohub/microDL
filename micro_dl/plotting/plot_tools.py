import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def singe_target_violins(dir_dict,
                         save_path,
                         orientations=['xy', 'xz'],
                         min_corr=0.5,
                         plot_color=['#008000', '#3CB371'],
                         use_labels=True):
    """
    Takes one dict containing channels as keys and paths to where metrics
    are stored as values. E.g.
    actin_dict = {
    'ret': '/data/models/registered_kidney_2019_z5_l1_rosin/predictions',
    'ret_bf': '/data/models/registered_kidney_2019_z5_l1_rosin_2channels/predictions',
    'ret_xy': '/data/models/registered_kidney_2019_z5_l1_rosin_channels/predictions',
    'ret_xy_bf': '/data/models/registered_kidney_2019_z5_l1_rosin_4channels/predictions',
    }
    And plots violins with Pearson correlation of one orientation to the left
    and the other to the right. Saves plot in save_path.
    Colors used for green and magenta are:
    green_color = ['#3CB371', '#008000']
    magenta_color = ['#EE82EE', '#FF00FF']

    :param dict dir_dict: See above
    :param str save_path: Full path including extension to where plot is saved
    :param list orientations: Two orientations, subset of xy, yz, xz, xyz
    :param float min_corr: Bottom limit in y axis for plot
    :param list plot_color: Specify two colors for left/right side of violins
    :param bool use_labels: Set labels or not
    """

    assert len(orientations) == 2, "Must use one orientation for left/right each"
    available_orientations = {'xy', 'xz', 'yz', 'xyz'}
    assert set(orientations).issubset(available_orientations), \
        "Orientations must be subset of {}".format(available_orientations)

    all_metrics = [pd.DataFrame(), pd.DataFrame()]
    for key in dir_dict:
        metrics_dir = dir_dict[key]
        for i, orientation in enumerate(orientations):
            df_name = 'metrics_{}.csv'.format(orientation)
            metrics_name = os.path.join(metrics_dir, df_name)
            metrics_df = pd.read_csv(metrics_name)
            metrics_df['channels'] = key
            metrics_df['orientation'] = orientation
            all_metrics[i] = all_metrics[i].append(metrics_df)

    # Use same nbr of samples for both sides
    len_0 = all_metrics[0].shape[0]
    len_1 = all_metrics[1].shape[0]
    if len_1 > len_0:
        all_metrics[1] = all_metrics[1].sample(n=len_0)
    else:
        all_metrics[0] = all_metrics[0].sample(n=len_0)
    # Concatenate
    all_metrics = pd.concat([all_metrics[0], all_metrics[1]])
    # Plot violins
    fig = plt.figure()
    fig.set_size_inches((9.2, 3))
    ax = sns.violinplot(
        x='channels',
        y='corr',
        hue='orientation',
        data=all_metrics,
        scale='area',
        linewidth=1,
        palette=plot_color,
        bw=0.05,
        inner='quartile',
        split=True,
    )
    ax.set_ylim(bottom=min_corr, top=1)
    ax.legend(loc="lower right", borderaxespad=0.1)

    if use_labels:
        ax.set_xticklabels(labels=list(dir_dict))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(labels=[])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)


def double_target_violins(double_dict,
                          save_path,
                          orientation='xy',
                          plot_color=['g', 'm'],
                          use_labels=True):
    """
    takes a dict of two targets and plot the targets next to each other.
    E.g. using actin_dict from above and a similarly constructed nuclei_dict:
    double_dict = {'actin': actin_dict, 'nuclei': nuclei_dict}

    :param dict double_dict: Dict containing two dicts, see above.
    :param str save_path: Full path including extension to where plot is saved
    :param str orientation: One orientation, in xy, yz, xz, xyz
    :param list plot_color: Specify two colors for the two targets
    :param bool use_labels: Set labels or not
    """

    assert len(double_dict) == 2, 'Dict should contain two dicts'
    dict_names = list(double_dict)
    assert list(double_dict[dict_names[0]]) == list(double_dict[dict_names[1]]),\
        "Both dicts need to contain the same channels"

    all_metrics = pd.DataFrame()
    for dict_name in dict_names:
        temp_dict = double_dict[dict_name]
        for key in temp_dict:
            metrics_dir = temp_dict[key]
            df_name = 'metrics_{}.csv'.format(orientation)
            metrics_name = os.path.join(metrics_dir, df_name)
            metrics_df = pd.read_csv(metrics_name)
            metrics_df['channels'] = key
            metrics_df['orientation'] = orientation
            metrics_df['target'] = dict_name
            all_metrics = all_metrics.append(metrics_df)

    fig = plt.figure()
    fig.set_size_inches((9.2, 3))
    ax = sns.violinplot(
        x='channels',
        y='corr',
        hue='target',
        data=all_metrics,
        scale='area',
        linewidth=1,
        palette=plot_color,
        bw=0.05,
        inner='quartile',
        split=False,
    )
    ax.set_xticklabels(labels=list(temp_dict))
    ax.set_ylim(top=1)
    ax.legend(loc="lower right", borderaxespad=0.1)

    if use_labels:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(labels=[])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)
