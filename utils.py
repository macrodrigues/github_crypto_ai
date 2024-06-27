""" This function generates charts concerning user activity """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def process_commits_row(row):
    """ This function extracts meaningful information from each row in
    the dataset and return a dictionary. Note that the 'repo' variable is used
    to later on merge with the repositories dataset. """
    repo = \
        row['url'].split('repos')[1].split('commits')[0].split("/")[-2]

    return {
        'sha': row['sha'],
        'repo': repo,
        'node_id': row['node_id'],
        'author': row['commit']['author']['name'],
        'date': row['commit']['author']['date'],
        'message': row['commit']['message']}


def process_developers(df_filt):
    """ This function organizes de filtered dataframe by developer and year """
    most_active_dev = df_filt.groupby('author').agg({'sha': 'count'})\
        .reset_index().sort_values(
        by='sha', ascending=False).head(10)['author'].values
    df_developers = df_filt[
        df_filt.author.isin(list(most_active_dev))]
    df_developers['date'] = pd.to_datetime(df_developers['date'])
    df_developers['year'] = df_developers['date'].dt.year
    commits_per_author_per_year = df_developers.groupby([
        'author', 'year']).size().reset_index(name='commit_count')
    return commits_per_author_per_year


def gen_df_filt(df):
    """ This function iterates over the dataset while applying the
    'process_commits_row' function to each row. The result is a list of
    dictionaries which is finally returned as a DataFrame. """
    all_features = []
    for row in df.iterrows():
        for i in range(1, len(row[1]) - 1):
            try:
                all_features.append(process_commits_row(row[1][i]))
            except Exception:
                pass

    return pd.DataFrame(all_features)


def gen_group_week(filt_dataframe):
    """ This function groups the filtered datasets by week """
    df_commits_temp = filt_dataframe.sort_values(
        by='date')[['date', 'sha', 'repo', 'author', 'message']]
    df_commits_temp.date = df_commits_temp.date.apply(
        lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
    df_commits_temp['week'] = df_commits_temp['date'].dt.strftime('%W-%Y')

    df_commits_temp = df_commits_temp.groupby('week').agg(
        {
            'sha': 'count',
            'repo': 'nunique',
            'author': 'nunique'}).reset_index().rename(
            columns={
                'sha': 'n_commits',
                'author': 'n_participants',
                'repo': 'n_repos'})

    df_commits_temp['week_num'] = \
        df_commits_temp['week'].str.split('-').str[0].astype(int)
    df_commits_temp['year'] = \
        df_commits_temp['week'].str.split('-').str[1].astype(int)
    df_commits_temp = df_commits_temp.sort_values(by=['year', 'week_num'])

    return df_commits_temp


def gen_group_year(filt_dataframe):
    """ This function groups the filtered datasets by year """
    df_commits_temp = filt_dataframe.sort_values(
        by='date')[['date', 'sha', 'repo', 'author', 'message']]
    df_commits_temp['date'] = pd.to_datetime(df_commits_temp['date'])
    df_commits_temp['year'] = df_commits_temp['date'].dt.year

    df_commits_temp = df_commits_temp.groupby('year').agg(
        {
            'sha': 'count',
            'repo': 'nunique',
            'author': 'nunique'}).reset_index().rename(
            columns={
                'sha': 'n_commits',
                'author': 'n_participants',
                'repo': 'n_repos'})

    return df_commits_temp


def gen_totals(dev_dataframe):
    """ This function prints data for the table in the report"""
    for author in dev_dataframe.author.unique():
        total = dev_dataframe[
            dev_dataframe.author == author].sum()['commit_count']
        print(f"Author: {author}")
        print(f"Total: {total}")


# Charts
def chart_user_activity(df, protocol):
    """ Make bar chart for user activity over time """
    fig, ax1 = plt.subplots(figsize=(20, 8))

    # You can adjust the number of colors as needed
    colors = sns.color_palette('tab10', len(df['year'].unique()))

    # Create a dictionary mapping each unique year to a color
    year_colors = dict(zip(df['year'].unique(), colors))

    sns.barplot(
        df,
        x='week',
        y='n_commits',
        hue='year',
        ax=ax1,
        dodge=False,
        palette=year_colors)
    ax1.set_xlabel('Weeks', fontsize=18)
    ax1.set_ylabel('Commits', fontsize=18)
    ax1.tick_params('y', labelsize=12)
    ax1.set_title(f"{protocol} Developers Activity Over Time", fontsize=30)
    ax1.grid(axis='y', which='major', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    sns.lineplot(
        df,
        x='week',
        y='n_participants',
        ax=ax2,
        linestyle='--',
        marker='o',
        color='r',
        alpha=0.5)
    ax2.set_ylabel('Number of Participants', fontsize=18, color='r')
    ax2.tick_params('y', colors='r', labelsize=12)
    ax2.xaxis.set_ticks(df['week'][::10])

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Remove year part from x-tick labels
    labels = [item.get_text().split('-')[0] for item in ax1.get_xticklabels()]
    ax1.set_xticklabels(labels)
    ax1.tick_params('x', labelsize=12)

    ax2.set_ylim(bottom=0)

    plt.show()


def bar_activity_n_repos(datasets, main_title, subplot_titles):
    """ Make bar char for activity and repos over time """
    num_datasets = len(datasets)
    num_rows = (num_datasets + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 7*num_rows))
    fig.suptitle(main_title, fontsize=30)

    for i, (df, title) in enumerate(zip(datasets, subplot_titles)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        sns.barplot(
            x="year",
            y="n_commits",
            hue='n_repos',
            data=df,
            dodge=False,
            palette='Blues',
            ax=ax)

        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Commits', fontsize=14)
        ax.set_xlabel('Years', fontsize=14)
        ax.set_title(title, fontsize=20)
        ax.grid(axis='y', which='major', linestyle='--', linewidth=0.5)
        legend = ax.legend()
        legend.set_title("Repositories")  # Setting legend title

        # Add a red horizontal line for the mean of commits
        mean_commits = df['n_commits'].mean()
        ax.axhline(y=mean_commits, color='red', linestyle='--', label='mean')

        # Annotate mean value above the line
        ax.text(
            0.05,
            mean_commits * 1.05,
            f'Mean: {mean_commits:.2f}',
            color='red',
            fontsize=12,
            va='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot title position
    plt.subplots_adjust(hspace=0.2, wspace=0.2)  # Add gap between charts
    plt.show()


def bar_developers(df, protocol):
    """ Make bar chart with developers activity"""
    plt.figure(figsize=(15, 8))

    # You can adjust the number of colors as needed
    colors = sns.color_palette('Blues', len(df['year'].unique()))

    # Create a dictionary mapping each unique year to a color
    year_colors = dict(zip(df.sort_values(
        by='year', ascending=True)['year'].unique(), colors))

    ax = sns.barplot(
        data=df,
        x='author',
        y='commit_count',
        hue='year',
        palette=year_colors)
    ax.set_xlabel('Weeks', fontsize=18)
    ax.set_ylabel('Commits', fontsize=18)
    ax.tick_params('y', labelsize=12)
    ax.tick_params('x', labelsize=12, rotation=45)
    ax.set_title(f"{protocol}: Commits per Author Each Year", fontsize=30)
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)

    plt.xlabel('Developer')
    plt.ylabel('Number of Commits')
    plt.legend(title='Year')
    plt.show()


def bar_rank_developers(df_filt, protocol):
    """ Rank developers """
    plt.figure(figsize=(15, 8))
    df_filt['date'] = pd.to_datetime(df_filt['date'])
    df_filt['year'] = df_filt['date'].dt.year
    df = df_filt.groupby([
        'author', 'year']).size().reset_index(name='commit_count')

    idx = df.groupby('year')['commit_count'].idxmax()

    df = df.loc[idx].reset_index(drop=True)

    # You can adjust the number of colors as needed
    colors = sns.color_palette('Blues', len(df['author'].unique()))

    # Create a dictionary mapping each unique year to a color
    year_colors = dict(zip(df['author'].unique(), colors))

    ax = sns.barplot(
        data=df,
        x='year',
        y='commit_count',
        hue='author',
        palette=year_colors)
    ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel('Number of Commits', fontsize=18)
    ax.tick_params('y', labelsize=12)
    ax.tick_params('x', labelsize=12, rotation=45)
    ax.set_title(f"{protocol}: Rank Developers per Year", fontsize=30)
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)

    # Add values on top of bars, skipping zeros
    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.annotate(f'{height}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontsize=12, color='black', weight='bold')

    plt.legend(title='Developer')
    plt.show()


def bar_rank_repos(df_filt, protocol, head=True):

    """ Make bar chart for user activity over time """
    fig, ax1 = plt.subplots(figsize=(20, 8))

    if head:
        df_filt = df_filt.head(10)

    sns.barplot(
        df_filt,
        x='repo',
        y='count',
        ax=ax1,
        dodge=False)
    ax1.set_xlabel('Repositories', fontsize=18)
    ax1.set_ylabel('Number of Commits', fontsize=18)
    ax1.tick_params('y', labelsize=12)
    ax1.tick_params('x', labelsize=12, rotation=45)
    ax1.set_title(f"{protocol}: Rank Repositories", fontsize=30)
    ax1.grid(axis='y', which='major', linestyle='--', linewidth=0.5)
    ax1.spines['top'].set_visible(False)

    # Add values on top of bars, skipping zeros
    for p in ax1.patches:
        height = int(p.get_height())
        if height > 0:
            ax1.annotate(f'{height}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontsize=12, color='black', weight='bold')

    plt.show()


def gen_correlations(df_merged, protocol, date):
    """ Correlate commits and Close price per day"""
    plt.figure(figsize=(15, 8))

    ax = sns.jointplot(
        data=df_merged,
        x='sha',
        y='Close',
        hue='year',
        palette='tab10')
    ax.set_axis_labels('Number of Commits', 'Close Price (USD)')
    plt.show()


def gen_correlations_over_time(df, protocol, spacing):
    fig, ax1 = plt.subplots(figsize=(20, 8))

    sns.barplot(
        data=df,
        x='weekYear',
        y='sha',
        hue='year',
        ax=ax1,
        palette='tab10')

    ax1.set_xlabel('Weeks', fontsize=18)
    ax1.set_ylabel('Number of Commits', fontsize=18)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_title(
        f"{protocol}: Commits and Token Price Over Time", fontsize=30)
    ax1.grid(axis='y', which='major', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    sns.lineplot(
        data=df,
        x='weekYear',
        y='Close',
        err_style=None,
        ax=ax2,
        linestyle='--',
        marker='o',
        color='r')

    ax2.set_ylabel('Close Price (USD)', fontsize=18, color='r')
    ax2.tick_params(axis='y', colors='r', labelsize=12)
    ax2.xaxis.set_ticks(df['weekYear'][::spacing])

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Remove year part from x-tick labels
    labels = [item.get_text().split('-')[0] for item in ax1.get_xticklabels()]
    ax1.set_xticklabels(labels)
    ax1.tick_params('x', labelsize=12)

    ax2.set_ylim(bottom=0)

    plt.show()
