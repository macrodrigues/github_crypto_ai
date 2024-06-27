import plotly.graph_objects as go
import plotly


def chart_user_activity(df, protocol):
    """ Make bar chart for user activity over time """
    # Create a list of unique years and assign each year a color
    unique_years = df['year'].unique()
    year_color_map = {year: color for year, color in zip(
        unique_years, plotly.colors.qualitative.Plotly)}

    # Map the years to their respective colors
    colors = df['year'].map(year_color_map)

    data = go.Bar(
        x=df['week'],
        y=df['n_commits'],
        meta=df['year'],
        marker=dict(color=colors),
        showlegend=False,
        hovertemplate="""<br>Week: %{x}
        <br>Commits: %{y}
        <br>Year: %{meta}
        <extra></extra>""",

    )

    layout = go.Layout(
        width=1000,
        height=600,
        showlegend=True,  # Enable the legend
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,1)',  # Black plot background
        paper_bgcolor='rgba(0,0,0,1)',  # Black paper background
        title=dict(
            text=f"{protocol} Developers Activity Over Time",
            x=0.5,  # center the title
            y=0.95,  # top of the plot
            font=dict(
                family="Arial, sans-serif",
                size=30
            )
        ),
        yaxis=dict(
            title=dict(
                text='Number of Commits',
                font=dict(size=18)  # Set x-axis title font size
            ),
            showgrid=False,
            side='left'
        ),
        xaxis=dict(
            title=dict(
                text='Weeks',
                font=dict(size=18)  # Set x-axis title font size
            ),
            showgrid=False,
            tickmode='array',
            tickvals=df['week'].iloc[::8],
            ticktext=df['week'].iloc[::8].apply(lambda x: x.split("-")[0]),
            tickfont=dict(size=14)
        )
    )

    fig = go.Figure(data=data, layout=layout)

    # Add custom legend
    for year, color in year_color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            legendgroup=str(year),
            showlegend=True,
            name=str(year)
        ))

    fig.update_layout(
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color='white'  # General text color
        )
    )

    with open(f"charts/{protocol.lower()}_bar_user_activity.html",
              'w',
              encoding='utf-8') as f:
        f.write(fig.to_html(include_plotlyjs='cdn'))

    fig.show(config={'displayModeBar': False})
