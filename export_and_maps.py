def make_tree_map(df_group):
    fig = go.Figure()

    x = 0.
    y = 0.
    width = 100.
    height = 100.

    df_group = df_group[np.isfinite(df_group['TI PMPM'])]
    values = list(df_group["TI PMPM"])
    abs_values = [abs(val) for val in values]
    labels = list(df_group["titles”])

    normed = squarify.normalize_sizes(abs_values, width, height)
    rects = squarify.squarify(normed, x, y, width, height)

    # Choose colors from http://colorbrewer2.org/ under "Export"
    #color_brewer = []
    #for i in range(len(values)):
    #    rgb = colorsys.hsv_to_rgb(i / 300., 1.0, 1.0)
    #    color_brewer.append('rgb'+ str(tuple([round(255 * x) for x in rgb])))
    #color_brewer = ['rgb(166,206,227)', 'rgb(31,120,180)', 'rgb(178,223,138)',
    #                'rgb(51,160,44)', 'rgb(251,154,153)', 'rgb(227,26,28)']
    color_brewer = ['purple' if (val < 0) else 'green' for val in values]
    shapes = []
    annotations = []
    #counter = 0

    for r, val, label, color in zip(rects, values, labels, color_brewer):
        shapes.append(
            dict(
                type='rect',
                x0=r['x'],
                y0=r['y'],
                x1=r['x'] + r['dx'],
                y1=r['y'] + r['dy'],
                line=dict(width=2),
                fillcolor=color
            )
        )
        annotations.append(
            dict(
                x=r['x'] + (r['dx'] / 2),
                y=r['y'] + (r['dy'] / 2),
                text=label,
                showarrow=False
            )
        )

    # For hover text
    fig.add_trace(go.Scatter(
        x=[r['x'] + (r['dx'] / 2) for r in rects],
        y=[r['y'] + (r['dy'] / 2) for r in rects],
        text=[str(v) for v in values],
        mode='text',
    ))

    fig.update_layout(
        height=700,
        width=700,
       xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        shapes=shapes,
        annotations=annotations,
        hovermode='closest'
    )

    return fig



def frange(start, stop, step):
    i = start
    f_range = []
    while i < stop:
        f_range.append(i)
        i += step
    return f_range
