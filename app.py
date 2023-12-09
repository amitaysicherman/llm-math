import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import dash_bootstrap_components as dbc
import random
import pandas as pd

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "29rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

SCROLL_STYLE = {"maxHeight": "500px",
                "overflow-y": "scroll"}
TOTAL = 50
from db import PileNumbersDataset
from results_analyzer import ResultsAnalyzer

results_analyzer = ResultsAnalyzer()
data = {
    'correct_count': lambda: results_analyzer.correct_count_mesh,
    'common': lambda: results_analyzer.most_common_results_mesh,
    'common_wrong': lambda: results_analyzer.most_common_wrong_mesh,
    'common_wrong_count': lambda
    : results_analyzer.most_common_wrong_count_mesh,
    'common_count': lambda: results_analyzer.most_common_count_mesh,
    'sentence_count_(log)': lambda: results_analyzer.sentence_count_mesh,
    'sentence_wrong_(log)': lambda: results_analyzer.sentence_wrong_mesh
}
names = ['common', 'common_wrong', 'correct_count',
         'common_count', 'common_wrong_count', 'sentence_count_(log)',
         'sentence_wrong_(log)'
         ]
pile_number_dataset = PileNumbersDataset('assets/db')

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

sidebar = html.Div(
    [
        html.H2("LLM MATH", className="display-4"),
        html.Hr(),
        dcc.Loading(html.Div(id='click-output-main'))
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE, children=[
    dbc.Tabs(
        [
            dbc.Tab(label=n.upper().replace("_", " "), tab_id=n) for n in names
        ],
        id="tabs",
        active_tab=names[0],
    ),
    html.Div(id="tab-content",
             children=[

                 dbc.Row(dcc.Graph(id='heatmap')),
                 dbc.Row([
                     dbc.Col(dbc.Label('Min Color Threshold:')),
                     dbc.Col(dcc.Input(id='min-threshold', type='number',
                                       placeholder='Enter Min Threshold')),
                     dbc.Col(dbc.Label('Max Color Threshold:')),
                     dbc.Col(dcc.Input(id='max-threshold', type='number',
                                       placeholder='Enter Max Threshold')),
                     dbc.Col(dbc.Label('Binary Threshold:')),
                     dbc.Col(dcc.Input(id='binary-threshold', type='number',
                                       placeholder='Enter Binary Threshold')),

                     dbc.Col(dbc.Checklist(
                         options=[
                             {"label": "Color By Symmetric", "value": 1},
                         ],
                         value=[],
                         id="is-symmetric",
                         inline=True,
                         switch=True,
                     )),

                 ]),
                 html.Hr(),
                 dbc.Row(html.Div(id='click-output-details')),
             ]),
])

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(
    Output('heatmap', 'figure'),
    [Input("tabs", "active_tab"),
     Input('max-threshold', 'value'),
     Input('min-threshold', 'value'),
     Input('binary-threshold', 'value'),
     Input('is-symmetric', 'value')]
)
def update_heatmap(selected_mode, max_threshold, min_threshold,
    binary_threshold, is_symmetric):
    # Generate heatmap based on the selected mode (replace this with your logic)
    heatmap_data = data[selected_mode]()
    if len(is_symmetric):
        print("is_symmetric", is_symmetric)
        heatmap_data = pd.DataFrame(
            np.abs(heatmap_data.values - heatmap_data.T.values),
            index=heatmap_data.index, columns=heatmap_data.columns)

    if max_threshold:
        heatmap_data = heatmap_data.clip(upper=float(max_threshold))
    if min_threshold:
        heatmap_data = heatmap_data.clip(lower=float(min_threshold))
    if binary_threshold:
        binary_threshold = float(binary_threshold)
        heatmap_data = (heatmap_data > binary_threshold).astype(int)
    if "sentence" in selected_mode:
        heatmap_data = pd.DataFrame(np.log10(1 + heatmap_data.values),
                                    index=heatmap_data.index,
                                    columns=heatmap_data.columns)

    heatmap_trace = go.Heatmap(x=heatmap_data.index, y=heatmap_data.columns,
                               z=heatmap_data.values.T, colorscale='Blues')

    # Set layout
    layout = go.Layout(
        xaxis=dict(title='x', showgrid=False),
        yaxis=dict(title='y', showgrid=False),
        clickmode='event+select',  # Enable click events
        width=900,  # Set the width
        height=900,  # Set the height
    )

    # Create figure
    fig = go.Figure(data=[heatmap_trace], layout=layout)

    return fig


def get_clicked_point(x, y):
    most_common_wrong = int(results_analyzer.most_common_wrong_mesh.loc[x, y])
    all_counts = results_analyzer.all_counts_mesh.loc[x, y]
    all_counts = eval(all_counts)
    full_string_ = results_analyzer.full_string_mesh.loc[x, y]
    full_string = []
    for s in full_string_.split("<br/>"):
        full_string.append(html.Div(s))
    full_string.pop()

    db_sentences = pile_number_dataset.query(np.array([x, y, x + y]))
    db_sentences_len = len(db_sentences)
    if len(db_sentences) > 100:
        db_sentences = random.sample(db_sentences, 100)
    wrong_sentences = pile_number_dataset.query(
        np.array([x, y, most_common_wrong]))

    wrong_sentences_len = len(wrong_sentences)
    if len(wrong_sentences) > 100:
        wrong_sentences = random.sample(wrong_sentences, 100)
    key_values = []
    for name, value in data.items():
        v = int(value().loc[x, y])
        n = " ".join([x[0].upper() + x[1:] for x in name.split("_")]) + ":"
        key_values.append(
            html.P([html.Strong(n), f"{v}"]))
    main_view = html.Div([
        html.H4(f"{x}+{y}=({x + y})"),
        *key_values,
        html.P(html.Strong(f"DB Sentences ({db_sentences_len})")),
        html.P(html.Strong(f"Wrong Sentences ({wrong_sentences_len})")),
        html.P(html.Strong(f"All Counts")),
        html.Div(html.Ul([html.Li(f'Result {k}, Count {v}') for k, v in
                          all_counts.items()]))

    ])

    details_view = html.Div([
        dcc.Checklist(
            options=[
                {'label': 'Expand Queries',
                 'value': 'expand-full-string'}],
            id='expand-full-string',
            inline=True
        ),
        html.Div(full_string, id='full-string-content',
                 style={'display': 'none'}),
        dcc.Checklist(
            options=[
                {'label': f'Expand Correct DB Sentences ({db_sentences_len})',
                 'value': 'expand-db-sentences'}],
            id='expand-db-sentences',
            inline=True
        ),
        html.Div(
            html.Ul([html.Li(sentence) for sentence in db_sentences]),
            id='db-sentences-content',
            style={'display': 'none'}),
        dcc.Checklist(
            options=[
                {'label': f'Expand Wrong DB Sentences ({wrong_sentences_len})',
                 'value': 'expand-wrong-sentences'}],
            id='expand-wrong-sentences',
            inline=True
        ),
        html.Div(
            html.Ul([html.Li(sentence) for sentence in wrong_sentences]),
            id='wrong-sentences-content',
            style={'display': 'none', "maxHeight": "400px",
                   "overflow": "scroll"}),

    ])

    return main_view, details_view


@app.callback(
    Output('full-string-content', 'style'),
    [Input('expand-full-string', 'value')]
)
def toggle_full_string_visibility(value):
    if value and 'expand-full-string' in value:
        return {'display': 'block', **SCROLL_STYLE}
    return {'display': 'none'}


@app.callback(
    Output('db-sentences-content', 'style'),
    [Input('expand-db-sentences', 'value')]
)
def toggle_db_sentences_visibility(value):
    if value and 'expand-db-sentences' in value:
        return {'display': 'block', **SCROLL_STYLE}
    return {'display': 'none'}


@app.callback(
    Output('wrong-sentences-content', 'style'),
    [Input('expand-wrong-sentences', 'value')]
)
def toggle_wrong_sentences_visibility(value):
    if value and 'expand-wrong-sentences' in value:
        return {'display': 'block', **SCROLL_STYLE}
    return {'display': 'none'}


# Callback to display clicked point coordinates
@app.callback(
    [Output('click-output-main', 'children'),
     Output('click-output-details', 'children')],
    [Input('heatmap', 'clickData')]
)
def display_click_data(click_data):
    if click_data is not None:
        x_value = int(click_data['points'][0]['x'])
        y_value = int(click_data['points'][0]['y'])
        return get_clicked_point(x_value,y_value)
    else:
        return ["Click on the heatmap to get statistics", html.Div()]


# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
