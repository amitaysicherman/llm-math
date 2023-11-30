import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

load_figure_template('LUX')
TOTAL = 50
from db import PileNumbersDataset
from results_analyzer import ResultsAnalyzer

results_analyzer = ResultsAnalyzer()
data = {
    'correct_count': results_analyzer.correct_count_mesh,
    'most_common': results_analyzer.most_common_results_mesh,
    'most_common_wrong': results_analyzer.most_common_wrong_mesh,
    'most_common_wrong_count': results_analyzer.most_common_wrong_count_mesh,
    'most_common_count': results_analyzer.most_common_count_mesh,
}
names = ['most_common', 'most_common_wrong', 'correct_count',
         'most_common_count', 'most_common_wrong_count',
         ]
pile_number_dataset = PileNumbersDataset('assets/db')

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
app.layout = dbc.Container([
    html.H1("LLM Math Addition Analyzer"),
    html.Hr(),
    dbc.Tabs(
        [
            dbc.Tab(label=n.upper().replace("_", " "), tab_id=n) for n in names
        ],
        id="tabs",
        active_tab=names[0],
    ),

    html.Div(id="tab-content", className="p-4", children=[
        dbc.Row([
            dbc.Col(dcc.Graph(id='heatmap')),
            dbc.Col(dcc.Loading(html.Div(id='click-output-main')))
        ]),
        dbc.Row(dcc.Loading(html.Div(id='click-output-details')))

    ]),
]
)


@app.callback(
    Output('heatmap', 'figure'),
    [Input("tabs", "active_tab")]
)
def update_heatmap(selected_mode):
    # Generate heatmap based on the selected mode (replace this with your logic)
    heatmap_data = data[selected_mode]

    heatmap_trace = go.Heatmap(x=heatmap_data.index, y=heatmap_data.columns,
                               z=heatmap_data, colorscale='Viridis')

    # Set layout
    layout = go.Layout(
        title=f"Heatmap - Mode {selected_mode}",
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'),

        clickmode='event+select',  # Enable click events
        width=900,  # Set the width
        height=900,  # Set the height
    )

    # Create figure
    fig = go.Figure(data=[heatmap_trace], layout=layout)

    return fig


def get_clicked_point(x, y):
    correct_count = int(results_analyzer.correct_count_mesh.loc[x, y])
    most_common_count = int(results_analyzer.most_common_count_mesh.loc[x, y])
    most_common_wrong_count = int(
        results_analyzer.most_common_wrong_count_mesh.loc[
            x, y])
    most_common_wrong = int(results_analyzer.most_common_wrong_mesh.loc[x, y])
    most_common = int(results_analyzer.most_common_results_mesh.loc[x, y])
    all_counts = results_analyzer.all_counts_mesh.loc[x, y]
    all_counts = eval(all_counts)
    full_string_ = results_analyzer.full_string_mesh.loc[x, y]
    full_string = []
    for s in full_string_.split("<br/>"):
        full_string.append(html.Div(s))
    full_string.pop()
    db_sentences = pile_number_dataset.query(np.array([x, y, x + y]))
    db_sentences_len = len(db_sentences)

    wrong_sentences = pile_number_dataset.query(
        np.array([x, y, most_common_wrong]))
    wrong_sentences_len = len(wrong_sentences)

    main_view = dbc.Card(
        dbc.CardBody([
            html.H4(f"{x}+{y}=({x + y})"),
            html.P(
                [html.Strong(f"Correct Count="),
                 f"{correct_count / TOTAL:.0%}"]),
            html.P([html.Strong(f"Most Common="), f"{most_common}"]),
            html.P([html.Strong(f"Most Common Count="),
                    f"{most_common_count / TOTAL:.0%}"]),
            html.P(
                [html.Strong(f"Most Common Wrong="), f"{most_common_wrong}"]),
            html.P([html.Strong(f"Most Common Wrong Count="),
                    f"{most_common_wrong_count / TOTAL:.0%}"]),
            html.P(html.Strong(f"All Counts")),
            html.Div(html.Ul([html.Li(f'Result {k}, Count {v}') for k,v in all_counts.items()]))
        ])
    )
    details_view = dbc.Card(
        dbc.CardBody([
            html.P("Full String:", id="full-string-header",
                   style={'cursor': 'pointer', 'text-decoration': 'underline'}),
            dcc.Checklist(
                options=[
                    {'label': 'Expand Full String',
                     'value': 'expand-full-string'}],
                id='expand-full-string',
                inline=True
            ),
            html.Div(full_string, id='full-string-content',
                     style={'display': 'none'}),
            html.P("DB Sentences:", id="db-sentences-header",
                   style={'cursor': 'pointer', 'text-decoration': 'underline'}),
            dcc.Checklist(
                options=[{'label': f'Expand DB Sentences ({db_sentences_len})',
                          'value': 'expand-db-sentences'}],
                id='expand-db-sentences',
                inline=True
            ),
            html.Div(
                html.Ul([html.Li(sentence) for sentence in db_sentences]),
                     id='db-sentences-content', style={'display': 'none'}),
            html.P("Wrong Sentences:", id="wrong-sentences-header",
                   style={'cursor': 'pointer', 'text-decoration': 'underline'}),
            dcc.Checklist(
                options=[
                    {'label': f'Expand Wrong Sentences ({wrong_sentences_len})',
                     'value': 'expand-wrong-sentences'}],
                id='expand-wrong-sentences',
                inline=True
            ),
            html.Div(
                html.Ul([html.Li(sentence) for sentence in wrong_sentences]),
                id='wrong-sentences-content', style={'display': 'none'}),

        ])
    )
    return main_view, details_view


@app.callback(
    Output('full-string-content', 'style'),
    [Input('expand-full-string', 'value')]
)
def toggle_full_string_visibility(value):
    if value and 'expand-full-string' in value:
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    Output('db-sentences-content', 'style'),
    [Input('expand-db-sentences', 'value')]
)
def toggle_db_sentences_visibility(value):
    if value and 'expand-db-sentences' in value:
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    [Output('wrong-sentences-content', 'style'),
     Output('wrong-sentences-header', 'style')],
    [Input('expand-wrong-sentences', 'value')]
)
def toggle_wrong_sentences_visibility(value):
    if value and 'expand-wrong-sentences' in value:
        return {'display': 'block'}, {'text-decoration': 'none'}
    return {'display': 'none'}, {'text-decoration': 'underline'}


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
        return get_clicked_point(y_value,x_value)
    else:
        return ["Click on the heatmap to get coordinates", html.Div()]


# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
