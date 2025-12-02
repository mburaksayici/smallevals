"""Plotly Dash UI application for viewing and analyzing retrieval evaluation results."""

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
from pathlib import Path
import json
import base64
from io import StringIO

from smallevals.utils.results_manager import (
    list_results,
    load_result,
    get_result_metadata,
    RESULTS_DIR
)

from smallevals.ui_dash.report_generator import generate_html_report
from smallevals.eval.analysis import (
    analyze_chunk_length,
    analyze_word_char_ratio,
    analyze_query_similarity,
    identify_devil_chunks,
)
import plotly.graph_objects as go
import plotly.express as px

from smallevals.ui_dash.components import (
    create_metric_card,
    create_version_info_card,
    create_header,
    create_loading_spinner,
    create_analysis_section,
    create_chart_container,
    create_alert,
    apply_chart_theme
)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True
)

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>smallevals - Retrieval Evaluation Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                font-family: 'Inter', sans-serif;
            }
            body {
                margin: 0;
                background-color: #f8f9fa;
            }
            .metric-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                transition: all 0.3s;
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
                background: linear-gradient(135deg, #5568d3 0%, #6a3f8f 100%);
            }
            .card {
                border: none;
                border-radius: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .nav-tabs .nav-link {
                border-radius: 8px 8px 0 0;
                color: #6b7280;
                font-weight: 500;
            }
            .nav-tabs .nav-link.active {
                color: #667eea;
                background-color: white;
                border-color: #dee2e6 #dee2e6 white;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = html.Div([
    dcc.Store(id='version-data-store'),
    dcc.Store(id='filtered-data-store'),
    dcc.Store(id='analysis-results-store', data={}),
    dcc.Store(id='table-original-data-store'),  # Store original untruncated table data
    dcc.Download(id="download-csv"),
    dcc.Download(id="download-report"),
    dcc.Download(id="download-enriched-csv"),
    
    create_header(),
    
    # Compact configuration at top
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Result", className="mb-1", style={"fontSize": "0.9rem"}),
                                dcc.Dropdown(
                                    id='version-dropdown',
                                    options=[],
                                    value=None,
                                    placeholder="Select a result...",
                                    style={"fontSize": "0.9rem"}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Div(id='version-info-container', style={"fontSize": "0.85rem", "paddingTop": "1.5rem"})
                            ], width=8)
                        ])
                    ])
                ], className="mb-3")
            ])
        ])
    ], fluid=True),
    
    # Main content - full width
    dbc.Container([
        dbc.Tabs([
            dbc.Tab(label="📊 Metrics Summary", tab_id="metrics", id="tab-metrics"),
            dbc.Tab(label="📋 Results Table", tab_id="table", id="tab-table"),
            dbc.Tab(label="📈 Rank Distribution", tab_id="distribution", id="tab-distribution"),
            dbc.Tab(label="📏 Chunk Length Analysis", tab_id="chunk-length", id="tab-chunk-length"),
            dbc.Tab(label="📊 Word-Char Ratio Analysis", tab_id="word-char", id="tab-word-char"),
            #dbc.Tab(label="🔗 Query Similarity Analysis", tab_id="query-similarity", id="tab-query-similarity"),
            dbc.Tab(label="👹 Devil Chunks Analysis", tab_id="devil-chunks", id="tab-devil-chunks"),
        ], id="main-tabs", active_tab="metrics"),
        html.Div(id='tab-content', className="mt-4")
    ], fluid=True)
], style={"minHeight": "100vh"})


# Import callbacks
from smallevals.ui_dash.callbacks import register_callbacks
register_callbacks(app)


def main():
    """Main entry point for the Dash application."""
    app.run(debug=True, host='127.0.0.1', port=8050)


if __name__ == "__main__":
    main()

