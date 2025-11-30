"""Callbacks for the Dash dashboard interactivity."""

from dash import Input, Output, State, callback_context, html, dcc, dash_table, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import json
import ast
import base64
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px

from smallevals.utils.results_manager import list_results, load_result, get_result_metadata, RESULTS_DIR
from smallevals.ui_dash.ranking import (
    calculate_metrics_from_df,
    filter_by_rank,
    get_rank_distribution,
    rank_by_metric,
)
from smallevals.ui_dash.report_generator import generate_html_report
from smallevals.eval.analysis import (
    analyze_chunk_length,
    analyze_word_char_ratio,
    analyze_query_similarity,
    identify_devil_chunks,
)

from smallevals.ui_dash.components import (
    create_metric_card,
    create_version_info_card,
    create_chart_container,
    create_alert,
    apply_chart_theme
)


def register_callbacks(app):
    """Register all callbacks with the Dash app."""
    
    # Update version dropdown options on load
    @app.callback(
        Output('version-dropdown', 'options'),
        Output('version-dropdown', 'value'),
        Input('version-dropdown', 'id')
    )
    def update_version_options(_):
        results = list_results()
        options = [{'label': v, 'value': v} for v in results]
        value = results[0] if results else None
        return options, value
    
    # Load version data when version is selected
    @app.callback(
        Output('version-data-store', 'data'),
        Output('version-info-container', 'children'),
        Input('version-dropdown', 'value')
    )
    def load_version_data(selected_version):
        if not selected_version:
            return None, html.Div("Please select a result", style={"fontSize": "0.85rem"})
        
        try:
            result_data = load_result(selected_version)
            metadata = result_data.get("metadata", {})
            config = metadata.get("config", {})
            results_df = result_data.get("results_df")
            
            if results_df is None:
                return None, create_alert("No retrieval results found for this result.", "error")
            
            if isinstance(results_df, list):
                results_df = pd.DataFrame(results_df)
            
            # Store data
            store_data = {
                'results_df': results_df.to_dict('records'),
                'metadata': metadata,
                'selected_version': selected_version
            }
            
            # Compact version info (using config for display)
            version_info = html.Div([
                html.Span([
                    html.Strong("Model: "),
                    config.get('embedding_model', 'N/A')
                ], className="me-3"),
                html.Span([
                    html.Strong("Top-K: "),
                    str(config.get('top_k', 5))
                ], className="me-3"),
                html.Span([
                    html.Strong("Created: "),
                    metadata.get('created_at', 'N/A')
                ])
            ])
            
            return store_data, version_info
            
        except Exception as e:
            return None, create_alert(f"Error loading result: {str(e)}", "error")
    
    # Filter data - no filtering, just pass through
    @app.callback(
        Output('filtered-data-store', 'data'),
        Input('version-data-store', 'data')
    )
    def filter_data(version_data):
        if not version_data:
            return None
        
        try:
            results_df = pd.DataFrame(version_data['results_df'])
            return results_df.to_dict('records')
        except Exception as e:
            return None
    
    # Main tab content
    @app.callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'active_tab'),
        State('filtered-data-store', 'data'),
        State('version-data-store', 'data'),
        State('analysis-results-store', 'data'),
        prevent_initial_call=False
    )
    def update_tab_content(active_tab, filtered_data, version_data, analysis_results):
        if not filtered_data or not version_data:
            return dbc.Alert("Please select a version and wait for data to load.", color="info")
        
        filtered_df = pd.DataFrame(filtered_data)
        metadata = version_data.get('metadata', {})
        config = metadata.get('config', {})
        top_k = config.get('top_k', metadata.get('top_k', 5))  # Check config first, then metadata for backward compat
        
        if active_tab == 'metrics':
            return render_metrics_tab(filtered_df, metadata, top_k, version_data.get('selected_version'))
        elif active_tab == 'table':
            return render_table_tab(filtered_df, version_data.get('selected_version'))
        elif active_tab == 'distribution':
            return render_distribution_tab(filtered_df, top_k)
        elif active_tab == 'chunk-length':
            return render_chunk_length_tab(filtered_df, metadata, top_k)
        elif active_tab == 'word-char':
            return render_word_char_tab(filtered_df, metadata, top_k)
        elif active_tab == 'query-similarity':
            return render_query_similarity_tab(filtered_df, metadata, top_k)
        elif active_tab == 'devil-chunks':
            return render_devil_chunks_tab(filtered_df, metadata, top_k)
        return html.Div("Select a tab")
    
    # Analysis callbacks
    @app.callback(
        Output('analyze-chunk-length-output', 'children'),
        Input('analyze-chunk-length-btn', 'n_clicks'),
        Input('chunk-length-metric-dropdown', 'value'),
        State('version-data-store', 'data'),
        prevent_initial_call=True
    )
    def analyze_chunk_length_callback(n_clicks, selected_metric, version_data):
        if not version_data:
            return create_alert("Please select a version first.", "warning")
        
        if not n_clicks:
            return html.P("Click 'Analyze Chunk Length' button to run analysis.", className="text-muted")
        
        try:
            results_df = pd.DataFrame(version_data['results_df'])
            metadata = version_data.get('metadata', {})
            config = metadata.get('config', {})
            top_k = config.get('top_k', metadata.get('top_k', 5))  # Check config first, then metadata for backward compat
            
            df_analyzed, viz_data = analyze_chunk_length(results_df, top_k=top_k)
            
            # Calculate all metrics by segment
            segments = list(viz_data['mrr_by_segment'].keys())
            metric_values = {}
            
            for segment in segments:
                segment_df = df_analyzed[df_analyzed['chunk_size_segment'] == segment]
                if len(segment_df) > 0 and 'chunk_position' in segment_df.columns:
                    positions = segment_df['chunk_position']
                    
                    if selected_metric == "mrr":
                        reciprocal_ranks = positions.apply(lambda x: 1.0 / x if pd.notna(x) else 0.0)
                        metric_values[segment] = float(reciprocal_ranks.mean())
                    elif selected_metric == "hit_rate":
                        found_in_topk = (positions.notna() & (positions <= top_k)).sum()
                        metric_values[segment] = float(found_in_topk / len(segment_df))
                    elif selected_metric == "precision":
                        found_in_topk = (positions.notna() & (positions <= top_k)).sum()
                        metric_values[segment] = float(found_in_topk / len(segment_df))
                    elif selected_metric == "recall":
                        found_in_topk = (positions.notna() & (positions <= top_k)).sum()
                        metric_values[segment] = float(found_in_topk / len(segment_df))
            
            values = [metric_values.get(seg, 0) for seg in segments]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=segments, y=values, name=selected_metric.upper(), marker_color='#667eea'))
            fig = apply_chart_theme(fig)
            fig.update_layout(
                title=f'{selected_metric.upper()} by Chunk Size Segment',
                xaxis_title='Segment',
                yaxis_title=selected_metric.upper()
            )
            
            return html.Div([
                create_alert("Chunk length analysis complete!", "success"),
                create_chart_container(fig, f"{selected_metric.upper()} by Chunk Size Segment")
            ])
        except Exception as e:
            return create_alert(f"Error: {str(e)}", "error")
    
    @app.callback(
        Output('analyze-word-char-output', 'children'),
        Input('analyze-word-char-btn', 'n_clicks'),
        State('version-data-store', 'data'),
        prevent_initial_call=True
    )
    def analyze_word_char_callback(n_clicks, version_data):
        if not version_data:
            return create_alert("Please select a version first.", "warning")
        
        try:
            results_df = pd.DataFrame(version_data['results_df'])
            metadata = version_data.get('metadata', {})
            config = metadata.get('config', {})
            top_k = config.get('top_k', metadata.get('top_k', 5))  # Check config first, then metadata for backward compat
            
            df_analyzed, viz_data = analyze_word_char_ratio(results_df, top_k=top_k)
            
            # Calculate average ratio
            avg_ratio = df_analyzed['word_char_ratio'].mean() if 'word_char_ratio' in df_analyzed.columns else 0.0
            
            # Calculate word/stripped char ratio (from token density analysis)
            word_stripped_char_ratio = 0.0
            if 'chunk' in df_analyzed.columns:
                import re
                def _count_words(text):
                    if pd.isna(text):
                        return 0
                    return len(re.findall(r'\b\w+\b', str(text)))
                def _strip_characters(text):
                    if pd.isna(text):
                        return ""
                    return re.sub(r'[^\w]', '', str(text))
                word_counts = df_analyzed['chunk'].astype(str).apply(_count_words)
                stripped_char_counts = df_analyzed['chunk'].astype(str).apply(_strip_characters).str.len()
                word_stripped_char_ratio = (word_counts / stripped_char_counts).mean() if stripped_char_counts.sum() > 0 else 0.0
            
            # Get low ratio chunks (below 0.5)
            low_ratio_chunks = df_analyzed[df_analyzed['word_char_ratio'] < 0.5].copy()
            low_ratio_chunks = low_ratio_chunks.sort_values('word_char_ratio', ascending=True)
            
            segments = list(viz_data['mrr_by_segment'].keys())
            mrr_values = list(viz_data['mrr_by_segment'].values())
            hit_rate_values = list(viz_data['hit_rate_by_segment'].values())
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=segments, y=mrr_values, name='MRR', marker_color='#667eea'))
            fig.add_trace(go.Bar(x=segments, y=hit_rate_values, name='Hit Rate', marker_color='#f5576c'))
            fig = apply_chart_theme(fig)
            fig.update_layout(
                title='MRR and Hit Rate by Word-Char Ratio Segment',
                xaxis_title='Segment',
                yaxis_title='Score',
                barmode='group'
            )
            
            low_ratio_table = None
            if len(low_ratio_chunks) > 0:
                low_ratio_display = low_ratio_chunks[['chunk_id', 'chunk', 'word_char_ratio']].head(50)
                low_ratio_display['chunk'] = low_ratio_display['chunk'].astype(str).str[:200] + '...'
                low_ratio_table = html.Div([
                    html.H5("Low Ratio Chunks (Ratio < 0.5)", className="mb-3"),
                    dash_table.DataTable(
                        data=low_ratio_display.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in low_ratio_display.columns],
                        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
                        style_header={'backgroundColor': '#667eea', 'color': 'white', 'fontWeight': '600'},
                        style_data={'wordWrap': 'break-word'},
                        page_size=20
                    )
                ])
            
            metrics_row = dbc.Row([
                dbc.Col(create_metric_card("Avg Word/Char Ratio", f"{avg_ratio:.3f}"), width=6),
                dbc.Col(create_metric_card("Avg Word/Stripped Char Ratio", f"{word_stripped_char_ratio:.3f}"), width=6),
            ], className="mb-4")
            
            return html.Div([
                create_alert("Word-char ratio analysis complete!", "success"),
                metrics_row,
                create_chart_container(fig, "Word-Character Ratio Analysis"),
                low_ratio_table
            ])
        except Exception as e:
            return create_alert(f"Error: {str(e)}", "error")
    
    @app.callback(
        Output('analyze-query-similarity-output', 'children'),
        Input('analyze-query-similarity-btn', 'n_clicks'),
        State('version-data-store', 'data'),
        prevent_initial_call=True
    )
    def analyze_query_similarity_callback(n_clicks, version_data):
        if not version_data:
            return create_alert("Please select a version first.", "warning")
        
        try:
            results_df = pd.DataFrame(version_data['results_df'])
            metadata = version_data.get('metadata', {})
            config = metadata.get('config', {})
            top_k = config.get('top_k', metadata.get('top_k', 5))  # Check config first, then metadata for backward compat
            
            df_analyzed, viz_data = analyze_query_similarity(results_df, top_k=top_k)
            
            if 'group_metrics' in viz_data:
                groups_data = [
                    {
                        'Group ID': gid,
                        'MRR': metrics['mrr'],
                        'Hit Rate': metrics['hit_rate'],
                        'Count': metrics['count'],
                        'Avg Similarity': metrics.get('avg_similarity', 0.0)
                    }
                    for gid, metrics in viz_data['group_metrics'].items()
                ]
                groups_df = pd.DataFrame(groups_data)
                
                table = dash_table.DataTable(
                    data=groups_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in groups_df.columns],
                    style_cell={'textAlign': 'left'},
                    style_header={'backgroundColor': '#667eea', 'color': 'white', 'fontWeight': '600'}
                )
                
                alerts = [create_alert("Query similarity analysis complete!", "success")]
                if viz_data.get('low_performing_groups'):
                    alerts.append(create_alert(
                        f"Found {len(viz_data['low_performing_groups'])} low-performing query groups",
                        "warning"
                    ))
                
                return html.Div(alerts + [table])
        except Exception as e:
            return create_alert(f"Error: {str(e)}", "error")
    
    @app.callback(
        Output('identify-devil-chunks-output', 'children'),
        Input('identify-devil-chunks-btn', 'n_clicks'),
        State('version-data-store', 'data'),
        prevent_initial_call=True
    )
    def identify_devil_chunks_callback(n_clicks, version_data):
        if not n_clicks:
            return html.P("Click the button to identify devil chunks.", className="text-muted")
        
        if not version_data:
            return create_alert("Please select a result first.", "warning")
        
        try:
            results_df = pd.DataFrame(version_data['results_df'])
            metadata = version_data.get('metadata', {})
            config = metadata.get('config', {})
            top_k = config.get('top_k', metadata.get('top_k', 5))
            
            df_analyzed, viz_data = identify_devil_chunks(results_df, top_k=top_k)
            
            if not viz_data.get('devil_chunks'):
                return html.Div([
                    create_alert("No devil chunks found. All chunks are performing well!", "info"),
                    html.P(f"Analyzed {len(results_df)} queries."),
                ])
            
            # Get chunk texts and first retrieved doc examples
            chunk_texts = {}
            first_retrieved_examples = {}
            questions_examples = {}
            
            for chunk_id in viz_data['devil_chunks'].keys():
                chunk_id_str = str(chunk_id)
                chunk_rows = df_analyzed[df_analyzed['chunk_id'].astype(str) == chunk_id_str]
                
                if len(chunk_rows) > 0:
                    chunk_texts[chunk_id] = chunk_rows.iloc[0].get('chunk', 'N/A')
                
                # Find first occurrence where this chunk appears in retrieved results
                for idx, row in df_analyzed.iterrows():
                    retrieved_ids = row.get('retrieved_ids', [])
                    if isinstance(retrieved_ids, str):
                        try:
                            retrieved_ids = ast.literal_eval(retrieved_ids)
                        except (ValueError, SyntaxError):
                            try:
                                retrieved_ids = json.loads(retrieved_ids)
                            except:
                                retrieved_ids = []
                    
                    if not isinstance(retrieved_ids, list):
                        retrieved_ids = []
                    
                    # Check if this devil chunk appears in retrieved results
                    retrieved_ids_str = [str(rid) for rid in retrieved_ids]
                    if chunk_id_str in retrieved_ids_str and chunk_id not in first_retrieved_examples:
                        retrieved_docs = row.get('retrieved_docs', [])
                        if isinstance(retrieved_docs, str):
                            try:
                                retrieved_docs = ast.literal_eval(retrieved_docs)
                            except (ValueError, SyntaxError):
                                try:
                                    retrieved_docs = json.loads(retrieved_docs)
                                except:
                                    retrieved_docs = []
                        
                        if not isinstance(retrieved_docs, list):
                            retrieved_docs = []
                        
                        # Get the first retrieved doc (rank 1)
                        if retrieved_docs and len(retrieved_docs) > 0:
                            first_retrieved_examples[chunk_id] = str(retrieved_docs[0])
                            questions_examples[chunk_id] = row.get('question', 'N/A')
                            break
            
            devil_data = [
                {
                    'Chunk ID': chunk_id,
                    'Chunk': chunk_texts.get(chunk_id, 'N/A')[:200] + '...' if len(chunk_texts.get(chunk_id, '')) > 200 else chunk_texts.get(chunk_id, 'N/A'),
                    'Question': questions_examples.get(chunk_id, 'N/A')[:150] + '...' if len(questions_examples.get(chunk_id, '')) > 150 else questions_examples.get(chunk_id, 'N/A'),
                    'First Retrieved Doc': first_retrieved_examples.get(chunk_id, 'N/A')[:200] + '...' if len(str(first_retrieved_examples.get(chunk_id, ''))) > 200 else first_retrieved_examples.get(chunk_id, 'N/A'),
                    'Devil Score': stats['score'],
                    'Appearances': stats['appearances'],
                    'Devil Count': stats['devil_count']
                }
                for chunk_id, stats in viz_data['devil_chunks'].items()
            ]
            devil_df = pd.DataFrame(devil_data)
            
            # Chart
            chunk_ids = list(viz_data['devil_chunks'].keys())[:10]
            scores = [viz_data['devil_chunks'][cid]['score'] for cid in chunk_ids]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[str(cid) for cid in chunk_ids], y=scores, marker_color='#ff6b6b'))
            fig = apply_chart_theme(fig)
            fig.update_layout(
                title='Top 10 Devil Chunks by Score',
                xaxis_title='Chunk ID',
                yaxis_title='Devil Score'
            )
            
            return html.Div([
                create_alert("Devil chunks identification complete!", "success"),
                html.P(f"Total Devil Chunks: {viz_data.get('total_devil_chunks', 0)}"),
                dash_table.DataTable(
                    data=devil_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in devil_df.columns],
                    style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
                    style_header={'backgroundColor': '#667eea', 'color': 'white', 'fontWeight': '600'},
                    style_data={'wordWrap': 'break-word'},
                    page_size=20
                ),
                create_chart_container(fig, "Top 10 Devil Chunks")
            ])
        except Exception as e:
            return create_alert(f"Error: {str(e)}", "error")
    
    # Download callbacks
    @app.callback(
        Output("download-csv", "data"),
        Input("download-csv-btn", "n_clicks"),
        State('filtered-data-store', 'data'),
        State('version-data-store', 'data'),
        prevent_initial_call=True
    )
    def download_csv(n_clicks, filtered_data, version_data):
        if not filtered_data or not n_clicks:
            return no_update
        
        df = pd.DataFrame(filtered_data)
        csv_string = df.to_csv(index=False)
        return dict(content=csv_string, filename=f"filtered_results_{version_data.get('selected_version', 'unknown')}.csv")
    
    @app.callback(
        Output("download-report", "data"),
        Input("download-report-btn", "n_clicks"),
        State('filtered-data-store', 'data'),
        State('version-data-store', 'data'),
        prevent_initial_call=True
    )
    def download_report(n_clicks, filtered_data, version_data):
        if not filtered_data or not n_clicks:
            return no_update
        
        try:
            filtered_df = pd.DataFrame(filtered_data)
            metadata = version_data.get('metadata', {})
            top_k = metadata.get('top_k', 5)
            metrics = calculate_metrics_from_df(filtered_df, top_k=top_k)
            
            html_content = generate_html_report(filtered_df, metrics, version_metadata=metadata, top_k=top_k)
            return dict(content=html_content, filename=f"report_{version_data.get('selected_version', 'unknown')}.html")
        except Exception as e:
            return no_update
    
    @app.callback(
        Output("download-enriched-csv", "data"),
        Input("download-enriched-csv-btn", "n_clicks"),
        State('version-data-store', 'data'),
        prevent_initial_call=True
    )
    def download_enriched_csv(n_clicks, version_data):
        if not version_data:
            return no_update
        
        try:
            results_df = pd.DataFrame(version_data['results_df'])
            csv_string = results_df.to_csv(index=False)
            return dict(content=csv_string, filename=f"enriched_results_{version_data.get('selected_version', 'unknown')}.csv")
        except Exception as e:
            return no_update
    
    # Table search/sort callback
    @app.callback(
        Output("table-display", "children"),
        Output('table-original-data-store', 'data'),
        Input("table-search", "value"),
        Input("table-sort", "value"),
        Input("table-rows", "value"),
        State('filtered-data-store', 'data')
    )
    def update_table_display(search_term, sort_by, num_rows, filtered_data):
        if not filtered_data:
            return html.Div("No data available"), None
        
        display_df = pd.DataFrame(filtered_data)
        
        # Apply search
        if search_term:
            mask = (
                display_df['question'].str.contains(search_term, case=False, na=False) |
                display_df['answer'].str.contains(search_term, case=False, na=False) |
                display_df['chunk'].str.contains(search_term, case=False, na=False)
            )
            display_df = display_df[mask]
        
        # Apply sorting
        if sort_by == "position_asc":
            display_df = rank_by_metric(display_df, metric="position", ascending=True)
        elif sort_by == "position_desc":
            display_df = rank_by_metric(display_df, metric="position", ascending=False)
        elif sort_by == "mrr":
            display_df = rank_by_metric(display_df, metric="mrr", ascending=False)
        elif sort_by == "hit_rate":
            display_df = rank_by_metric(display_df, metric="hit_rate", ascending=False)
        
        num_rows = num_rows or 50
        display_df = display_df.head(num_rows)
        
        # STORE ORIGINAL DATA BEFORE TRUNCATION
        # Save original untruncated dataframe for modal display
        original_df = display_df.copy()
        original_data = original_df.to_dict('records')
        
        # Select only required columns
        required_cols = ['chunk_id', 'chunk', 'question', 'answer']
        # Use chunk_position as retrieved_rank if available
        if 'chunk_position' in display_df.columns:
            display_df = display_df.copy()
            display_df['retrieved_rank'] = display_df['chunk_position']
            required_cols.append('retrieved_rank')
        
        # Filter to only show columns that exist
        available_cols = [col for col in required_cols if col in display_df.columns]
        display_df = display_df[available_cols]
        
        # Truncate long text for display
        if 'chunk' in display_df.columns:
            display_df = display_df.copy()
            display_df['chunk'] = display_df['chunk'].astype(str).str[:300] + '...'
        if 'question' in display_df.columns:
            display_df = display_df.copy()
            display_df['question'] = display_df['question'].astype(str).str[:200] + '...'
        if 'answer' in display_df.columns:
            display_df = display_df.copy()
            display_df['answer'] = display_df['answer'].astype(str).str[:200] + '...'
        
        # Define column widths using style_cell_conditional
        column_widths = {
            'chunk_id': '10%',
            'chunk': '35%',
            'question': '25%',
            'answer': '20%',
            'retrieved_rank': '10%'
        }
        
        # Build style_cell_conditional for column widths and clickable styling
        style_cell_conditional = [
            {
                'if': {'column_id': col},
                'width': width
            }
            for col, width in column_widths.items()
            if col in display_df.columns
        ]
        
        # Add clickable styling for chunk column
        if 'chunk' in display_df.columns:
            style_cell_conditional.append({
                'if': {'column_id': 'chunk'},
                'cursor': 'pointer',
                'textDecoration': 'underline',
                'textDecorationStyle': 'dotted'
            })
        
        return html.Div([
            html.P(f"Showing {len(display_df)} of {len(pd.DataFrame(filtered_data))} results"),
            dash_table.DataTable(
                id='results-table',
                data=display_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in display_df.columns],
                page_size=20,
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                style_cell_conditional=style_cell_conditional,
                style_header={'backgroundColor': '#667eea', 'color': 'white', 'fontWeight': '600'},
                style_table={'width': '100%', 'overflowX': 'auto'},
                style_data={'wordWrap': 'break-word'},
                row_selectable='single',
                active_cell={'row': 0, 'column': 0}
            ),
            # Chunk text modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Full Chunk Text")),
                    dbc.ModalBody(id="chunk-modal-body", style={"maxHeight": "70vh", "overflowY": "auto"}),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-chunk-modal", className="ms-auto", n_clicks=0)
                    ),
                ],
                id="chunk-modal",
                is_open=False,
                size="lg",
                scrollable=True,
            ),
            # Row details modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Retrieval Result Details")),
                    dbc.ModalBody(id="row-details-modal-body", style={"maxHeight": "70vh", "overflowY": "auto"}),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-row-modal", className="ms-auto", n_clicks=0)
                    ),
                ],
                id="row-details-modal",
                is_open=False,
                size="xl",
                scrollable=True,
            ),
        ]), original_data
    
    # Chunk cell click callback - opens chunk modal
    @app.callback(
        Output("chunk-modal", "is_open"),
        Output("chunk-modal-body", "children"),
        Input("results-table", "active_cell"),
        Input("close-chunk-modal", "n_clicks"),
        State("table-original-data-store", "data"),
        State("results-table", "data"),
        prevent_initial_call=True
    )
    def toggle_chunk_modal(active_cell, close_clicks, original_data, table_data):
        ctx = callback_context
        if not ctx.triggered:
            return False, ""
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Close modal if close button clicked
        if trigger_id == "close-chunk-modal":
            return False, ""
        
        # Open modal if chunk cell clicked
        if active_cell and original_data and table_data:
            column_id = active_cell.get('column_id')
            row = active_cell.get('row')
            
            # Only open if chunk column was clicked
            if column_id == 'chunk' and row is not None and row < len(table_data):
                # Find matching row in original_data by chunk_id
                current_row_data = table_data[row]
                chunk_id = current_row_data.get('chunk_id')
                
                # Find matching row in original_data
                matching_row = None
                for orig_row in original_data:
                    if orig_row.get('chunk_id') == chunk_id:
                        matching_row = orig_row
                        break
                
                if matching_row:
                    chunk_text = matching_row.get('chunk', '')
                    # Handle case where chunk might be stored as string representation
                    if isinstance(chunk_text, str):
                        # Remove truncation if present
                        chunk_text = chunk_text.replace('...', '')
                    
                    modal_content = html.Div([
                        html.P(chunk_text, style={"whiteSpace": "pre-wrap", "wordWrap": "break-word"})
                    ])
                    return True, modal_content
        
        return False, ""
    
    # Row click callback - opens row details modal
    @app.callback(
        Output("row-details-modal", "is_open"),
        Output("row-details-modal-body", "children"),
        Input("results-table", "active_cell"),
        Input("results-table", "selected_rows"),
        Input("close-row-modal", "n_clicks"),
        State("table-original-data-store", "data"),
        State("results-table", "data"),
        prevent_initial_call=True
    )
    def toggle_row_details_modal(active_cell, selected_rows, close_clicks, original_data, table_data):
        ctx = callback_context
        if not ctx.triggered:
            return False, ""
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Close modal if close button clicked
        if trigger_id == "close-row-modal":
            return False, ""
        
        # Determine row data from active_cell or selected_rows
        row_data = None
        if active_cell and table_data and original_data:
            column_id = active_cell.get('column_id')
            # Don't open row modal if chunk column clicked (that opens chunk modal)
            if column_id != 'chunk':
                row_idx = active_cell.get('row')
                if row_idx is not None and row_idx < len(table_data):
                    # Find matching row in original_data by chunk_id
                    current_row_data = table_data[row_idx]
                    chunk_id = current_row_data.get('chunk_id')
                    
                    # Find matching row in original_data
                    for orig_row in original_data:
                        if orig_row.get('chunk_id') == chunk_id:
                            row_data = orig_row
                            break
        elif selected_rows and len(selected_rows) > 0 and table_data and original_data:
            row_idx = selected_rows[0]
            if row_idx < len(table_data):
                # Find matching row in original_data by chunk_id
                current_row_data = table_data[row_idx]
                chunk_id = current_row_data.get('chunk_id')
                
                # Find matching row in original_data
                for orig_row in original_data:
                    if orig_row.get('chunk_id') == chunk_id:
                        row_data = orig_row
                        break
        
        # Open modal if row clicked (and not chunk column)
        if row_data:
            # Extract data
            chunk_text = row_data.get('chunk', '')
            question = row_data.get('question', '')
            answer = row_data.get('answer', '')
            chunk_id = row_data.get('chunk_id', 'N/A')
            chunk_position = row_data.get('chunk_position')
            retrieved_docs = row_data.get('retrieved_docs', [])
            retrieved_ids = row_data.get('retrieved_ids', [])
            
            # Parse retrieved_docs if it's stored as string
            if isinstance(retrieved_docs, str):
                try:
                    retrieved_docs = json.loads(retrieved_docs)
                except (json.JSONDecodeError, TypeError):
                    # If JSON parsing fails, treat as empty list
                    retrieved_docs = []
            
            # Parse retrieved_ids if it's stored as string
            if isinstance(retrieved_ids, str):
                try:
                    retrieved_ids = json.loads(retrieved_ids)
                except (json.JSONDecodeError, TypeError):
                    # If JSON parsing fails, treat as empty list
                    retrieved_ids = []
            
            # Ensure lists
            if not isinstance(retrieved_docs, list):
                retrieved_docs = []
            if not isinstance(retrieved_ids, list):
                retrieved_ids = []
            
            # Build retrieved chunks display
            retrieved_chunks_html = []
            if retrieved_docs:
                for idx, (doc, doc_id) in enumerate(zip(retrieved_docs, retrieved_ids), start=1):
                    is_correct = (chunk_position is not None and idx == chunk_position) or (doc_id == chunk_id)
                    highlight_style = {
                        "backgroundColor": "#d4edda",
                        "border": "2px solid #28a745",
                        "padding": "10px",
                        "marginBottom": "10px",
                        "borderRadius": "5px"
                    } if is_correct else {
                        "backgroundColor": "#f8f9fa",
                        "border": "1px solid #dee2e6",
                        "padding": "10px",
                        "marginBottom": "10px",
                        "borderRadius": "5px"
                    }
                    
                    retrieved_chunks_html.append(
                        html.Div([
                            html.H6(f"Rank {idx}: {doc_id if doc_id else 'N/A'}", 
                                   style={"fontWeight": "bold", "marginBottom": "5px"}),
                            html.P(doc, style={"whiteSpace": "pre-wrap", "wordWrap": "break-word", "margin": "0"})
                        ], style=highlight_style)
                    )
            else:
                retrieved_chunks_html.append(html.P("No retrieved chunks available.", className="text-muted"))
            
            # Build modal content
            modal_content = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H5("Chunk ID", className="mb-2"),
                        html.P(chunk_id, className="text-muted")
                    ], width=6),
                    dbc.Col([
                        html.H5("Retrieved Position", className="mb-2"),
                        html.P(
                            f"Position {chunk_position}" if chunk_position else "Not found",
                            className="text-muted"
                        )
                    ], width=6)
                ], className="mb-4"),
                html.Hr(),
                html.H5("Chunk Text", className="mb-2"),
                html.Div([
                    html.P(chunk_text, style={"whiteSpace": "pre-wrap", "wordWrap": "break-word"})
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "marginBottom": "20px"
                }),
                html.H5("Question", className="mb-2"),
                html.Div([
                    html.P(question, style={"whiteSpace": "pre-wrap", "wordWrap": "break-word"})
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "marginBottom": "20px"
                }),
                html.H5("Answer", className="mb-2"),
                html.Div([
                    html.P(answer, style={"whiteSpace": "pre-wrap", "wordWrap": "break-word"})
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "padding": "15px",
                    "borderRadius": "5px",
                    "marginBottom": "20px"
                }),
                html.H5("Retrieved Chunks", className="mb-3"),
                html.Div(retrieved_chunks_html)
            ])
            
            return True, modal_content
        
        return False, ""


def render_metrics_tab(filtered_df, metadata, top_k, selected_version):
    """Render the metrics summary tab."""
    metrics = calculate_metrics_from_df(filtered_df, top_k=top_k)
    
    hit_rate_key = f"hit_rate@{top_k}"
    precision_key = f"precision@{top_k}"
    recall_key = f"recall@{top_k}"
    
    # Calculate word to char ratio
    word_char_ratio = 0.0
    if 'chunk' in filtered_df.columns:
        import re
        def _count_words(text):
            if pd.isna(text):
                return 0
            return len(re.findall(r'\b\w+\b', str(text)))
        word_counts = filtered_df['chunk'].astype(str).apply(_count_words)
        char_counts = filtered_df['chunk'].astype(str).str.len()
        word_char_ratio = (word_counts / char_counts).mean() if char_counts.sum() > 0 else 0.0
    
    # Count devil chunks (chunks that appear frequently but not at rank 1)
    devil_chunks_count = 0
    if 'chunk_id' in filtered_df.columns and 'chunk_position' in filtered_df.columns:
        chunk_appearances = filtered_df.groupby('chunk_id').size()
        chunk_rank1_count = filtered_df[filtered_df['chunk_position'] == 1].groupby('chunk_id').size()
        # Devil chunks: appear >= 3 times but rank 1 count < appearances * 0.3
        for chunk_id in chunk_appearances.index:
            appearances = chunk_appearances[chunk_id]
            rank1 = chunk_rank1_count.get(chunk_id, 0)
            if appearances >= 3 and rank1 < appearances * 0.3:
                devil_chunks_count += 1
    
    main_metrics = dbc.Row([
        dbc.Col(create_metric_card("MRR", f"{metrics.get('mrr', 0):.3f}", "Mean Reciprocal Rank"), width=3),
        dbc.Col(create_metric_card(f"Hit Rate@{top_k}", f"{metrics.get(hit_rate_key, 0):.3f}", 
                                   "Fraction of queries where relevant chunk was found in top-k"), width=3),
        dbc.Col(create_metric_card(f"Precision@{top_k}", f"{metrics.get(precision_key, 0):.3f}", "Precision at K"), width=3),
        dbc.Col(create_metric_card(f"Recall@{top_k}", f"{metrics.get(recall_key, 0):.3f}", "Recall at K"), width=3),
    ], className="mb-4")
    
    stats_metrics = dbc.Row([
        dbc.Col(create_metric_card("Total Queries", str(metrics.get('num_queries', 0))), width=3),
        dbc.Col(create_metric_card("Found in Top-K", str(metrics.get('num_found', 0))), width=3),
        dbc.Col(create_metric_card("Not Found", str(metrics.get('num_not_found', 0))), width=3),
        dbc.Col(create_metric_card("Word/Char Ratio", f"{word_char_ratio:.3f}", "Average word to character ratio"), width=3),
    ], className="mb-4")
    
    additional_stats = dbc.Row([
        dbc.Col(create_metric_card("Devil Chunks", str(devil_chunks_count), "Chunks appearing frequently but rarely at rank 1"), width=3),
    ], className="mb-4")
    
    return html.Div([
        html.H3("Summary Metrics", className="mb-4"),
        main_metrics,
        html.H4("Statistics", className="mb-3"),
        stats_metrics,
        additional_stats,
        html.Hr(),
        html.H4("Report Generation", className="mb-3"),
        dbc.Button("📄 Generate HTML Report", id="download-report-btn", color="primary", className="me-2"),
        html.Div(id="report-status")
    ])


def render_table_tab(filtered_df, selected_version):
    """Render the results table tab."""
    default_cols = ['chunk_id', 'question', 'answer', 'chunk_position', 'num_retrieved']
    available_cols = [col for col in filtered_df.columns if col in default_cols or col.startswith('retrieved_')]
    
    return html.Div([
        html.H3("Results Table", className="mb-4"),
        html.Div([
            html.Label("🔍 Search", className="me-2"),
            dcc.Input(
                id="table-search",
                type="text",
                placeholder="Search in questions, answers, or chunks...",
                className="form-control mb-3",
                style={"width": "100%"}
            ),
            dbc.Row([
                dbc.Col([
                    html.Label("Sort by", className="mb-2"),
                    dcc.Dropdown(
                        id="table-sort",
                        options=[
                            {"label": "None", "value": "none"},
                            {"label": "Position (Asc)", "value": "position_asc"},
                            {"label": "Position (Desc)", "value": "position_desc"},
                            {"label": "MRR Score", "value": "mrr"},
                            {"label": "Hit Rate", "value": "hit_rate"}
                        ],
                        value="none",
                        className="mb-3"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Rows to display", className="mb-2"),
                    dcc.Input(
                        id="table-rows",
                        type="number",
                        value=50,
                        min=10,
                        max=1000,
                        step=10,
                        className="form-control mb-3"
                    )
                ], width=6)
            ]),
            html.Div(id="table-display"),
            dbc.Button("📥 Download Filtered Results (CSV)", id="download-csv-btn", color="primary", className="mt-3")
        ])
    ])


def render_distribution_tab(filtered_df, top_k):
    """Render the rank distribution tab."""
    rank_dist = get_rank_distribution(filtered_df, top_k=top_k)
    
    dist_data = {
        'Rank': [rank.replace('_', ' ').title() for rank in rank_dist.keys() if rank != 'total'],
        'Count': [count for rank, count in rank_dist.items() if rank != 'total']
    }
    dist_df = pd.DataFrame(dist_data)
    
    fig = px.bar(
        dist_df,
        x='Rank',
        y='Count',
        title='Distribution of Retrieval Positions',
        labels={'Count': 'Number of Queries', 'Rank': 'Position'},
        color='Count',
        color_continuous_scale='Blues'
    )
    fig = apply_chart_theme(fig)
    
    dist_table_data = [
        {
            "Rank": rank.replace('_', ' ').title(),
            "Count": count,
            "Percentage": f"{count/rank_dist['total']*100:.1f}%"
        }
        for rank, count in rank_dist.items()
        if rank != 'total'
    ]
    dist_table_df = pd.DataFrame(dist_table_data)
    
    return html.Div([
        html.H3("Rank Distribution", className="mb-4"),
        create_chart_container(fig, "Distribution of Retrieval Positions"),
        html.H4("Distribution Table", className="mb-3"),
        dash_table.DataTable(
            data=dist_table_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in dist_table_df.columns],
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': '#667eea', 'color': 'white', 'fontWeight': '600'}
        )
    ])


def render_chunk_length_tab(filtered_df, metadata, top_k):
    """Render the chunk length analysis tab."""
    return html.Div([
        html.H3("📏 Chunk Length Analysis", className="mb-4"),
        html.P("Analyze how chunk size affects retrieval performance.", className="text-muted mb-4"),
        html.Label("Select Metric:", className="mb-2"),
        dcc.Dropdown(
            id='chunk-length-metric-dropdown',
            options=[
                {"label": "MRR", "value": "mrr"},
                {"label": "Hit Rate", "value": "hit_rate"},
                {"label": "Precision", "value": "precision"},
                {"label": "Recall", "value": "recall"}
            ],
            value="mrr",
            className="mb-3"
        ),
        dbc.Button("🔍 Analyze Chunk Length", id="analyze-chunk-length-btn", color="primary", className="mb-4"),
        html.Div(id="analyze-chunk-length-output")
    ])

def render_word_char_tab(filtered_df, metadata, top_k):
    """Render the word-char ratio analysis tab."""
    return html.Div([
        html.H3("📊 Word-Char Ratio Analysis", className="mb-4"),
        html.P("Analyze word to character ratio and its impact on retrieval.", className="text-muted mb-4"),
        dbc.Button("📊 Analyze Word-Char Ratio", id="analyze-word-char-btn", color="primary", className="mb-4"),
        html.Div(id="analyze-word-char-output")
    ])

def render_query_similarity_tab(filtered_df, metadata, top_k):
    """Render the query similarity analysis tab."""
    return html.Div([
        html.H3("🔗 Query Similarity Analysis", className="mb-4"),
        html.P("Analyze query similarity patterns and their impact on retrieval.", className="text-muted mb-4"),
        dbc.Button("🔗 Analyze Query Similarity", id="analyze-query-similarity-btn", color="primary", className="mb-4"),
        html.Div(id="analyze-query-similarity-output")
    ])

def render_devil_chunks_tab(filtered_df, metadata, top_k):
    """Render the devil chunks analysis tab."""
    return html.Div([
        html.H3("👹 Devil Chunks Analysis", className="mb-4"),
        html.P("Identify chunks that appear frequently but are not relevant.", className="text-muted mb-4"),
        dbc.Button("👹 Identify Devil Chunks", id="identify-devil-chunks-btn", color="primary", className="mb-4"),
        html.Div(id="identify-devil-chunks-output")
    ])


