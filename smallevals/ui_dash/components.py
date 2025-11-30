"""Reusable UI components for the Dash dashboard."""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_metric_card(label: str, value: str, help_text: str = None, color: str = "#667eea"):
    """Create a professional metric card with gradient styling."""
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                label,
                className="metric-label",
                style={
                    "fontSize": "0.9rem",
                    "color": "#6b7280",
                    "fontWeight": "500",
                    "marginBottom": "0.5rem"
                }
            ),
            html.Div(
                value,
                className="metric-value",
                style={
                    "fontSize": "2.5rem",
                    "fontWeight": "700",
                    "color": color,
                    "margin": "0"
                }
            ),
            html.Div(
                help_text or "",
                className="metric-help",
                style={
                    "fontSize": "0.75rem",
                    "color": "#9ca3af",
                    "marginTop": "0.25rem"
                }
            ) if help_text else None
        ]),
        className="metric-card",
        style={
            "background": "white",
            "padding": "1.5rem",
            "borderRadius": "12px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "textAlign": "center",
            "transition": "transform 0.2s",
            "height": "100%"
        }
    )


def create_version_info_card(metadata: dict):
    """Create a card displaying version information."""
    if not metadata:
        return dbc.Card(
            dbc.CardBody([
                html.P("No metadata available", className="text-muted")
            ]),
            className="mb-3"
        )
    
    return dbc.Card(
        dbc.CardBody([
            html.H5("Version Information", className="mb-3"),
            html.P([
                html.Strong("Model: "),
                metadata.get('embedding_model', 'N/A')
            ], className="mb-2"),
            html.P([
                html.Strong("Created: "),
                metadata.get('created_at', 'N/A')
            ], className="mb-2"),
            html.P([
                html.Strong("Top-K: "),
                str(metadata.get('top_k', 5))
            ], className="mb-2"),
            html.P([
                html.Strong("Description: "),
                metadata.get('description', 'N/A')
            ], className="mb-0") if metadata.get('description') else None
        ]),
        className="mb-3",
        style={
            "background": "white",
            "borderRadius": "12px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
        }
    )


def create_header():
    """Create the professional header with gradient background."""
    return html.Div([
        html.Div([
            html.H1(
                "📊 smallevals",
                style={
                    "margin": "0",
                    "fontSize": "2.5rem",
                    "fontWeight": "700",
                    "color": "white"
                }
            ),
            html.P(
                "Retrieval Evaluation Dashboard",
                style={
                    "margin": "0.5rem 0 0 0",
                    "opacity": "0.9",
                    "fontSize": "1.1rem",
                    "color": "white"
                }
            )
        ], style={"padding": "2rem"})
    ], style={
        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "color": "white",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
        "marginBottom": "2rem"
    })


def create_loading_spinner():
    """Create a loading spinner component."""
    return dbc.Spinner(
        html.Div(id="loading-output"),
        color="primary",
        size="lg"
    )


def create_analysis_section(title: str, button_id: str, button_text: str, description: str = None):
    """Create a section for analysis features."""
    return html.Div([
        html.H4(title, className="mb-3"),
        html.P(description, className="text-muted mb-3") if description else None,
        dbc.Button(
            button_text,
            id=button_id,
            color="primary",
            className="mb-3",
            style={
                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                "border": "none",
                "borderRadius": "8px",
                "padding": "0.75rem 1.5rem",
                "fontWeight": "500"
            }
        ),
        html.Div(id=f"{button_id}-output")
    ], className="mb-4")


def create_chart_container(figure, title: str = None):
    """Create a container for Plotly charts with consistent styling."""
    return html.Div([
        html.H5(title, className="mb-3") if title else None,
        dcc.Graph(
            figure=figure,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]
            },
            style={"height": "500px"}
        )
    ], className="mb-4", style={
        "background": "white",
        "padding": "1.5rem",
        "borderRadius": "12px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
    })


def create_alert(message: str, alert_type: str = "success"):
    """Create an alert component."""
    color_map = {
        "success": "success",
        "error": "danger",
        "warning": "warning",
        "info": "info"
    }
    return dbc.Alert(
        message,
        color=color_map.get(alert_type, "info"),
        dismissable=True,
        className="mt-3"
    )


def apply_chart_theme(figure):
    """Apply consistent theming to Plotly figures."""
    figure.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12, color="#1f2937"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        colorway=["#667eea", "#764ba2", "#f5576c", "#4ecdc4", "#f59e0b"],
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode="closest"
    )
    return figure

