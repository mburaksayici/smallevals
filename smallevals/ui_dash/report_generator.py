"""HTML report generation for evaluation results."""

import pandas as pd
from typing import Dict, Any
from datetime import datetime
from typing import *
import base64
import os


def get_logo_base64():
    """Get the logo as a base64 encoded data URI."""
    try:
        # Get the path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        logo_path = os.path.join(project_root, 'logo', 'smallevals_emoji_128_128.png')
        
        if os.path.exists(logo_path):
            with open(logo_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode()
                return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error loading logo: {e}")
    return None


def generate_html_report(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    version_metadata: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> str:
    """
    Generate an HTML report from evaluation results.
    
    Args:
        df: Results dataframe
        metrics: Dictionary of calculated metrics
        version_metadata: Optional metadata about the evaluation version
        top_k: Value of K used for metrics
        
    Returns:
        HTML string
    """
    version_metadata = version_metadata or {}
    logo_base64 = get_logo_base64()
    
    # Generate summary statistics
    total_queries = metrics.get('num_queries', len(df))
    
    hit_rate_key = f'hit_rate@{top_k}'
    precision_key = f'precision@{top_k}'
    recall_key = f'recall@{top_k}'
    ndcg_key = f'ndcg@{top_k}'
    
    # Create rank distribution and calculate statistics from DataFrame
    rank_dist = {}
    num_found_in_topk = 0
    num_found_in_top1 = 0
    num_not_found = 0
    
    if 'chunk_position' in df.columns:
        for rank in range(1, top_k + 1):
            count = len(df[df['chunk_position'] == rank])
            rank_dist[rank] = count
            num_found_in_topk += count
        
        # Count found in top-1 (rank 1)
        num_found_in_top1 = len(df[df['chunk_position'] == 1])
        
        # Count not found: NaN or position > top_k
        not_found_mask = df['chunk_position'].isna() | (df['chunk_position'] > top_k)
        num_not_found = len(df[not_found_mask])
        rank_dist['not_found'] = num_not_found
    else:
        # If no chunk_position column, assume all not found
        num_not_found = len(df)
        rank_dist['not_found'] = num_not_found
    
    # Fallback to metrics if DataFrame doesn't have chunk_position
    if num_found_in_topk == 0 and num_not_found == 0:
        num_found_in_topk = metrics.get('num_found', 0)
        num_not_found = metrics.get('num_not_found', 0)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>smallevals Retrieval Evaluation Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            --dark-bg: #0f172a;
            --card-bg: rgba(255, 255, 255, 0.95);
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: rgba(148, 163, 184, 0.2);
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            min-height: 100vh;
            padding: 40px 20px;
            position: relative;
            overflow-x: hidden;
        }}
        
        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(118, 75, 162, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 20%, rgba(79, 172, 254, 0.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            padding: 50px;
            border-radius: 24px;
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.3),
                0 0 0 1px rgba(255, 255, 255, 0.1);
            position: relative;
            z-index: 1;
            animation: fadeInUp 0.6s ease-out;
        }}
        
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 50px;
            padding-bottom: 30px;
            border-bottom: 2px solid var(--border-color);
            position: relative;
        }}
        
        .header-logo {{
            width: 64px;
            height: 64px;
            margin: 0 auto 20px;
            display: block;
        }}
        
        .header::after {{
            content: '';
            position: absolute;
            bottom: -2px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 2px;
            background: var(--primary-gradient);
            border-radius: 2px;
        }}
        
        h1 {{
            font-size: 3.5rem;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 15px;
            letter-spacing: -0.02em;
        }}
        
        .subtitle {{
            font-size: 1.1rem;
            color: var(--text-secondary);
            font-weight: 400;
        }}
        
        h2 {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-top: 50px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        h2::after {{
            content: '';
            flex: 1;
            height: 2px;
            background: linear-gradient(to right, var(--primary-gradient), transparent);
            border-radius: 2px;
        }}
        
        .metadata {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            backdrop-filter: blur(10px);
            padding: 25px 30px;
            border-radius: 16px;
            margin-bottom: 40px;
            border: 1px solid var(--border-color);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        
        .metadata-item {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .metadata-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .metadata-value {{
            font-size: 1.1rem;
            color: var(--text-primary);
            font-weight: 600;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}
        
        .metric-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--primary-gradient);
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-8px);
            box-shadow: 
                0 20px 40px rgba(102, 126, 234, 0.2),
                0 0 0 1px rgba(102, 126, 234, 0.1);
        }}
        
        .metric-card:hover::before {{
            transform: scaleX(1);
        }}
        
        .metric-card.mrr {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        }}
        
        .metric-card.hit-rate {{
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%);
        }}
        
        .metric-card.precision {{
            background: linear-gradient(135deg, rgba(250, 112, 154, 0.1) 0%, rgba(254, 225, 64, 0.1) 100%);
        }}
        
        .metric-card.recall {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        }}
        
        .metric-card h3 {{
            font-size: 0.95rem;
            color: var(--text-secondary);
            font-weight: 600;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .metric-card .value {{
            font-size: 3rem;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
            margin-bottom: 10px;
        }}
        
        .metric-card.mrr .value {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .metric-card.hit-rate .value {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .metric-card.precision .value {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .metric-card.recall .value {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .statistics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-item {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 25px;
            border-radius: 16px;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .stat-item::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .stat-item:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.15);
        }}
        
        .stat-item:hover::before {{
            opacity: 1;
        }}
        
        .stat-item .label {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 12px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            position: relative;
            z-index: 1;
        }}
        
        .stat-item .value {{
            font-size: 2.5rem;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            position: relative;
            z-index: 1;
        }}
        
        .rank-distribution {{
            margin-top: 30px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        thead {{
            background: var(--primary-gradient);
        }}
        
        th {{
            padding: 20px;
            text-align: left;
            color: white;
            font-weight: 600;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        th:first-child {{
            border-top-left-radius: 20px;
        }}
        
        th:last-child {{
            border-top-right-radius: 20px;
        }}
        
        tbody tr {{
            transition: all 0.2s ease;
            border-bottom: 1px solid var(--border-color);
        }}
        
        tbody tr:last-child {{
            border-bottom: none;
        }}
        
        tbody tr:hover {{
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.05) 0%, transparent 100%);
            transform: scale(1.01);
        }}
        
        td {{
            padding: 18px 20px;
            color: var(--text-primary);
            font-weight: 500;
        }}
        
        .rank-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            color: #667eea;
        }}
        
        .rank-badge.rank-1 {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
            color: #10b981;
        }}
        
        .rank-badge.not-found {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
            color: #ef4444;
        }}
        
        .percentage-bar {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .bar-container {{
            flex: 1;
            height: 8px;
            background: rgba(148, 163, 184, 0.2);
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .bar-fill {{
            height: 100%;
            background: var(--primary-gradient);
            border-radius: 10px;
            transition: width 1s ease-out;
            animation: slideIn 1s ease-out;
        }}
        
        @keyframes slideIn {{
            from {{
                width: 0;
            }}
        }}
        
        .bar-fill.rank-1 {{
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        }}
        
        .bar-fill.not-found {{
            background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
        }}
        
        .footer {{
            margin-top: 60px;
            padding-top: 30px;
            border-top: 2px solid var(--border-color);
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.95rem;
            position: relative;
        }}
        
        .footer::before {{
            content: '';
            position: absolute;
            top: -2px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 2px;
            background: var(--primary-gradient);
            border-radius: 2px;
        }}
        
        .footer-brand {{
            font-weight: 700;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 30px 20px;
            }}
            
            h1 {{
                font-size: 2.5rem;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            
            .statistics {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            {f'<img src="{logo_base64}" alt="smallevals logo" class="header-logo">' if logo_base64 else ''}
            <h1>smallevals Retrieval Evaluation Report</h1>
            <p class="subtitle">Comprehensive Analysis of Vector Database Performance</p>
        </div>
        
        <div class="metadata">
            <div class="metadata-item">
                <span class="metadata-label">Generated</span>
                <span class="metadata-value">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
            {f'<div class="metadata-item"><span class="metadata-label">Version</span><span class="metadata-value">{version_metadata.get("selected_version", "Unknown")}</span></div>' if version_metadata.get('selected_version') else ''}
            {f'<div class="metadata-item"><span class="metadata-label">Top-K</span><span class="metadata-value">{top_k}</span></div>' if top_k else ''}
            {f'<div class="metadata-item"><span class="metadata-label">Vector DB</span><span class="metadata-value">{version_metadata.get("vector_db", "Unknown")}</span></div>' if version_metadata.get('vector_db') else ''}
            {f'<div class="metadata-item"><span class="metadata-label">Embedding Model</span><span class="metadata-value">{version_metadata.get("embedding_model", "Unknown")}</span></div>' if version_metadata.get('embedding_model') else ''}
            {f'<div class="metadata-item"><span class="metadata-label">Description</span><span class="metadata-value">{version_metadata.get("description", "")}</span></div>' if version_metadata.get('description') else ''}
        </div>
        
        <h2>Key Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card mrr">
                <h3>Mean Reciprocal Rank</h3>
                <div class="value">{metrics.get('mrr', 0):.4f}</div>
            </div>
            <div class="metric-card hit-rate">
                <h3>Hit Rate@{top_k}</h3>
                <div class="value">{metrics.get(hit_rate_key, 0):.4f}</div>
            </div>
            <div class="metric-card precision">
                <h3>nDCG@{top_k}</h3>
                <div class="value">{metrics.get(ndcg_key, 0):.4f}</div>
            </div>
            <div class="metric-card recall">
                <h3>Recall@{top_k}</h3>
                <div class="value">{metrics.get(recall_key, 0):.4f}</div>
            </div>
        </div>
        
        <h2>Top-1 Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card hit-rate">
                <h3>Hit Rate@1</h3>
                <div class="value">{metrics.get('hit_rate@1', 0):.4f}</div>
            </div>
            <div class="metric-card precision">
                <h3>nDCG@1</h3>
                <div class="value">{metrics.get('ndcg@1', 0):.4f}</div>
            </div>
            <div class="metric-card recall">
                <h3>Recall@1</h3>
                <div class="value">{metrics.get('recall@1', 0):.4f}</div>
            </div>
        </div>
        
        <h2>Statistics</h2>
        <div class="statistics">
            <div class="stat-item">
                <div class="label">Total Queries</div>
                <div class="value">{total_queries}</div>
            </div>
            <div class="stat-item">
                <div class="label">Found in Top-1</div>
                <div class="value">{num_found_in_top1}</div>
            </div>
            <div class="stat-item">
                <div class="label">Found in Top-{top_k}</div>
                <div class="value">{num_found_in_topk}</div>
            </div>
            <div class="stat-item">
                <div class="label">Not Found</div>
                <div class="value">{num_not_found}</div>
            </div>
        </div>
        
        <h2>Rank Distribution</h2>
        <div class="rank-distribution">
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add rank distribution rows
    if rank_dist:
        for rank in range(1, top_k + 1):
            count = rank_dist.get(rank, 0)
            percentage = (count / total_queries * 100) if total_queries > 0 else 0
            rank_class = 'rank-1' if rank == 1 else ''
            html_content += f"""
                    <tr>
                        <td><span class="rank-badge {rank_class}">Rank {rank}</span></td>
                        <td><strong>{count}</strong></td>
                        <td>
                            <div class="percentage-bar">
                                <div class="bar-container">
                                    <div class="bar-fill {rank_class}" style="width: {percentage}%"></div>
                                </div>
                                <span style="min-width: 50px; text-align: right; font-weight: 600;">{percentage:.1f}%</span>
                            </div>
                        </td>
                    </tr>
"""
        not_found_count = rank_dist.get('not_found', 0)
        not_found_pct = (not_found_count / total_queries * 100) if total_queries > 0 else 0
        html_content += f"""
                    <tr>
                        <td><span class="rank-badge not-found">Not Found</span></td>
                        <td><strong>{not_found_count}</strong></td>
                        <td>
                            <div class="percentage-bar">
                                <div class="bar-container">
                                    <div class="bar-fill not-found" style="width: {not_found_pct}%"></div>
                                </div>
                                <span style="min-width: 50px; text-align: right; font-weight: 600;">{not_found_pct:.1f}%</span>
                            </div>
                        </td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Generated by <span class="footer-brand">smallevals</span> - Retrieval Evaluation Framework</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html_content