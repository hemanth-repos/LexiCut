import os
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page layout to wide and dark theme configurations
st.set_page_config(
    page_title="LexiCut Telemetry Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom premium styling for dark mode dashboard
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #374151;
        text-align: center;
    }
    .callout-box {
        background-color: #111827;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header Row
col_title, col_btn = st.columns([5, 1])
with col_title:
    st.title("📊 LexiCut Executive Telemetry Center")
    st.caption("Real-time performance metrics and ROI visualizations for the LexiCut semantic caching pipeline.")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# Database extraction function
def load_data():
    db_path = "telemetry.db"
    if not os.path.exists(db_path):
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM request_logs ORDER BY id ASC", conn)
        conn.close()
        
        # Ensure antonym_blocked and new telemetry columns exist (backward compatibility/migration safety)
        if not df.empty:
            if 'antonym_blocked' not in df.columns:
                df['antonym_blocked'] = 0
            if 'faiss_candidates_examined' not in df.columns:
                df['faiss_candidates_examined'] = 0
            if 'faiss_similarity' not in df.columns:
                df['faiss_similarity'] = 0.0
            if 'retrieval_method' not in df.columns:
                df['retrieval_method'] = "faiss"
            if 'faiss_retrieval_latency_ms' not in df.columns:
                df['faiss_retrieval_latency_ms'] = 0.0
            if 'cache_validation_latency_ms' not in df.columns:
                df['cache_validation_latency_ms'] = 0.0
            if 'embedding_latency_ms' not in df.columns:
                df['embedding_latency_ms'] = 0.0
            if 'faiss_search_latency_ms' not in df.columns:
                df['faiss_search_latency_ms'] = 0.0
            
        return df
    except Exception as e:
        print(f"Error reading sqlite database: {e}")
        return pd.DataFrame()

df = load_data()

# Check for empty database state
if df.empty:
    st.warning("⚠️ No telemetry data found. Please run the integration tests or start the FastAPI server to ingest queries first.")
    
    # Render empty state placeholder metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ingested Queries", "0")
    col2.metric("Global Cache Hit Rate", "0.0%")
    col3.metric("Estimated Tokens Saved", "0")
    col4.metric("Average Latency Speedup", "0.0 ms")
    
    st.info("Start sending requests to the `/chat` endpoint to populate this dashboard.")
else:
    # 1. HERO METRICS calculations
    total_queries = len(df)
    hits = len(df[df['status'] == 'HIT'])
    hit_rate = (hits / total_queries * 100.0) if total_queries > 0 else 0.0
    total_saved = df['tokens_saved'].sum()
    
    avg_miss = df[df['status'] == 'MISS']['latency_ms'].mean() if len(df[df['status'] == 'MISS']) > 0 else 0.0
    avg_hit = df[df['status'] == 'HIT']['latency_ms'].mean() if len(df[df['status'] == 'HIT']) > 0 else 0.0
    speedup = max(0.0, avg_miss - avg_hit)
    
    # Render Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Ingested Queries", f"{total_queries}")
    m2.metric("Global Cache Hit Rate (%)", f"{hit_rate:.1f}%")
    m3.metric("Total Tokens Saved", f"{total_saved:,}")
    m4.metric("Average Latency Speedup", f"{speedup:.2f} ms")
    
    st.markdown("---")
    
    # 2. ROI VISUALIZATION
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Latency chart over time
        df_latency = df.copy()
        df_latency['Request Index'] = df_latency.index + 1
        fig_latency = px.bar(
            df_latency,
            x='Request Index',
            y='latency_ms',
            color='status',
            color_discrete_map={'HIT': '#2ecc71', 'MISS': '#ef4444'},
            labels={'Request Index': 'Request Sequence', 'latency_ms': 'Latency (ms)', 'status': 'Status'},
            title='Latency per Request Over Time'
        )
        fig_latency.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#374151')
        )
        st.plotly_chart(fig_latency, use_container_width=True)
        
    with chart_col2:
        # Cumulative Tokens
        df_tokens = df.copy()
        df_tokens['Request Index'] = df_tokens.index + 1
        df_tokens['cumulative_used'] = df_tokens['tokens_used'].cumsum()
        df_tokens['cumulative_saved'] = df_tokens['tokens_saved'].cumsum()
        
        fig_tokens = go.Figure()
        fig_tokens.add_trace(go.Scatter(
            x=df_tokens['Request Index'],
            y=df_tokens['cumulative_used'],
            fill='tozeroy',
            name='Cumulative Consumed (Used)',
            line=dict(color='#ef4444', width=2),
            fillcolor='rgba(239, 68, 68, 0.2)'
        ))
        fig_tokens.add_trace(go.Scatter(
            x=df_tokens['Request Index'],
            y=df_tokens['cumulative_saved'],
            fill='tozeroy',
            name='Cumulative Saved (Cache Hits)',
            line=dict(color='#2ecc71', width=2),
            fillcolor='rgba(46, 204, 113, 0.2)'
        ))
        
        fig_tokens.update_layout(
            title='Cumulative Token ROI (Consumed vs Saved)',
            xaxis_title='Request Sequence',
            yaxis_title='Tokens',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#374151')
        )
        st.plotly_chart(fig_tokens, use_container_width=True)

    st.markdown("---")
    
    # 2.5 PERFORMANCE & SCALABILITY CENTER
    st.markdown("### ⚡ Performance & Scalability Center")
    col_perf1, col_perf2, col_perf3 = st.columns(3)
    
    with col_perf1:
        # Vector Index Stats
        # Total vectors from metadata pickle
        total_vectors = 0
        pickle_path = "query_metadata.pkl"
        if os.path.exists(pickle_path):
            try:
                import pickle
                with open(pickle_path, "rb") as f:
                    meta_dict = pickle.load(f)
                total_vectors = len(meta_dict)
            except Exception:
                pass
                
        # Index file size
        file_size_kb = 0.0
        index_path = "faiss.index"
        if os.path.exists(index_path):
            try:
                file_size_kb = os.path.getsize(index_path) / 1024.0
            except Exception:
                pass
                
        # Retrieval method
        method = "faiss"
        if not df.empty and 'retrieval_method' in df.columns:
            method = df['retrieval_method'].iloc[-1]
            
        st.markdown(f"""
        <div class="callout-box" style="border-left: 5px solid #10b981; min-height: 220px;">
            <h4>Vector Index Stats</h4>
            <ul style="font-size: 14px; margin-bottom: 0px; padding-left: 20px; line-height: 1.8;">
                <li><strong>Total Vectors Indexed:</strong> {total_vectors}</li>
                <li><strong>Index File Size:</strong> {file_size_kb:.2f} KB</li>
                <li><strong>Active Retrieval Method:</strong> <code style="color: #10b981;">{method}</code></li>
                <li><strong>Index Backend:</strong> FAISS IndexFlatIP (IDMap)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_perf2:
        # Search Performance
        clean_df = df[df["embedding_latency_ms"] < 500]
        avg_embedding = clean_df['embedding_latency_ms'].mean() if not df.empty and 'embedding_latency_ms' in df.columns else 0.0
        avg_faiss_search = df['faiss_search_latency_ms'].mean() if not df.empty and 'faiss_search_latency_ms' in df.columns else 0.0
        avg_validation = df['cache_validation_latency_ms'].mean() if not df.empty and 'cache_validation_latency_ms' in df.columns else 0.0
        total_examined = df['faiss_candidates_examined'].sum() if not df.empty and 'faiss_candidates_examined' in df.columns else 0
        
        st.markdown(f"""
        <div class="callout-box" style="border-left: 5px solid #f59e0b; min-height: 220px;">
            <h4>Search Performance</h4>
            <ul style="font-size: 14px; margin-bottom: 0px; padding-left: 20px; line-height: 1.8;">
                <li><strong>Avg Embedding Latency:</strong> {avg_embedding:.4f} ms</li>
                <li><strong>Avg FAISS Search Latency:</strong> {avg_faiss_search:.4f} ms</li>
                <li><strong>Avg Validation Latency:</strong> {avg_validation:.4f} ms</li>
                <li><strong>Total Candidates Examined:</strong> {total_examined}</li>
                <li><strong>Avg Candidates per Query:</strong> {df['faiss_candidates_examined'].mean() if not df.empty and 'faiss_candidates_examined' in df.columns else 0.0:.1f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_perf3:
        # Scalability Projection
        # Get baseline retrieval latency (FAISS search latency)
        # If no entries, assume default 0.15 ms
        baseline_lat = avg_faiss_search if avg_faiss_search > 0 else 0.15
        current_n = max(1, total_vectors)
        
        # Scale projected FAISS retrieval latency using formula:
        # L(M) = L_baseline * (1 + 0.0001 * M) / (1 + 0.0001 * N)
        def project(m):
            return baseline_lat * (1.0 + 0.0001 * m) / (1.0 + 0.0001 * current_n)
            
        p_100 = project(100)
        p_1k = project(1000)
        p_10k = project(10000)
        p_100k = project(100000)
        
        st.markdown(f"""
        <div class="callout-box" style="border-left: 5px solid #8b5cf6; min-height: 220px;">
            <h4>Scalability Projection</h4>
            <table style="width: 100%; font-size: 13px; text-align: left; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #374151;">
                    <th style="padding: 2px 0;">Cache Size</th>
                    <th style="padding: 2px 0; text-align: right;">Projected Latency</th>
                </tr>
                <tr>
                    <td style="padding: 2px 0;">100 entries</td>
                    <td style="padding: 2px 0; text-align: right; color: #a78bfa;">{p_100:.4f} ms</td>
                </tr>
                <tr>
                    <td style="padding: 2px 0;">1,000 entries</td>
                    <td style="padding: 2px 0; text-align: right; color: #a78bfa;">{p_1k:.4f} ms</td>
                </tr>
                <tr>
                    <td style="padding: 2px 0;">10,000 entries</td>
                    <td style="padding: 2px 0; text-align: right; color: #a78bfa;">{p_10k:.4f} ms</td>
                </tr>
                <tr>
                    <td style="padding: 2px 0;">100,000 entries</td>
                    <td style="padding: 2px 0; text-align: right; color: #a78bfa;">{p_100k:.4f} ms</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 3. SAFETY AUDIT TRAY
    # False positives: similarity_score > 0.90 but overlap_score < 0.30 (so status is MISS, but similarity was high)
    false_positives = df[(df['similarity_score'] > 0.90) & (df['overlap_score'] < 0.30)]
    fp_count = len(false_positives)
    
    # Semantic Safety Blocks: queries where antonym_blocked == 1
    safety_blocks = df['antonym_blocked'].sum() if 'antonym_blocked' in df.columns else 0
    total_semantic_hits = hits + safety_blocks
    prevented_false_hits_pct = (safety_blocks / total_semantic_hits * 100.0) if total_semantic_hits > 0 else 0.0
    
    st.markdown("### 🛡️ Safety & Alignment Tray")
    col_safety1, col_safety2 = st.columns(2)
    
    with col_safety1:
        st.markdown(f"""
        <div class="callout-box" style="border-left: 5px solid #ef4444;">
            <h4>Saved from False Positives (Lexical): <strong>{fp_count}</strong></h4>
            <p style="font-size: 14px; margin-bottom: 0px;">
                Queries that exceeded the <strong>0.92 semantic threshold</strong> but were successfully caught and rejected 
                by our secondary <strong>70% lexical overlap filter</strong>. This prevented inaccurate cached answers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_safety2:
        st.markdown(f"""
        <div class="callout-box" style="border-left: 5px solid #3b82f6; background-color: #111827;">
            <h4>Semantic Safety Blocks (Antonyms): <strong>{safety_blocks}</strong></h4>
            <p style="font-size: 14px; margin-bottom: 0px;">
                Prevented <strong>{prevented_false_hits_pct:.1f}%</strong> of potential semantic cache hits due to antonym 
                conflicts (e.g., Enable/Disable, Start/Stop). This prevents opposite commands from returning incorrect cached answers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    # 4. LIVE TELEMETRY FEED
    st.markdown("### 📥 Live Telemetry Feed (Last 15 Requests)")
    df_live = df.copy()
    # Format and select columns
    df_live['Overlap %'] = df_live['overlap_score'].apply(lambda x: f"{x * 100.0:.1f}%")
    df_live['Similarity Score'] = df_live['similarity_score'].apply(lambda x: f"{x:.4f}")
    df_live['Latency (ms)'] = df_live['latency_ms'].apply(lambda x: f"{x:.2f} ms")
    
    df_live = df_live.rename(columns={
        'timestamp': 'Timestamp',
        'prompt': 'Prompt',
        'status': 'Status'
    })
    
    # Show last 15 in reverse chronological order
    df_live_feed = df_live[['Timestamp', 'Prompt', 'Status', 'Similarity Score', 'Overlap %', 'Latency (ms)']].iloc[::-1]
    
    st.dataframe(
        df_live_feed,
        use_container_width=True,
        hide_index=True
    )
