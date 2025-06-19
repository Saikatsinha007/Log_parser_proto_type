
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
import pandas as pd
import re
import json
import time
from datetime import datetime
from collections import defaultdict
import plotly.express as px
import umap.umap_ as umap
from sklearn.cluster import KMeans
import hashlib

# -----------------------------
# Setup
# -----------------------------
st.set_page_config(page_title="Advanced Log Vector Search", layout="wide", page_icon="🔍")
st.title("🚀 Advanced Log Vector Search Engine")

# Custom CSS for better styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1DA1F2;
    }
    .stTextArea textarea {
        min-height: 150px;
    }
    .log-card {
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .error-log {
        border-left: 4px solid #FF4B4B;
        background-color: #FFF5F5;
    }
    .info-log {
        border-left: 4px solid #1DA1F2;
        background-color: #F5FAFF;
    }
    .warning-log {
        border-left: 4px solid #F0B429;
        background-color: #FFF9E6;
    }
    .match-score {
        font-size: 0.8em;
        color: #666;
        float: right;
    }
    .log-type-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Components with Progress
# -----------------------------
@st.cache_resource(show_spinner="Loading AI components...")
def load_components():
    with st.spinner("Initializing template miner..."):
        persistence = FilePersistence("drain3_state.json")
        template_miner = TemplateMiner(persistence)
    
    with st.spinner("Loading sentence transformer model..."):
        model = SentenceTransformer("all-mpnet-base-v2")  # More powerful than MiniLM
    
    with st.spinner("Preparing FAISS index..."):
        index = faiss.IndexFlatIP(768)  # Using Inner Product for similarity
        index = faiss.IndexIDMap(index)
    
    return template_miner, model, index

template_miner, model, index = load_components()

# -----------------------------
# Data Structures
# -----------------------------
if 'log_templates' not in st.session_state:
    st.session_state.log_templates = []
if 'metadata_store' not in st.session_state:
    st.session_state.metadata_store = []
if 'template_to_raw_logs' not in st.session_state:
    st.session_state.template_to_raw_logs = defaultdict(list)
if 'log_vectors' not in st.session_state:
    st.session_state.log_vectors = None
if 'log_stats' not in st.session_state:
    st.session_state.log_stats = {
        'total_logs': 0,
        'unique_templates': 0,
        'log_types': defaultdict(int),
        'timestamps': []
    }

# -----------------------------
# Sidebar - Upload & Query
# -----------------------------
st.sidebar.header("📂 Data Management")
uploaded_file = st.sidebar.file_uploader(
    "Upload Log Files", 
    type=["txt", "log", "json", "csv"],
    accept_multiple_files=True
)

st.sidebar.header("🔍 Search Options")
query_log = st.sidebar.text_area(
    "Enter Log Query", 
    placeholder="Paste a log message to find similar ones..."
)

search_options = st.sidebar.expander("Advanced Search Settings", expanded=False)
with search_options:
    k_neighbors = st.slider("Number of results", 1, 20, 5)
    similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5)
    search_space = st.radio("Search in", ["All logs", "Errors only", "Warnings only", "Custom filter"])
    if search_space == "Custom filter":
        custom_filter = st.text_input("Filter regex pattern")

# -----------------------------
# Log Processing Utilities
# -----------------------------
def detect_log_type(log):
    log = log.strip()
    if log.startswith("{"):
        try:
            json.loads(log)
            return "JSON"
        except:
            pass
    
    log = log.upper()
    if "ERROR" in log:
        return "ERROR"
    elif "WARN" in log or "WARNING" in log:
        return "WARNING"
    elif "INFO" in log:
        return "INFO"
    elif "DEBUG" in log:
        return "DEBUG"
    elif "EXCEPTION" in log or "TRACEBACK" in log:
        return "EXCEPTION"
    elif "HTTP" in log or "GET" in log or "POST" in log or "PUT" in log or "DELETE" in log:
        return "HTTP"
    elif re.match(r"^\d{4}-\d{2}-\d{2}", log):
        return "TIMESTAMPED"
    elif re.match(r"^\[.*\] (INFO|DEBUG|ERROR|WARN)", log):
        return "BRACKETED"
    else:
        return "OTHER"

def extract_timestamp(log):
    # Common timestamp patterns
    patterns = [
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
        r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',
        r'\[(\d{2}-\w{3}-\d{4} \d{2}:\d{2}:\d{2})\]',
        r'\w{3} \d{2}, \d{4} \d{2}:\d{2}:\d{2}'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, log)
        if match:
            try:
                return datetime.strptime(match.group(), "%Y-%m-%d %H:%M:%S")
            except:
                try:
                    return datetime.strptime(match.group(), "%d/%m/%Y %H:%M:%S")
                except:
                    continue
    return None

def generate_log_id(log):
    return hashlib.md5(log.encode()).hexdigest()

# -----------------------------
# Upload Processing
# -----------------------------
if uploaded_file and len(uploaded_file) > 0:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_file)
    processed_logs = 0
    
    for file_idx, file in enumerate(uploaded_file):
        try:
            file_type = file.name.split('.')[-1].lower()
            if file_type in ['txt', 'log']:
                raw_logs = file.read().decode("utf-8").splitlines()
            elif file_type == 'json':
                raw_logs = json.loads(file.read().decode("utf-8"))
                if isinstance(raw_logs, dict):
                    raw_logs = [json.dumps(raw_logs)]
            elif file_type == 'csv':
                df = pd.read_csv(file)
                raw_logs = df.to_dict(orient='records')
            else:
                raw_logs = [file.read().decode("utf-8")]
            
            total_logs = len(raw_logs)
            batch_size = min(1000, max(100, total_logs // 10))
            
            for i in range(0, total_logs, batch_size):
                batch = raw_logs[i:i+batch_size]
                batch_templates = []
                batch_vectors = []
                batch_metadata = []
                
                for line in batch:
                    if not line or not str(line).strip():
                        continue
                    
                    line = str(line).strip()
                    result = template_miner.add_log_message(line)
                    template = result["template_mined"] if result else line
                    log_type = detect_log_type(line)
                    timestamp = extract_timestamp(line)
                    log_id = generate_log_id(line)
                    
                    metadata = {
                        "id": log_id,
                        "template": template,
                        "original": line,
                        "type": log_type,
                        "timestamp": timestamp,
                        "source_file": file.name
                    }
                    
                    batch_metadata.append(metadata)
                    batch_templates.append(template)
                    st.session_state.template_to_raw_logs[template].append(line)
                    
                    # Update stats
                    st.session_state.log_stats['total_logs'] += 1
                    st.session_state.log_stats['log_types'][log_type] += 1
                    if timestamp:
                        st.session_state.log_stats['timestamps'].append(timestamp)
                
                # Encode batch
                if batch_templates:
                    batch_vectors = model.encode(batch_templates, convert_to_numpy=True)
                    
                    # Add to FAISS index
                    ids = np.array([hash(template) % (2**31) for template in batch_templates])
                    index.add_with_ids(batch_vectors, ids)
                    
                    # Store data
                    st.session_state.log_templates.extend(batch_templates)
                    st.session_state.metadata_store.extend(batch_metadata)
                    if st.session_state.log_vectors is None:
                        st.session_state.log_vectors = batch_vectors
                    else:
                        st.session_state.log_vectors = np.vstack([st.session_state.log_vectors, batch_vectors])
                
                processed_logs += len(batch)
                progress = processed_logs / (total_files * total_logs)
                progress_bar.progress(min(1.0, progress))
                status_text.text(f"Processing {file.name}: {processed_logs}/{total_files * total_logs} logs")
        
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    st.session_state.log_stats['unique_templates'] = len(st.session_state.log_templates)
    st.success(f"Processed {st.session_state.log_stats['total_logs']} logs with {st.session_state.log_stats['unique_templates']} unique templates")
    progress_bar.empty()
    status_text.empty()

# -----------------------------
# Search Query Execution
# -----------------------------
if query_log:
    with st.spinner("Searching similar logs..."):
        start_time = time.time()
        
        # Encode query
        query_vector = model.encode([query_log])
        
        # Prepare filter if needed
        if search_space == "Errors only":
            valid_ids = [hash(t['template']) % (2**31) for t in st.session_state.metadata_store if t['type'] == "ERROR"]
        elif search_space == "Warnings only":
            valid_ids = [hash(t['template']) % (2**31) for t in st.session_state.metadata_store if t['type'] == "WARNING"]
        elif search_space == "Custom filter" and 'custom_filter' in locals():
            valid_ids = [hash(t['template']) % (2**31) for t in st.session_state.metadata_store if re.search(custom_filter, t['original'])]
        else:
            valid_ids = None
        
        # Search with optional filtering
        if valid_ids:
            D, I = [], []
            for vec, idx in zip(st.session_state.log_vectors, [hash(t['template']) % (2**31) for t in st.session_state.metadata_store]):
                if idx in valid_ids:
                    sim = np.dot(query_vector, vec.T)[0][0]
                    if sim >= similarity_threshold:
                        D.append(sim)
                        I.append(idx)
            D, I = np.array(D), np.array(I)
            top_k = min(k_neighbors, len(D))
            if top_k > 0:
                top_indices = np.argpartition(D, -top_k)[-top_k:]
                D, I = D[top_indices], I[top_indices]
        else:
            D, I = index.search(query_vector, k=k_neighbors)
            D, I = D[0], I[0]
        
        # Get results
        results = []
        for score, idx in zip(D, I):
            for meta in st.session_state.metadata_store:
                if hash(meta['template']) % (2**31) == idx:
                    results.append({
                        'score': float(score),
                        'metadata': meta
                    })
                    break
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        st.subheader(f"🔍 Top {len(results)} Similar Logs ({(time.time() - start_time):.2f}s)")
        
        if not results:
            st.warning("No matching logs found above the similarity threshold.")
        else:
            for i, result in enumerate(results[:k_neighbors]):
                meta = result['metadata']
                score = result['score']
                
                # Determine card class based on log type
                card_class = ""
                if meta['type'] == "ERROR" or meta['type'] == "EXCEPTION":
                    card_class = "error-log"
                elif meta['type'] == "WARNING":
                    card_class = "warning-log"
                else:
                    card_class = "info-log"
                
                # Display result card
                st.markdown(
                    f"""
                    <div class="log-card {card_class}">
                        <span class="log-type-tag" style="background-color: {'#FF4B4B' if meta['type'] == 'ERROR' else '#F0B429' if meta['type'] == 'WARNING' else '#1DA1F2'}; color: white;">
                            {meta['type']}
                        </span>
                        <span class="match-score">Similarity: {score:.3f}</span>
                        <div style="margin-top: 8px;">{meta['original']}</div>
                        <div style="font-size: 0.8em; color: #666; margin-top: 8px;">
                            Source: {meta['source_file']} • Template: {meta['template'][:100] + ('...' if len(meta['template']) > 100 else '')}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

# -----------------------------
# Visualization Dashboard
# -----------------------------
if st.session_state.metadata_store:
    st.sidebar.header("📊 Visualization Options")
    viz_option = st.sidebar.selectbox(
        "Choose Visualization", 
        ["None", "Log Type Distribution", "Temporal Analysis", "Log Vector Clustering"]
    )
    
    if viz_option != "None":
        st.subheader("📊 Log Analytics Dashboard")
        
        if viz_option == "Log Type Distribution":
            type_counts = pd.DataFrame.from_dict(st.session_state.log_stats['log_types'], orient='index', columns=['count'])
            type_counts = type_counts.sort_values('count', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Log Type Distribution")
                fig = px.pie(type_counts, values='count', names=type_counts.index)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Top Log Types")
                fig = px.bar(type_counts.head(10), orientation='h')
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Temporal Analysis" and st.session_state.log_stats['timestamps']:
            timestamps = pd.to_datetime(pd.Series(st.session_state.log_stats['timestamps']))
            time_df = timestamps.value_counts().resample('H').count().reset_index()
            time_df.columns = ['timestamp', 'count']
            
            st.markdown("### Log Frequency Over Time")
            fig = px.line(time_df, x='timestamp', y='count')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Heatmap by Hour/Day")
            time_df['hour'] = time_df['timestamp'].dt.hour
            time_df['day'] = time_df['timestamp'].dt.day_name()
            heatmap_data = time_df.groupby(['day', 'hour']).sum().reset_index()
            fig = px.density_heatmap(heatmap_data, x='hour', y='day', z='count', 
                                   category_orders={"day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]})
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Log Vector Clustering" and st.session_state.log_vectors is not None:
            with st.spinner("Reducing dimensions and clustering..."):
                # Reduce dimensions with UMAP
                reducer = umap.UMAP(n_components=2, random_state=42)
                vectors_2d = reducer.fit_transform(st.session_state.log_vectors)
                
                # Cluster with KMeans
                n_clusters = min(10, len(st.session_state.log_templates))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(st.session_state.log_vectors)
                
                # Prepare visualization data
                viz_df = pd.DataFrame({
                    'x': vectors_2d[:, 0],
                    'y': vectors_2d[:, 1],
                    'cluster': clusters,
                    'type': [m['type'] for m in st.session_state.metadata_store],
                    'template': [t[:50] + '...' if len(t) > 50 else t for t in st.session_state.log_templates]
                })
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Log Vector Clusters")
                    fig = px.scatter(viz_df, x='x', y='y', color='cluster', 
                                   hover_data=['type', 'template'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### Clusters by Log Type")
                    fig = px.scatter(viz_df, x='x', y='y', color='type',
                                   hover_data=['cluster', 'template'])
                    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Log Explorer
# -----------------------------
if st.session_state.metadata_store:
    with st.expander("📝 Log Explorer", expanded=True):
        search_col, filter_col = st.columns(2)
        
        with search_col:
            log_search = st.text_input("Search within logs")
        
        with filter_col:
            type_filter = st.multiselect(
                "Filter by log type",
                options=list(set(m['type'] for m in st.session_state.metadata_store)),
                default=[]
            )
        
        # Apply filters
        filtered_logs = st.session_state.metadata_store
        if log_search:
            filtered_logs = [m for m in filtered_logs if log_search.lower() in m['original'].lower()]
        if type_filter:
            filtered_logs = [m for m in filtered_logs if m['type'] in type_filter]
        
        # Pagination
        page_size = 20
        total_pages = (len(filtered_logs) // page_size) + 1
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        # Display logs
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_logs))
        
        st.write(f"Showing {start_idx + 1}-{end_idx} of {len(filtered_logs)} logs")
        
        for log in filtered_logs[start_idx:end_idx]:
            card_class = "error-log" if log['type'] in ["ERROR", "EXCEPTION"] else "warning-log" if log['type'] == "WARNING" else "info-log"
            st.markdown(
                f"""
                <div class="log-card {card_class}">
                    <span class="log-type-tag" style="background-color: {'#FF4B4B' if log['type'] in ['ERROR', 'EXCEPTION'] else '#F0B429' if log['type'] == 'WARNING' else '#1DA1F2'}; color: white;">
                        {log['type']}
                    </span>
                    <div style="margin-top: 8px;">{log['original']}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 8px;">
                        Source: {log['source_file']} • Template: {log['template'][:100] + ('...' if len(log['template']) > 100 else '')}
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

# -----------------------------
# Data Export
# -----------------------------
if st.session_state.metadata_store:
    with st.sidebar.expander("💾 Export Options"):
        export_format = st.selectbox("Export format", ["CSV", "JSON", "Parquet"])
        export_button = st.button("Export Log Data")
        
        if export_button:
            export_df = pd.DataFrame(st.session_state.metadata_store)
            
            if export_format == "CSV":
                csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    csv,
                    "log_export.csv",
                    "text/csv"
                )
            elif export_format == "JSON":
                json_data = export_df.to_json(orient='records', indent=2)
                st.download_button(
                    "Download JSON",
                    json_data,
                    "log_export.json",
                    "application/json"
                )
            elif export_format == "Parquet":
                parquet = export_df.to_parquet(index=False)
                st.download_button(
                    "Download Parquet",
                    parquet,
                    "log_export.parquet",
                    "application/octet-stream"
                )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("""
🚀 Advanced Log Vector Search Engine | 
Built with FAISS, Drain3, and Sentence Transformers | 
© LogAnalytics 2025
""")
