"""
Streamlit UI for PCA-Mixed Neural Style Transfer with automatic saving.
Three tabs: Run, Grid, Comparisons.
"""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.runner import run_once
from src.io_utils import load_image, tensor_to_image
from src.config import DEFAULT_CONFIG, get_data_path
from src.utils import get_device, set_seed


# Page config
st.set_page_config(
    page_title="PCA-Mixed NST",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state
if 'runs_history' not in st.session_state:
    st.session_state.runs_history = []


def get_example_images(image_type: str) -> list:
    """Get list of example images."""
    if image_type == "content":
        dir_path = get_data_path("content_examples")
    else:
        dir_path = get_data_path("style_examples")
    
    if os.path.exists(dir_path):
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        return sorted(files)
    return []


def discover_runs() -> list:
    """Discover all run folders."""
    runs_dir = Path("data/outputs/runs")
    if not runs_dir.exists():
        return []
    
    runs = []
    for run_folder in sorted(runs_dir.iterdir(), reverse=True):
        if run_folder.is_dir():
            meta_path = run_folder / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    runs.append({
                        'folder': run_folder,
                        'meta': meta,
                        'name': run_folder.name
                    })
                except Exception:
                    pass
    
    return runs


def main():
    st.title("🎨 PCA-Mixed Neural Style Transfer")
    st.markdown("Automatic saving of images, metrics, and metadata for every run")
    
    tab1, tab2, tab3 = st.tabs(["Run", "Grid", "Comparisons"])
    
    with tab1:
        st.header("Single Run")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Images")
            
            content_option = st.radio("Content source", ["Upload", "Example"], horizontal=True)
            if content_option == "Upload":
                content_file = st.file_uploader("Upload content", type=["jpg", "jpeg", "png"], key="content")
                if content_file:
                    temp_dir = Path("data/temp")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    content_path = temp_dir / f"content_{content_file.name}"
                    with open(content_path, "wb") as f:
                        f.write(content_file.getbuffer())
                    st.image(load_image(str(content_path)), use_container_width=True)
                else:
                    content_path = None
            else:
                content_examples = get_example_images("content")
                if content_examples:
                    content_choice = st.selectbox("Choose content", content_examples, key="content_sel")
                    content_path = str(Path(get_data_path("content_examples")) / content_choice)
                    st.image(load_image(content_path), use_container_width=True)
                else:
                    content_path = None
            
            style1_option = st.radio("Style1 source", ["Upload", "Example"], horizontal=True, key="style1_opt")
            if style1_option == "Upload":
                style1_file = st.file_uploader("Upload style 1", type=["jpg", "jpeg", "png"], key="style1")
                if style1_file:
                    temp_dir = Path("data/temp")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    style1_path = temp_dir / f"style1_{style1_file.name}"
                    with open(style1_path, "wb") as f:
                        f.write(style1_file.getbuffer())
                    st.image(load_image(str(style1_path)), use_container_width=True)
                else:
                    style1_path = None
            else:
                style_examples = get_example_images("style")
                if style_examples:
                    style1_choice = st.selectbox("Choose style 1", style_examples, key="style1_sel")
                    style1_path = str(Path(get_data_path("style_examples")) / style1_choice)
                    st.image(load_image(style1_path), use_container_width=True)
                else:
                    style1_path = None
            
            style2_option = st.radio("Style2 source", ["Upload", "Example"], horizontal=True, key="style2_opt")
            if style2_option == "Upload":
                style2_file = st.file_uploader("Upload style 2", type=["jpg", "jpeg", "png"], key="style2")
                if style2_file:
                    temp_dir = Path("data/temp")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    style2_path = temp_dir / f"style2_{style2_file.name}"
                    with open(style2_path, "wb") as f:
                        f.write(style2_file.getbuffer())
                    st.image(load_image(str(style2_path)), use_container_width=True)
                else:
                    style2_path = None
            else:
                style_examples = get_example_images("style")
                if style_examples:
                    style2_choice = st.selectbox("Choose style 2", style_examples, key="style2_sel")
                    style2_path = str(Path(get_data_path("style_examples")) / style2_choice)
                    st.image(load_image(style2_path), use_container_width=True)
                else:
                    style2_path = None
        
        with col2:
            st.subheader("Parameters")
            
            method = st.selectbox(
                "Method",
                ["pca_joint", "pca_simple", "gram-linear", "covariance-linear", "gatys"],
                help="pca_joint/simple: PCA mixing, gram-linear: Gram matrix mixing, gatys: single style"
            )
            
            alpha = st.slider("Alpha (mixing coefficient)", 0.0, 1.0, 0.5, 0.05)
            st.caption("0.0 = Style 2, 1.0 = Style 1")
            
            iterations = st.number_input("Iterations", 10, 5000, 100, step=10)
            snapshot_interval = st.number_input("Snapshot Interval", 1, 100, 10, step=1)
            height = st.number_input("Image Height", 256, 1024, 400, step=64)
            
            optimizer = st.selectbox("Optimizer", ["lbfgs", "adam"])
            
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                content_weight = st.number_input("Content Weight", 1e3, 1e6, 1e5, step=1e4, format="%.0e")
            with col_w2:
                style_weight = st.number_input("Style Weight", 1e3, 1e6, 3e4, step=1e4, format="%.0e")
            with col_w3:
                tv_weight = st.number_input("TV Weight", 1e-1, 1e2, 1e0, step=1e-1, format="%.1f")
            
            seed = st.number_input("Seed", 0, 999999, 42, step=1)
            
            device_option = st.selectbox("Device", ["auto", "cuda", "cpu"])
            device = get_device() if device_option == "auto" else device_option
        
        # Run button
        if st.button("🚀 Start Run", type="primary", use_container_width=True):
            if not content_path or not style1_path:
                st.error("Please provide content and style 1 images")
            elif method not in ["gatys"] and not style2_path:
                st.error("Please provide style 2 image for mixing methods")
            else:
                config = {
                    'content_img_path': content_path,
                    'style1_img_path': style1_path,
                    'style2_img_path': style2_path if style2_path else style1_path,
                    'mixing_method': method,
                    'alpha': alpha,
                    'iterations': iterations,
                    'snapshot_interval': snapshot_interval,
                    'height': height,
                    'optimizer': optimizer,
                    'content_weight': content_weight,
                    'style_weight': style_weight,
                    'tv_weight': tv_weight,
                    'seed': seed,
                    'device': device
                }
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                image_placeholder = st.empty()
                
                try:
                    # Run with progress callback for UI updates
                    import threading
                    run_complete = threading.Event()
                    run_folder_ref = [None]
                    
                    def run_in_thread():
                        try:
                            run_folder_ref[0] = run_once(config)
                        finally:
                            run_complete.set()
                    
                    # Start run in thread
                    thread = threading.Thread(target=run_in_thread, daemon=True)
                    thread.start()
                    
                    # Monitor progress by reading metrics CSV
                    metrics_path = None
                    latest_iter = 0
                    
                    while not run_complete.is_set() or latest_iter < iterations:
                        if run_folder_ref[0] is not None and metrics_path is None:
                            metrics_path = run_folder_ref[0] / "metrics" / "metrics_summary.csv"
                        
                        if metrics_path and metrics_path.exists():
                            try:
                                df = pd.read_csv(metrics_path)
                                if len(df) > 0:
                                    latest_iter = int(df.iloc[-1]['iteration'])
                                    progress = latest_iter / iterations
                                    progress_bar.progress(min(progress, 0.99))
                                    
                                    # Show latest snapshot
                                    snapshot_num = (latest_iter // snapshot_interval) * snapshot_interval
                                    if snapshot_num > 0:
                                        snapshot_path = run_folder_ref[0] / "images" / f"iter_{snapshot_num:03d}.png"
                                        if snapshot_path.exists():
                                            image_placeholder.image(load_image(str(snapshot_path)), use_container_width=True)
                                    
                                    # Show latest metrics
                                    latest = df.iloc[-1]
                                    status_text.text(
                                        f"Iteration {latest_iter}/{iterations} | "
                                        f"Loss: {latest['total_loss']:.2e} | "
                                        f"Content: {latest['content_loss']:.2e} | "
                                        f"Style: {latest['style_loss']:.2e}"
                                    )
                            except Exception:
                                pass  # File might be locked, try again
                        
                        import time
                        time.sleep(0.3)
                    
                    # Wait for thread to complete
                    thread.join(timeout=1.0)
                    run_folder = run_folder_ref[0]
                    
                    if run_folder is None:
                        st.error("Run failed to complete")
                        return
                    
                    progress_bar.progress(1.0)
                    
                    # Show final image
                    final_path = run_folder / "images" / f"iter_{iterations:03d}.png"
                    if final_path.exists():
                        image_placeholder.image(load_image(str(final_path)), use_container_width=True)
                    
                    # Load final metrics
                    final_metrics_path = run_folder / "metrics" / "final_metrics.json"
                    if final_metrics_path.exists():
                        with open(final_metrics_path, 'r') as f:
                            final_metrics = json.load(f)
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Final Loss", f"{final_metrics.get('total_loss', 0):.4f}")
                            st.metric("Content Loss", f"{final_metrics.get('content_loss', 0):.4f}")
                        with col_m2:
                            st.metric("Style Loss", f"{final_metrics.get('style_loss', 0):.4f}")
                            if final_metrics.get('lpips'):
                                st.metric("LPIPS", f"{final_metrics['lpips']:.4f}")
                        with col_m3:
                            if final_metrics.get('ssim'):
                                st.metric("SSIM", f"{final_metrics['ssim']:.4f}")
                            if final_metrics.get('gram_dist'):
                                st.metric("Gram Dist", f"{final_metrics['gram_dist']:.2f}")
                    
                    st.success(f"✅ Run completed! All outputs automatically saved.")
                    st.info(f"📁 **Run folder**: `{run_folder}`")
                    st.info(f"📊 View saved outputs: [Comparisons tab](#comparisons)")
                    
                    # Show what was saved
                    with st.expander("📋 Files Saved"):
                        st.write(f"**Images**: {len(list((run_folder / 'images').glob('*.png')))} snapshot(s)")
                        st.write(f"**Metrics**: `metrics/metrics_summary.csv` ({iterations} rows)")
                        st.write(f"**Final Metrics**: `metrics/final_metrics.json`")
                        st.write(f"**Metadata**: `meta.json`")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with tab2:
        st.header("Grid / Batch Runs")
        
        st.subheader("Alpha Grid")
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            content_examples = get_example_images("content")
            style_examples = get_example_images("style")
            
            if content_examples and len(style_examples) >= 2:
                content_choice = st.selectbox("Content Image", content_examples, key="grid_content")
                style1_choice = st.selectbox("Style 1", style_examples, key="grid_style1")
                style2_choice = st.selectbox("Style 2", style_examples, index=1 if len(style_examples) > 1 else 0, key="grid_style2")
                
                method_grid = st.selectbox("Method", ["pca_joint", "pca_simple", "gram-linear"], key="grid_method")
                
                alpha_str = st.text_input("Alpha values (comma-separated)", "0.0,0.25,0.5,0.75,1.0", key="grid_alphas")
                try:
                    alphas = [float(x.strip()) for x in alpha_str.split(",")]
                except:
                    st.error("Invalid alpha values")
                    alphas = []
                
                iterations_grid = st.number_input("Iterations per run", 10, 1000, 100, step=10, key="grid_iters")
                snapshot_interval_grid = st.number_input("Snapshot Interval", 1, 100, 10, step=1, key="grid_snap")
                height_grid = st.number_input("Image Height", 256, 1024, 400, step=64, key="grid_height")
                seed_grid = st.number_input("Seed", 0, 999999, 42, step=1, key="grid_seed")
                
                if st.button("🚀 Run Grid", type="primary"):
                    if alphas:
                        content_path = str(Path(get_data_path("content_examples")) / content_choice)
                        style1_path = str(Path(get_data_path("style_examples")) / style1_choice)
                        style2_path = str(Path(get_data_path("style_examples")) / style2_choice)
                        
                        progress_table = st.empty()
                        run_folders = []
                        
                        for idx, alpha in enumerate(alphas):
                            config = {
                                'content_img_path': content_path,
                                'style1_img_path': style1_path,
                                'style2_img_path': style2_path,
                                'mixing_method': method_grid,
                                'alpha': alpha,
                                'iterations': iterations_grid,
                                'snapshot_interval': snapshot_interval_grid,
                                'height': height_grid,
                                'optimizer': 'lbfgs',
                                'seed': seed_grid,
                                'device': get_device()
                            }
                            
                            try:
                                run_folder = run_once(config)
                                run_folders.append(run_folder)
                                
                                # Update progress table
                                progress_data = []
                                for i, a in enumerate(alphas):
                                    status = "✅" if i <= idx else "⏳"
                                    progress_data.append({
                                        'Alpha': a,
                                        'Status': status,
                                        'Folder': str(run_folders[i]) if i < len(run_folders) else ''
                                    })
                                progress_table.dataframe(pd.DataFrame(progress_data))
                                
                            except Exception as e:
                                st.error(f"Error running alpha={alpha}: {e}")
                        
                        st.success(f"Grid completed! {len(run_folders)} runs saved.")
                        st.info("View comparisons: [Comparisons tab](#comparisons)")
            else:
                st.info("Please add content and style images to data directories")
    
    with tab3:
        st.header("Comparisons")
        
        runs = discover_runs()
        
        if not runs:
            st.info("No saved runs found. Run some experiments first!")
        else:
            st.subheader(f"Found {len(runs)} saved runs")
            
            # Run selection
            run_names = [r['name'] for r in runs]
            selected_runs = st.multiselect(
                "Select runs to compare (2-4 runs)",
                run_names,
                max_selections=4
            )
            
            if len(selected_runs) < 2:
                st.info("Select 2-4 runs to compare")
            else:
                selected_run_data = [r for r in runs if r['name'] in selected_runs]
                
                # Display images
                st.subheader("Final Images")
                cols = st.columns(len(selected_run_data))
                for idx, run_data in enumerate(selected_run_data):
                    with cols[idx]:
                        final_path = run_data['folder'] / "images" / f"iter_{run_data['meta'].get('iterations_completed', 0):03d}.png"
                        if final_path.exists():
                            st.image(load_image(str(final_path)), use_container_width=True)
                            st.caption(f"{run_data['meta'].get('mixing_method', 'unknown')} α={run_data['meta'].get('alpha', 0):.2f}")
                        else:
                            st.info("Final image not found")
                
                # Metrics comparison
                st.subheader("Metrics Comparison")
                
                metrics_data = []
                for run_data in selected_run_data:
                    final_metrics_path = run_data['folder'] / "metrics" / "final_metrics.json"
                    if final_metrics_path.exists():
                        with open(final_metrics_path, 'r') as f:
                            final_metrics = json.load(f)
                        
                        meta = run_data['meta']
                        metrics_data.append({
                            'Run': run_data['name'][:30] + "...",
                            'Method': meta.get('mixing_method', 'unknown'),
                            'Alpha': meta.get('alpha', 0),
                            'Iterations': meta.get('iterations_completed', 0),
                            'Content Loss': final_metrics.get('content_loss', 0),
                            'Style Loss': final_metrics.get('style_loss', 0),
                            'Total Loss': final_metrics.get('total_loss', 0),
                            'LPIPS': final_metrics.get('lpips', ''),
                            'SSIM': final_metrics.get('ssim', ''),
                            'Gram Dist': final_metrics.get('gram_dist', '')
                        })
                
                if metrics_data:
                    df_compare = pd.DataFrame(metrics_data)
                    st.dataframe(df_compare, use_container_width=True)
                    
                    # Simple plots
                    if len(metrics_data) > 1:
                        col_p1, col_p2 = st.columns(2)
                        
                        with col_p1:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            methods = [m['Method'] for m in metrics_data]
                            losses = [m['Total Loss'] for m in metrics_data]
                            ax.bar(range(len(methods)), losses)
                            ax.set_xticks(range(len(methods)))
                            ax.set_xticklabels(methods, rotation=45, ha='right')
                            ax.set_ylabel('Total Loss')
                            ax.set_title('Total Loss Comparison')
                            st.pyplot(fig)
                        
                        with col_p2:
                            if any(m.get('LPIPS') for m in metrics_data):
                                fig, ax = plt.subplots(figsize=(6, 4))
                                methods = [m['Method'] for m in metrics_data]
                                lpips_vals = [m.get('LPIPS', 0) if m.get('LPIPS') != '' else 0 for m in metrics_data]
                                if any(lpips_vals):
                                    ax.bar(range(len(methods)), lpips_vals)
                                    ax.set_xticks(range(len(methods)))
                                    ax.set_xticklabels(methods, rotation=45, ha='right')
                                    ax.set_ylabel('LPIPS')
                                    ax.set_title('LPIPS Comparison')
                                    st.pyplot(fig)
                
                # Show run folders
                st.subheader("Run Folders")
                for run_data in selected_run_data:
                    st.text(str(run_data['folder']))


if __name__ == "__main__":
    main()
