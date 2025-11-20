"""
Streamlit UI for PCA-Mixed Neural Style Transfer.
"""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
import os
import sys
import time
import zipfile
import io

# Add project root to path so we can import from src package
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from src package
from src.pca_gatys import pca_gatys_style_transfer
from src.gatys import gatys_style_transfer
from src.metrics import MetricsComputer
from src.io_utils import load_image, save_image, tensor_to_image, create_comparison_grid
from src.config import DEFAULT_CONFIG, get_project_root, get_data_path
from src.utils import get_device, set_seed


# Page config
st.set_page_config(
    page_title="PCA-Mixed NST",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state
if 'results_history' not in st.session_state:
    st.session_state.results_history = []


@st.cache_resource
def load_metrics_computer():
    """Load metrics computer (cached)."""
    device = get_device()
    return MetricsComputer(device=device, model_name='vgg19')


def main():
    st.title("🎨 PCA-Mixed Neural Style Transfer")
    st.markdown("Mix two artistic styles using PCA-based covariance mixing")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Method selection
        method = st.selectbox(
            "Method",
            ["PCA (joint)", "PCA (simple)", "Gram-linear", "Cov-linear", "Gatys style1", "Gatys style2"]
        )
        
        # Alpha slider
        alpha = st.slider("Alpha (mixing coefficient)", 0.0, 1.0, 0.5, 0.05)
        st.caption("0.0 = Style 2, 1.0 = Style 1")
        
        # Loss weights
        st.subheader("Loss Weights")
        content_weight = st.number_input("Content Weight", 1e3, 1e6, 1e5, step=1e4, format="%.0e")
        style_weight = st.number_input("Style Weight", 1e3, 1e6, 3e4, step=1e4, format="%.0e")
        tv_weight = st.number_input("TV Weight", 1e-1, 1e2, 1e0, step=1e-1, format="%.1f")
        
        # Optimization settings
        st.subheader("Optimization")
        optimizer = st.selectbox("Optimizer", ["lbfgs", "adam"])
        iterations = st.number_input("Iterations", 100, 5000, 1000 if optimizer == "lbfgs" else 3000, step=100)
        init_method = st.selectbox("Initialization", ["content", "random", "style"])
        height = st.number_input("Image Height", 256, 1024, 400, step=64)
        
        # Device
        device_option = st.selectbox("Device", ["auto", "cuda", "cpu"])
        if device_option == "auto":
            device = get_device()
        else:
            device = torch.device(device_option)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎨 Style Transfer", "📊 Batch Experiments", "📈 Precomputed Results"])
    
    with tab1:
        st.header("Single Style Transfer")
        
        # Image upload/selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Content Image")
            content_option = st.radio("Content source", ["Upload", "Example"], horizontal=True)
            if content_option == "Upload":
                content_file = st.file_uploader("Upload content", type=["jpg", "jpeg", "png"], key="content")
                if content_file:
                    content_path = save_uploaded_file(content_file, "content")
                else:
                    content_path = None
            else:
                content_examples = get_example_images("content")
                if content_examples:
                    content_choice = st.selectbox("Choose content", content_examples, key="content_sel")
                    content_path = os.path.join(get_data_path("content_examples"), content_choice)
                else:
                    content_path = None
            
            if content_path and os.path.exists(content_path):
                st.image(load_image(content_path), use_container_width=True)
        
        with col2:
            st.subheader("Style Image 1")
            style1_option = st.radio("Style1 source", ["Upload", "Example"], horizontal=True, key="style1_opt")
            if style1_option == "Upload":
                style1_file = st.file_uploader("Upload style 1", type=["jpg", "jpeg", "png"], key="style1")
                if style1_file:
                    style1_path = save_uploaded_file(style1_file, "style1")
                else:
                    style1_path = None
            else:
                style_examples = get_example_images("style")
                if style_examples:
                    style1_choice = st.selectbox("Choose style 1", style_examples, key="style1_sel")
                    style1_path = os.path.join(get_data_path("style_examples"), style1_choice)
                else:
                    style1_path = None
            
            if style1_path and os.path.exists(style1_path):
                st.image(load_image(style1_path), use_container_width=True)
        
        with col3:
            st.subheader("Style Image 2")
            style2_option = st.radio("Style2 source", ["Upload", "Example"], horizontal=True, key="style2_opt")
            if style2_option == "Upload":
                style2_file = st.file_uploader("Upload style 2", type=["jpg", "jpeg", "png"], key="style2")
                if style2_file:
                    style2_path = save_uploaded_file(style2_file, "style2")
                else:
                    style2_path = None
            else:
                style_examples = get_example_images("style")
                if style_examples:
                    style2_choice = st.selectbox("Choose style 2", style_examples, key="style2_sel")
                    style2_path = os.path.join(get_data_path("style_examples"), style2_choice)
                else:
                    style2_path = None
            
            if style2_path and os.path.exists(style2_path):
                st.image(load_image(style2_path), use_container_width=True)
        
        # Run button
        if st.button("🚀 Generate Style Transfer", type="primary", use_container_width=True):
            if not content_path or not style1_path:
                st.error("Please provide content and style 1 images")
            elif method in ["PCA (joint)", "PCA (simple)", "Gram-linear", "Cov-linear"] and not style2_path:
                st.error("Please provide style 2 image for mixing methods")
            else:
                with st.spinner("Generating style transfer..."):
                    # Build config
                    config = DEFAULT_CONFIG.copy()
                    config.update({
                        'content_weight': content_weight,
                        'style_weight': style_weight,
                        'tv_weight': tv_weight,
                        'optimizer': optimizer,
                        'iterations': iterations if optimizer == 'lbfgs' else None,
                        'adam_iterations': iterations if optimizer == 'adam' else None,
                        'init_method': init_method,
                        'height': height,
                        'device': device
                    })
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(iteration, loss_dict):
                        if optimizer == 'adam':
                            progress = iteration / iterations
                        else:
                            progress = min(iteration / iterations, 0.99)
                        progress_bar.progress(progress)
                        status_text.text(f"Iteration {iteration}/{iterations} | Loss: {loss_dict['total_loss']:.2e}")
                    
                    start_time = time.time()
                    
                    try:
                        # Map method names
                        method_map = {
                            "PCA (joint)": ("pca", "joint"),
                            "PCA (simple)": ("pca", "simple"),
                            "Gram-linear": ("pca", "gram-linear"),
                            "Cov-linear": ("pca", "covariance-linear"),
                            "Gatys style1": ("gatys", None),
                            "Gatys style2": ("gatys", None)
                        }
                        
                        method_type, mixing_method = method_map[method]
                        
                        if method_type == "gatys":
                            result, metrics = gatys_style_transfer(
                                content_path,
                                style1_path if "style1" in method else style2_path,
                                output_path=None,
                                config=config,
                                progress_callback=progress_callback
                            )
                        else:
                            result, metrics = pca_gatys_style_transfer(
                                content_path, style1_path, style2_path,
                                alpha=alpha, mixing_method=mixing_method,
                                output_path=None, config=config,
                                progress_callback=progress_callback
                            )
                        
                        runtime = time.time() - start_time
                        progress_bar.progress(1.0)
                        status_text.text(f"Complete! Runtime: {runtime:.2f}s")
                        
                        # Display result
                        st.success("Style transfer complete!")
                        
                        # Convert to display format
                        result_np = tensor_to_image(result, denormalize=True)
                        
                        # Show comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(result_np, caption="Generated Result", use_container_width=True)
                        
                        with col2:
                            # Compute and display metrics
                            metrics_computer = load_metrics_computer()
                            
                            from src.io_utils import prepare_img
                            content_img = prepare_img(content_path, height, device)
                            style1_img = prepare_img(style1_path, height, device) if style1_path else None
                            style2_img = prepare_img(style2_path, height, device) if style2_path else None
                            
                            all_metrics = metrics_computer.compute_all_metrics(
                                result, content_img, style1_img, style2_img, runtime
                            )
                            
                            st.subheader("Metrics")
                            st.metric("LPIPS (content)", f"{all_metrics.get('lpips_content', 0):.4f}")
                            st.metric("SSIM (content)", f"{all_metrics.get('ssim_content', 0):.4f}")
                            st.metric("PSNR (content)", f"{all_metrics.get('psnr_content', 0):.2f} dB")
                            st.metric("Runtime", f"{runtime:.2f}s")
                            
                            if style1_img is not None:
                                st.metric("Gram dist (style1)", f"{all_metrics.get('gram_dist_style1_avg', 0):.2f}")
                            if style2_img is not None:
                                st.metric("Gram dist (style2)", f"{all_metrics.get('gram_dist_style2_avg', 0):.2f}")
                        
                        # Save option
                        if st.button("💾 Save Result"):
                            output_dir = get_data_path("outputs")
                            os.makedirs(output_dir, exist_ok=True)
                            timestamp = int(time.time())
                            output_path = os.path.join(output_dir, f"result_{timestamp}.jpg")
                            save_image(result, output_path, denormalize=True)
                            st.success(f"Saved to {output_path}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    with tab2:
        st.header("Batch Experiments")
        st.markdown("Generate grids across multiple alpha values")
        
        # Select images
        content_examples = get_example_images("content")
        style_examples = get_example_images("style")
        
        if content_examples and len(style_examples) >= 2:
            content_choice = st.selectbox("Content Image", content_examples)
            style1_choice = st.selectbox("Style 1", style_examples)
            style2_choice = st.selectbox("Style 2", style_examples, index=1 if len(style_examples) > 1 else 0)
            
            # Alpha values
            alpha_str = st.text_input("Alpha values (comma-separated)", "0.0,0.25,0.5,0.75,1.0")
            try:
                alphas = [float(x.strip()) for x in alpha_str.split(",")]
            except:
                st.error("Invalid alpha values")
                alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
            
            # Methods
            selected_methods = st.multiselect(
                "Methods",
                ["pca-joint", "pca-simple", "gram-linear", "cov-linear", "gatys-style1", "gatys-style2"],
                default=["pca-joint", "gram-linear"]
            )
            
            if st.button("🚀 Run Batch Experiment"):
                with st.spinner("Running batch experiment..."):
                    from src.experiments import run_alpha_grid
                    
                    content_path = os.path.join(get_data_path("content_examples"), content_choice)
                    style1_path = os.path.join(get_data_path("style_examples"), style1_choice)
                    style2_path = os.path.join(get_data_path("style_examples"), style2_choice)
                    
                    output_dir = get_data_path("outputs/batch")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    config = DEFAULT_CONFIG.copy()
                    config.update({
                        'content_weight': content_weight,
                        'style_weight': style_weight,
                        'tv_weight': tv_weight,
                        'optimizer': optimizer,
                        'iterations': iterations if optimizer == 'lbfgs' else None,
                        'adam_iterations': iterations if optimizer == 'adam' else None,
                        'init_method': init_method,
                        'height': height,
                        'device': device
                    })
                    
                    df, grid_paths = run_alpha_grid(
                        content_path, style1_path, style2_path,
                        alphas, selected_methods, output_dir, config
                    )
                    
                    st.success("Batch experiment complete!")
                    
                    # Display grids
                    for method, grid_path in grid_paths.items():
                        st.subheader(f"{method} Grid")
                        st.image(grid_path, use_container_width=True)
                    
                    # Display metrics
                    if not df.empty:
                        st.subheader("Metrics Summary")
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            "metrics_summary.csv",
                            "text/csv"
                        )
        else:
            st.info("Please add content and style images to the data directories")
    
    with tab3:
        st.header("Precomputed Results")
        st.markdown("View previously generated results and metrics")
        
        results_dir = get_data_path("outputs")
        if os.path.exists(results_dir):
            result_files = list(Path(results_dir).glob("*.jpg")) + list(Path(results_dir).glob("*.png"))
            if result_files:
                selected_file = st.selectbox("Select result", [f.name for f in result_files])
                if selected_file:
                    img_path = os.path.join(results_dir, selected_file)
                    st.image(load_image(img_path), use_container_width=True)
            else:
                st.info("No precomputed results found")
        else:
            st.info("Results directory not found")


def save_uploaded_file(uploaded_file, prefix: str) -> str:
    """Save uploaded file to temp directory."""
    temp_dir = os.path.join(get_project_root(), "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, f"{prefix}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


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


if __name__ == "__main__":
    main()

