#!/usr/bin/env python3
"""
This script creates a web interface for generating topic semantic maps from CSV files.
Run with: streamlit run create_map.py
"""

import streamlit as st
from topic_semantic_map import SemanticTaxonomyVisualizer
import pandas as pd
import numpy as np
import tempfile
import os

def validate_csv_structure(df):
    """
    Validate that the uploaded CSV has the required structure.
    
    Args:
        df: pandas DataFrame to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    required_columns = ['topic_id', 'name']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    if df['topic_id'].duplicated().any():
        return False, "Duplicate topic_id values found. Each topic_id must be unique."
    
    if df['topic_id'].isnull().any():
        return False, "topic_id column contains null values."
    
    if df['name'].isnull().any():
        return False, "name column contains null values."
    
    return True, ""

def create_map_from_dataframe(df):
    """
    Creates a semantic map from a pandas DataFrame.
    
    Args:
        df: pandas DataFrame with topic data
        
    Returns:
        tuple: (visualizer, fig) or (None, None) if error
    """
    try:
        # Initialize the visualizer
        visualizer = SemanticTaxonomyVisualizer()

        # Convert DataFrame to the format expected by the visualizer
        # Convert NaN values to None for proper handling
        df_clean = df.where(pd.notnull(df), None)
        topics_data = df_clean.to_dict('records')
        
        # Load topics
        visualizer.load_topics_from_dict(topics_data)
        
        if len(visualizer.topics) == 0:
            st.error("No topics were loaded from the CSV file.")
            return None, None
        
        # Compute semantic embeddings
        with st.spinner("Computing semantic embeddings..."):
            visualizer.compute_embeddings(force_recompute=True)
        
        # Reduce dimensions to 2D
        with st.spinner("Reducing dimensions to 2D..."):
            n_topics = len(visualizer.topics)
            n_neighbors = min(5, max(2, n_topics - 1))  # Ensure valid n_neighbors
            
            try:
                visualizer.reduce_dimensions(
                    method='umap', 
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    random_state=42
                )
            except Exception as e:
                st.warning(f"UMAP failed: {e}. Trying t-SNE instead...")
                visualizer.reduce_dimensions(method='tsne', random_state=42)
        
        # Create the interactive visualization
        with st.spinner("Creating interactive visualization..."):
            fig = visualizer.create_interactive_visualization(
                title="Topic Semantic Map",
                show_hierarchy_edges=True
            )
        
        return visualizer, fig
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None, None

def main():
    """
    Main Streamlit application.
    """
    st.set_page_config(
        page_title="Topic Semantic Map Generator", 
        page_icon="🗺️",
        layout="wide"
    )
    
    st.title("🗺️ Topic Semantic Map Generator")
    st.markdown("""
    Upload a CSV file containing your educational topics to generate an interactive semantic map.
    The map will show relationships between topics based on their content similarity.
    """)
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **Required CSV Columns:**
        - `topic_id`: Unique identifier
        - `name`: Display name
        
        **Optional Columns:**
        - `curriculum_text`: Detailed description (recommended for better semantic analysis)
        - `parent_id`: For hierarchical structure
        - `country`: For color coding
        - `school_type`: For symbol shapes
        - `grade`: For additional classification
        
        **Example CSV structure:**
        ```
        topic_id,name,curriculum_text,parent_id
        math,Mathematics,"Core math concepts",
        algebra,Algebra,"Variables and equations",math
        ```
        """)
        
        st.header("📖 Need Help?")
        if st.button("View Data Preparation Guide"):
            st.markdown("[Open DATA_PREP_GUIDE.md](./DATA_PREP_GUIDE.md)")
    
    # File upload section
    st.header("📁 Upload Your CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing your topic data"
    )
    
    # Sample data section
    with st.expander("🔍 Don't have a CSV? Try our sample data"):
        sample_options = {
            "My Topics": "my_topics.csv",
            "German Math Curriculum": "german_math_curriculum.csv",
            "Test Data": "test_german_math_semantic_topics.csv"
        }
        
        selected_sample = st.selectbox("Choose sample data:", list(sample_options.keys()))
        
        if st.button("Load Sample Data"):
            try:
                sample_file = sample_options[selected_sample]
                if os.path.exists(sample_file):
                    df = pd.read_csv(sample_file)
                    st.session_state['uploaded_df'] = df
                    st.success(f"Loaded sample data: {selected_sample}")
                else:
                    st.error(f"Sample file {sample_file} not found.")
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_df'] = df
            st.success(f"File uploaded successfully! Found {len(df)} topics.")
            
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return
    
    # Display and process data if available
    if 'uploaded_df' in st.session_state:
        df = st.session_state['uploaded_df']
        
        # Show data preview
        st.header("📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Validate CSV structure
        is_valid, error_message = validate_csv_structure(df)
        
        if not is_valid:
            st.error(f"❌ CSV Validation Error: {error_message}")
            return
        else:
            st.success("✅ CSV structure is valid!")
        
        # Configuration options
        st.header("⚙️ Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            show_hierarchy = st.checkbox("Show hierarchy connections", value=True)
            
        with col2:
            map_title = st.text_input("Map title", value="Topic Semantic Map")
        
        # Generate map button
        if st.button("🚀 Generate Semantic Map", type="primary"):
            visualizer, fig = create_map_from_dataframe(df)
            
            if fig is not None:
                # Update title and hierarchy setting
                fig.update_layout(title=map_title)
                
                # Display the map
                st.header("🗺️ Interactive Semantic Map")
                st.plotly_chart(fig, use_container_width=True)
                
                # Provide download options
                st.header("💾 Download Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download HTML
                    html_str = fig.to_html()
                    st.download_button(
                        label="📄 Download HTML",
                        data=html_str,
                        file_name="topic_semantic_map.html",
                        mime="text/html"
                    )
                
                with col2:
                    # Download coordinates
                    if visualizer.coordinates is not None:
                        csv_str = visualizer.coordinates.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Coordinates CSV",
                            data=csv_str,
                            file_name="topic_coordinates.csv",
                            mime="text/csv"
                        )
                
                with col3:
                    # Download similarity matrix
                    if visualizer.similarity_matrix is not None:
                        # Convert similarity matrix to bytes
                        similarity_bytes = visualizer.similarity_matrix.tobytes()
                        st.download_button(
                            label="🔗 Download Similarity Matrix",
                            data=similarity_bytes,
                            file_name="similarity_matrix.npy",
                            mime="application/octet-stream"
                        )
                
                # Show statistics
                st.header("📈 Map Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Topics", len(visualizer.topics))
                
                with col2:
                    root_topics = len([t for t in visualizer.topics.values() if t.parent_id is None])
                    st.metric("Root Topics", root_topics)
                
                with col3:
                    countries = set(t.country for t in visualizer.topics.values() if t.country)
                    st.metric("Countries", len(countries) if countries else 0)
                
                with col4:
                    school_types = set(t.school_type for t in visualizer.topics.values() if t.school_type)
                    st.metric("School Types", len(school_types) if school_types else 0)

if __name__ == "__main__":
    main()
