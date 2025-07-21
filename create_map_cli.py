#!/usr/bin/env python3
"""
This script generates a topic semantic map from a CSV file using command line interface.
This is the original CLI version, for web interface use create_map.py with streamlit.
"""

from topic_semantic_map import SemanticTaxonomyVisualizer
import numpy as np

def create_map_from_csv(csv_path: str):
    """
    Loads topics from a CSV, creates a map, and displays it.
    
    Args:
        csv_path: The path to the input CSV file.
    """
    print(f"--- Creating Semantic Map from '{csv_path}' ---")

    # 1. Initialize the visualizer
    visualizer = SemanticTaxonomyVisualizer()

    # 2. Load topics directly from your CSV file
    print("1. Loading topic data...")
    try:
        visualizer.load_topics_from_csv(csv_path)
        print(f"   Loaded {len(visualizer.topics)} topics successfully.")
        
        # Debug: Check if curriculum_text is loaded properly
        for topic_id, topic in visualizer.topics.items():
            print(f"   - {topic_id}: '{topic.curriculum_text[:50]}...'")
            
    except FileNotFoundError:
        print(f"   Error: The file '{csv_path}' was not found.")
        return
    except Exception as e:
        print(f"   Error loading CSV: {e}")
        return

    # 3. Compute semantic embeddings for the topics
    print("\n2. Computing semantic embeddings...")
    try:
        visualizer.compute_embeddings(force_recompute=True)
        print("   Embeddings have been computed.")
        
        # Debug: Check if embeddings were computed
        for topic_id, topic in visualizer.topics.items():
            if topic.embedding is not None:
                print(f"   - {topic_id}: embedding shape {topic.embedding.shape}")
            else:
                print(f"   - {topic_id}: NO EMBEDDING!")
                
    except Exception as e:
        print(f"   Error computing embeddings: {e}")
        return

    # 4. Reduce embeddings to 2D for visualization
    print("\n3. Reducing dimensions to 2D...")
    try:
        # For small datasets, use smaller n_neighbors
        n_topics = len(visualizer.topics)
        n_neighbors = min(5, n_topics - 1)  # Ensure n_neighbors < n_samples
        
        visualizer.reduce_dimensions(
            method='umap', 
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42
        )
        print("   2D coordinates are ready.")
        
    except Exception as e:
        print(f"   Error reducing dimensions: {e}")
        print(f"   Number of topics: {len(visualizer.topics)}")
        # Try with t-SNE instead
        print("   Trying t-SNE instead...")
        try:
            visualizer.reduce_dimensions(method='tsne', random_state=42)
            print("   t-SNE reduction successful.")
        except Exception as e2:
            print(f"   t-SNE also failed: {e2}")
            return

    # 5. Create the interactive visualization
    print("\n4. Creating the interactive map...")
    try:
        fig = visualizer.create_interactive_visualization(
            title="My Topic Semantic Map (from CSV)",
            show_hierarchy_edges=True  # Shows links between parents and children
        )
        print("   Map created.")
    except Exception as e:
        print(f"   Error creating visualization: {e}")
        return

    # 6. Display the visualization in your browser
    print("\n5. Opening the map in your browser...")
    try:
        fig.show()
    except Exception as e:
        print(f"   Warning: Could not open in browser: {e}")

    # 7. (Optional) Save the map to an HTML file for sharing
    output_html_file = 'my_topic_map.html'
    print(f"\n6. Saving map to '{output_html_file}'...")
    try:
        fig.write_html(output_html_file)
        print(f"   Successfully saved to {output_html_file}")
    except Exception as e:
        print(f"   Error saving HTML file: {e}")

    print("\n--- Process Complete ---")


if __name__ == "__main__":
    # Make sure your CSV file is named 'my_topics.csv' or change the path here
    create_map_from_csv('my_topics.csv') 