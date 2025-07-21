#!/usr/bin/env python3
"""
This script generates a hierarchical topic semantic map for German Mathematics curriculum.
"""

from topic_semantic_map import SemanticTaxonomyVisualizer
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def create_german_math_map(csv_path: str = 'german_math_curriculum.csv'):
    """
    Creates a hierarchical semantic map for German mathematics curriculum.
    
    Args:
        csv_path: Path to the German math curriculum CSV file
    """
    print(f"--- Creating German Mathematics Curriculum Map ---")

    # 1. Initialize the visualizer
    visualizer = SemanticTaxonomyVisualizer()

    # 2. Load the German math curriculum
    print("1. Loading German mathematics curriculum...")
    try:
        visualizer.load_topics_from_csv(csv_path)
        print(f"   Loaded {len(visualizer.topics)} topics successfully.")
        
        # Show the hierarchical structure
        print(f"\n   Curriculum Structure:")
        subject_count = len([t for t in visualizer.topics.values() if t.parent_id is None])
        topic_count = len([t for t in visualizer.topics.values() if t.parent_id and t.parent_id in visualizer.topics and visualizer.topics[t.parent_id].parent_id is None])
        subtopic_count = len(visualizer.topics) - subject_count - topic_count
        
        print(f"   - Subjects: {subject_count}")
        print(f"   - Main Topics: {topic_count}")
        print(f"   - Subtopics & Concepts: {subtopic_count}")
            
    except FileNotFoundError:
        print(f"   Error: The file '{csv_path}' was not found.")
        return
    except Exception as e:
        print(f"   Error loading CSV: {e}")
        return

    # 3. Compute semantic embeddings
    print("\n2. Computing semantic embeddings...")
    try:
        visualizer.compute_embeddings(force_recompute=True)
        print("   Embeddings computed successfully.")
    except Exception as e:
        print(f"   Error computing embeddings: {e}")
        return

    # 4. Reduce to 2D coordinates with UMAP optimized for hierarchical data
    print("\n3. Creating 2D layout with UMAP...")
    try:
        n_topics = len(visualizer.topics)
        # Use parameters that preserve local structure for hierarchical relationships
        visualizer.reduce_dimensions(
            method='umap',
            n_neighbors=min(15, n_topics // 2),  # Adjust for dataset size
            min_dist=0.3,  # Allow some separation between clusters
            metric='cosine',  # Good for text embeddings
            random_state=42
        )
        print("   2D coordinates computed.")
    except Exception as e:
        print(f"   Error with UMAP: {e}")
        print("   Trying t-SNE instead...")
        try:
            visualizer.reduce_dimensions(
                method='tsne', 
                perplexity=min(30, n_topics // 4),
                random_state=42
            )
            print("   t-SNE coordinates computed.")
        except Exception as e2:
            print(f"   Both UMAP and t-SNE failed: {e2}")
            return

    # 5. Create enhanced hierarchical visualization
    print("\n4. Creating enhanced hierarchical visualization...")
    try:
        fig = create_enhanced_hierarchical_visualization(visualizer)
        print("   Enhanced visualization created.")
    except Exception as e:
        print(f"   Error creating visualization: {e}")
        return

    # 6. Display and save
    print("\n5. Opening visualization...")
    try:
        fig.show()
    except Exception as e:
        print(f"   Warning: Could not open in browser: {e}")

    print("\n6. Saving visualization...")
    try:
        fig.write_html('german_math_curriculum_map.html')
        print("   Saved as: german_math_curriculum_map.html")
    except Exception as e:
        print(f"   Error saving: {e}")

    print("\n--- German Mathematics Curriculum Map Complete ---")
    return visualizer, fig


def create_enhanced_hierarchical_visualization(visualizer):
    """
    Create an enhanced visualization that highlights the hierarchical structure.
    """
    # Build hierarchy graph
    visualizer.build_hierarchy_graph()
    
    # Prepare data with hierarchy levels
    plot_data = []
    for topic_id, topic in visualizer.topics.items():
        # Determine hierarchy level
        level = get_hierarchy_level(topic_id, visualizer.topics)
        level_names = ['Subject', 'Topic', 'Subtopic', 'Concept', 'Detail']
        level_name = level_names[min(level, len(level_names)-1)]
        
        # Get main topic for coloring
        main_topic = get_main_topic(topic_id, visualizer.topics)
        
        plot_data.append({
            'topic_id': topic_id,
            'name': topic.name,
            'x': topic.x,
            'y': topic.y,
            'level': level,
            'level_name': level_name,
            'main_topic': main_topic,
            'curriculum_text': topic.curriculum_text,
            'parent_id': topic.parent_id
        })
    
    import pandas as pd
    df = pd.DataFrame(plot_data)
    
    # Create figure with custom colors for each main topic
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='main_topic',
        symbol='level_name',
        size=[get_size_by_level(level) for level in df['level']],
        hover_data=['name', 'level_name', 'curriculum_text'],
        title="German Mathematics Curriculum - Hierarchical Semantic Map",
        width=1400,
        height=900,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Add hierarchy edges
    edge_x = []
    edge_y = []
    
    for topic_id, topic in visualizer.topics.items():
        if topic.parent_id and topic.parent_id in visualizer.topics:
            parent = visualizer.topics[topic.parent_id]
            if parent.x is not None and parent.y is not None:
                edge_x.extend([parent.x, topic.x, None])
                edge_y.extend([parent.y, topic.y, None])
    
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(color='rgba(100,100,100,0.4)', width=1),
            name='Hierarchical Connections',
            showlegend=True,
            hoverinfo='none'
        ))
    
    # Customize layout
    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        xaxis=dict(title='Semantic Dimension 1', showgrid=True),
        yaxis=dict(title='Semantic Dimension 2', showgrid=True),
        font=dict(size=12),
        legend=dict(orientation="v", x=1.02, y=1),
        margin=dict(r=200)  # More space for legend
    )
    
    # Update marker sizes for better visibility
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='white'),
            opacity=0.8
        )
    )
    
    return fig


def get_hierarchy_level(topic_id, topics):
    """Get the hierarchy level of a topic (0=root, 1=child of root, etc.)"""
    level = 0
    current_id = topic_id
    
    while current_id in topics and topics[current_id].parent_id:
        current_id = topics[current_id].parent_id
        level += 1
        if level > 10:  # Prevent infinite loops
            break
    
    return level


def get_main_topic(topic_id, topics):
    """Get the main topic (level 1) that this topic belongs to"""
    current_id = topic_id
    path = [current_id]
    
    while current_id in topics and topics[current_id].parent_id:
        current_id = topics[current_id].parent_id
        path.append(current_id)
        if len(path) > 10:  # Prevent infinite loops
            break
    
    # Return the level 1 topic (child of root)
    if len(path) >= 2:
        return topics[path[-2]].name
    elif len(path) == 1:
        return topics[path[0]].name
    else:
        return "Unknown"


def get_size_by_level(level):
    """Get marker size based on hierarchy level"""
    sizes = [25, 20, 15, 12, 10]  # Larger for higher levels
    return sizes[min(level, len(sizes)-1)]


if __name__ == "__main__":
    create_german_math_map() 