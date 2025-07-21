#!/usr/bin/env python3
"""
Example usage script for the Topic Semantic Map Visualization system.
This script demonstrates how to create and use the interactive semantic map
for educational topic recommendations.
"""

from topic_semantic_map import SemanticTaxonomyVisualizer, create_mock_data
import json


def basic_example():
    """
    Basic example showing core functionality.
    """
    print("=== Basic Example: Topic Semantic Map ===\n")
    
    # Initialize the visualizer with a specific embedding model
    visualizer = SemanticTaxonomyVisualizer(embedding_model='all-MiniLM-L6-v2')
    
    # Load mock data (in practice, you'd load from CSV or your database)
    print("1. Loading topic data...")
    mock_data = create_mock_data()
    visualizer.load_topics_from_dict(mock_data)
    print(f"   Loaded {len(visualizer.topics)} topics")
    
    # Compute embeddings for semantic similarity
    print("\n2. Computing semantic embeddings...")
    visualizer.compute_embeddings()
    print("   Embeddings computed using Sentence-BERT")
    
    # Compute similarity matrix
    print("\n3. Computing similarity matrix...")
    similarity_matrix = visualizer.compute_similarity_matrix()
    print(f"   Similarity matrix shape: {similarity_matrix.shape}")
    
    # Reduce to 2D coordinates for visualization
    print("\n4. Reducing to 2D coordinates...")
    coordinates = visualizer.reduce_dimensions(
        method='umap',
        n_neighbors=5,  # Small dataset, so use fewer neighbors
        min_dist=0.3,   # Allow some spacing between topics
        random_state=42
    )
    print(f"   2D coordinates computed for {len(coordinates)} topics")
    
    # Build hierarchy graph
    print("\n5. Building hierarchy graph...")
    graph = visualizer.build_hierarchy_graph()
    print(f"   Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Create interactive visualization
    print("\n6. Creating interactive visualization...")
    fig = visualizer.create_interactive_visualization(
        title="Educational Topic Semantic Map - Example",
        show_hierarchy_edges=True,
        width=1200,
        height=800
    )
    
    # Highlight recommendations for a specific topic
    print("\n7. Adding recommendations for 'Basic Arithmetic'...")
    fig = visualizer.highlight_recommendations(
        fig, 
        'math_basic_arithmetic', 
        n_recommendations=3
    )
    
    # Display the visualization (will open in browser)
    print("\n8. Opening visualization in browser...")
    fig.show()
    
    return visualizer, fig


def recommendation_example():
    """
    Example showing how to use the recommendation system for a student.
    """
    print("\n\n=== Recommendation Example: Student Profile ===\n")
    
    # Initialize with existing data
    visualizer = SemanticTaxonomyVisualizer()
    visualizer.load_topics_from_dict(create_mock_data())
    visualizer.compute_embeddings()
    visualizer.compute_similarity_matrix()
    
    # Example student profile
    student_profile = {
        'name': 'Alice',
        'country': 'USA',
        'school_type': 'Elementary',
        'grade': 'Grade 3',
        'mastered_topics': ['math_basic_arithmetic', 'lang_reading_comprehension']
    }
    
    print(f"Student: {student_profile['name']}")
    print(f"Level: {student_profile['country']} {student_profile['school_type']}, {student_profile['grade']}")
    print(f"Mastered topics: {', '.join(student_profile['mastered_topics'])}")
    
    # Get recommendations for each mastered topic
    print(f"\nRecommendations based on mastered topics:")
    
    all_recommendations = {}
    
    for mastered_topic in student_profile['mastered_topics']:
        topic_name = visualizer.topics[mastered_topic].name
        print(f"\n  Based on '{topic_name}':")
        
        recommendations = visualizer.get_recommendations(
            mastered_topic, 
            n_recommendations=3,
            same_curriculum_path=True  # Only recommend within same curriculum
        )
        
        for i, (rec_topic_id, similarity) in enumerate(recommendations, 1):
            rec_topic = visualizer.topics[rec_topic_id]
            print(f"    {i}. {rec_topic.name} (Grade {rec_topic.grade}) - Similarity: {similarity:.3f}")
            
            # Store for final ranking
            if rec_topic_id not in all_recommendations:
                all_recommendations[rec_topic_id] = similarity
            else:
                all_recommendations[rec_topic_id] = max(all_recommendations[rec_topic_id], similarity)
    
    # Final ranked recommendations
    print(f"\n  Top overall recommendations for {student_profile['name']}:")
    sorted_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
    
    for i, (topic_id, score) in enumerate(sorted_recommendations[:5], 1):
        topic = visualizer.topics[topic_id]
        if topic_id not in student_profile['mastered_topics']:  # Don't recommend already mastered
            print(f"    {i}. {topic.name} (Grade {topic.grade}) - Score: {score:.3f}")


def filtering_example():
    """
    Example showing how to filter topics by curriculum criteria.
    """
    print("\n\n=== Filtering Example: Curriculum Navigation ===\n")
    
    visualizer = SemanticTaxonomyVisualizer()
    visualizer.load_topics_from_dict(create_mock_data())
    
    # Show all available criteria
    countries = set(topic.country for topic in visualizer.topics.values() if topic.country)
    school_types = set(topic.school_type for topic in visualizer.topics.values() if topic.school_type)
    grades = set(topic.grade for topic in visualizer.topics.values() if topic.grade)
    
    print("Available curriculum options:")
    print(f"  Countries: {', '.join(sorted(countries))}")
    print(f"  School Types: {', '.join(sorted(school_types))}")
    print(f"  Grades: {', '.join(sorted(grades))}")
    
    # Filter examples
    print(f"\nFiltering examples:")
    
    # Filter by country
    usa_topics = visualizer.filter_by_criteria(country='USA')
    print(f"\n  USA topics ({len(usa_topics)}):")
    for topic_id in usa_topics[:5]:  # Show first 5
        topic = visualizer.topics[topic_id]
        print(f"    - {topic.name} ({topic.school_type}, {topic.grade})")
    
    # Filter by school type
    elementary_topics = visualizer.filter_by_criteria(school_type='Elementary')
    print(f"\n  Elementary topics ({len(elementary_topics)}):")
    for topic_id in elementary_topics[:5]:
        topic = visualizer.topics[topic_id]
        print(f"    - {topic.name} ({topic.country}, {topic.grade})")
    
    # Combined filter
    usa_elementary_grade3 = visualizer.filter_by_criteria(
        country='USA',
        school_type='Elementary', 
        grade='Grade 3'
    )
    print(f"\n  USA Elementary Grade 3 topics ({len(usa_elementary_grade3)}):")
    for topic_id in usa_elementary_grade3:
        topic = visualizer.topics[topic_id]
        print(f"    - {topic.name}")


def save_results_example(visualizer):
    """
    Example showing how to save and export results.
    """
    print("\n\n=== Export Example: Saving Results ===\n")
    
    # Ensure we have computed coordinates
    if visualizer.coordinates is None:
        visualizer.reduce_dimensions(method='umap')
    
    # Save coordinates to CSV
    print("1. Saving 2D coordinates...")
    visualizer.coordinates.to_csv('topic_coordinates.csv', index=False)
    print("   Saved to: topic_coordinates.csv")
    
    # Save similarity matrix
    print("\n2. Saving similarity matrix...")
    if visualizer.similarity_matrix is not None:
        import numpy as np
        np.save('similarity_matrix.npy', visualizer.similarity_matrix)
        print("   Saved to: similarity_matrix.npy")
    
    # Export topic data with coordinates
    print("\n3. Exporting enriched topic data...")
    enriched_data = []
    for topic_id, topic in visualizer.topics.items():
        topic_dict = topic.to_dict()
        enriched_data.append(topic_dict)
    
    with open('enriched_topics.json', 'w') as f:
        json.dump(enriched_data, f, indent=2, default=str)  # default=str handles numpy arrays
    print("   Saved to: enriched_topics.json")
    
    # Create and save visualization
    print("\n4. Saving interactive visualization...")
    fig = visualizer.create_interactive_visualization()
    fig.write_html('topic_semantic_map.html')
    print("   Saved to: topic_semantic_map.html")


def main():
    """
    Run all examples in sequence.
    """
    print("Topic Semantic Map Visualization - Example Usage\n")
    print("=" * 60)
    
    try:
        # Run basic example
        visualizer, fig = basic_example()
        
        # Run recommendation example
        recommendation_example()
        
        # Run filtering example  
        filtering_example()
        
        # Save results
        save_results_example(visualizer)
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("\nGenerated files:")
        print("  - topic_coordinates.csv (2D coordinates)")
        print("  - similarity_matrix.npy (similarity scores)")
        print("  - enriched_topics.json (complete topic data)")
        print("  - topic_semantic_map.html (interactive visualization)")
        
        print("\nNext steps:")
        print("  1. Open 'topic_semantic_map.html' in your browser")
        print("  2. Hover over points to see topic details")
        print("  3. Look for highlighted recommendations")
        print("  4. Modify the code to use your own topic data!")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Make sure you have installed all dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()