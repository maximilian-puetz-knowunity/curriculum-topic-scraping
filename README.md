# Topic Semantic Map Visualization

An interactive semantic-map visualization system for educational topic taxonomies that combines hierarchical curriculum structure with semantic similarity for intelligent topic recommendation.

## Overview

This system creates interactive 2D visualizations where:
- Topics are positioned based on semantic similarity (closer = more similar)
- Hierarchical curriculum structure is overlaid with edges and colors
- Users can explore recommendations by clicking on topics
- Filtering is available by country, school type, and grade level

## Features

- **Web Interface**: Easy-to-use Streamlit web application with CSV upload functionality
- **Semantic Similarity**: Uses sentence transformers to embed topic descriptions and compute semantic similarity
- **Dimensionality Reduction**: UMAP or t-SNE to map high-dimensional embeddings to 2D coordinates
- **Interactive Visualization**: Plotly-based interactive plots with hover details and click functionality
- **Recommendation System**: Find semantically similar topics within curriculum constraints
- **Hierarchical Overlay**: Visualize curriculum structure alongside semantic relationships
- **Multi-level Filtering**: Filter by country, school type, grade, or combinations thereof
- **Data Validation**: Automatic CSV structure validation with helpful error messages
- **Download Options**: Export maps as HTML, coordinates as CSV, and similarity matrices

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start - Web Interface (Recommended)

1. **Start the web application:**
```bash
streamlit run create_map.py
```

2. **Upload your CSV file** through the web interface, or try the sample data

3. **Generate your semantic map** with a single click

The web interface provides:
- **CSV Upload**: Drag and drop or browse for your topic data
- **Data Preview**: See your data before processing
- **Validation**: Automatic checks for required columns and data integrity
- **Configuration**: Customize map title and hierarchy display
- **Sample Data**: Try with included example datasets
- **Download Options**: Export results in multiple formats

### CSV Requirements
Your CSV file must include these columns:
- `topic_id`: Unique identifier for each topic
- `name`: Human-readable display name

Optional but recommended columns:
- `curriculum_text`: Detailed description for semantic analysis
- `parent_id`: For hierarchical structure
- `country`, `school_type`, `grade`: For classification and filtering

## Quick Start - Python API

```python
from topic_semantic_map import SemanticTaxonomyVisualizer, create_mock_data

# Initialize visualizer
visualizer = SemanticTaxonomyVisualizer()

# Load data (or use mock data)
mock_data = create_mock_data()
visualizer.load_topics_from_dict(mock_data)

# Generate embeddings and compute similarity
visualizer.compute_embeddings()
visualizer.compute_similarity_matrix()

# Create 2D layout
visualizer.reduce_dimensions(method='umap')

# Create interactive visualization
fig = visualizer.create_interactive_visualization()
fig.show()

# Get recommendations for a topic
recommendations = visualizer.get_recommendations('math_basic_arithmetic')
print(recommendations)
```

For the original command-line interface, use `python create_map_cli.py`

## Data Format

Topics should be provided as a list of dictionaries with the following structure:

```python
{
    'topic_id': 'unique_identifier',
    'name': 'Display Name',
    'parent_id': 'parent_topic_id',  # Optional, for hierarchy
    'country': 'USA',               # Optional
    'school_type': 'Elementary',    # Optional
    'grade': 'Grade 3',            # Optional
    'curriculum_text': 'Detailed description for embedding',
    'embedding': np.array([...]),   # Optional, will be computed if not provided
    'metadata': {...}              # Optional additional data
}
```

## Core Classes

### TopicNode

Represents individual topics in the taxonomy:

```python
topic = TopicNode(
    topic_id='math_fractions',
    name='Fractions',
    parent_id='math_multiplication',
    country='USA',
    school_type='Elementary', 
    grade='Grade 4',
    curriculum_text='Understanding fractions, numerators and denominators'
)
```

### SemanticTaxonomyVisualizer

Main class for creating semantic maps:

#### Key Methods

**Data Loading:**
- `load_topics_from_dict(topics_data)`: Load from list of dictionaries
- `load_topics_from_csv(csv_path)`: Load from CSV file

**Embedding & Similarity:**
- `compute_embeddings()`: Generate sentence embeddings for topics
- `compute_similarity_matrix()`: Compute pairwise cosine similarity

**Dimensionality Reduction:**
- `reduce_dimensions(method='umap', **kwargs)`: Map to 2D coordinates
  - Methods: 'umap' (default) or 'tsne'
  - UMAP parameters: n_neighbors, min_dist, metric
  - t-SNE parameters: perplexity

**Visualization:**
- `create_interactive_visualization()`: Generate Plotly figure
- `highlight_recommendations(fig, topic_id)`: Add recommendation highlights

**Recommendations:**
- `get_recommendations(topic_id, n_recommendations=5, same_curriculum_path=True)`: Get similar topics

**Filtering:**
- `filter_by_criteria(country=None, school_type=None, grade=None)`: Filter topics

## Usage Examples

### Basic Visualization

```python
visualizer = SemanticTaxonomyVisualizer()
visualizer.load_topics_from_csv('topics.csv')
visualizer.compute_embeddings()
visualizer.reduce_dimensions(method='umap', n_neighbors=10, min_dist=0.1)

fig = visualizer.create_interactive_visualization(
    title="My Curriculum Map",
    show_hierarchy_edges=True
)
fig.show()
```

### Recommendation Pipeline

```python
def get_student_recommendations(student_profile):
    """
    Example recommendation pipeline for a student.
    
    Args:
        student_profile: Dict with 'country', 'school_type', 'grade', 'mastered_topics'
    """
    visualizer = SemanticTaxonomyVisualizer()
    visualizer.load_topics_from_csv('curriculum.csv')
    visualizer.compute_embeddings()
    
    # Filter topics for student's curriculum
    eligible_topics = visualizer.filter_by_criteria(
        country=student_profile['country'],
        school_type=student_profile['school_type'],
        grade=student_profile['grade']
    )
    
    # Get recommendations based on mastered topics
    all_recommendations = []
    for mastered_topic in student_profile['mastered_topics']:
        recs = visualizer.get_recommendations(mastered_topic, n_recommendations=3)
        all_recommendations.extend(recs)
    
    # Remove duplicates and already mastered topics
    unique_recs = {}
    for topic_id, score in all_recommendations:
        if (topic_id not in student_profile['mastered_topics'] and 
            topic_id in eligible_topics):
            if topic_id not in unique_recs or score > unique_recs[topic_id]:
                unique_recs[topic_id] = score
    
    # Sort by score and return top recommendations
    final_recs = sorted(unique_recs.items(), key=lambda x: x[1], reverse=True)
    return final_recs[:5]

# Example usage
student = {
    'country': 'USA',
    'school_type': 'Elementary',
    'grade': 'Grade 3',
    'mastered_topics': ['math_basic_arithmetic', 'lang_reading_comprehension']
}

recommendations = get_student_recommendations(student)
```

### Custom Embedding Model

```python
# Use a different sentence transformer model
visualizer = SemanticTaxonomyVisualizer(embedding_model='all-mpnet-base-v2')

# Or use pre-computed embeddings
topics_with_embeddings = [
    {
        'topic_id': 'custom_topic',
        'name': 'Custom Topic',
        'curriculum_text': 'Topic description',
        'embedding': np.array([0.1, 0.2, ...])  # Pre-computed embedding
    }
]
visualizer.load_topics_from_dict(topics_with_embeddings)
```

### Interactive Features

The generated visualization includes:

- **Hover Information**: Topic name, grade level, curriculum description
- **Color Coding**: Topics colored by country
- **Shape Coding**: Topics shaped by school type  
- **Hierarchy Edges**: Optional lines showing parent-child relationships
- **Recommendation Highlighting**: Click functionality to highlight similar topics

### Customization Options

```python
# Customize UMAP parameters
visualizer.reduce_dimensions(
    method='umap',
    n_neighbors=15,      # Larger = more global structure
    min_dist=0.1,        # Smaller = tighter clusters
    metric='cosine'      # Distance metric for embeddings
)

# Customize visualization
fig = visualizer.create_interactive_visualization(
    title="Custom Title",
    width=1400,
    height=900,
    show_hierarchy_edges=False
)

# Add custom styling
fig.update_layout(
    template='plotly_dark',
    font=dict(size=14, family='Arial')
)
```

## Advanced Usage

### Batch Processing Multiple Curricula

```python
def process_multiple_curricula(curricula_paths):
    """Process multiple curriculum files and create combined visualization."""
    all_topics = []
    
    for curriculum_file in curricula_paths:
        df = pd.read_csv(curriculum_file)
        topics = df.to_dict('records')
        all_topics.extend(topics)
    
    visualizer = SemanticTaxonomyVisualizer()
    visualizer.load_topics_from_dict(all_topics)
    visualizer.compute_embeddings()
    visualizer.reduce_dimensions()
    
    return visualizer
```

### Export Functionality

```python
# Export coordinates for external use
coordinates_df = visualizer.coordinates
coordinates_df.to_csv('topic_coordinates.csv', index=False)

# Export similarity matrix
np.save('similarity_matrix.npy', visualizer.similarity_matrix)

# Export visualization as HTML
fig = visualizer.create_interactive_visualization()
fig.write_html('topic_map.html')
```

## How the Similarity Matrix Works

The similarity matrix is the core component that enables semantic relationship detection between curriculum topics. Here's a detailed explanation of how it functions:

### Step-by-Step Process

#### 1. Text Embedding Generation
```python
# Each topic's curriculum_text is converted to a high-dimensional vector
visualizer.compute_embeddings()
```

**What happens:**
- The system uses Sentence-BERT (default: 'all-MiniLM-L6-v2') to convert text descriptions into 384-dimensional numerical vectors
- Each topic gets an embedding that captures its semantic meaning
- Similar concepts get similar vector representations

**Example:**
```
Topic: "Derivatives and differentiation rules"
→ Embedding: [0.123, -0.456, 0.789, ..., 0.234] (384 dimensions)

Topic: "Integration and antiderivatives" 
→ Embedding: [0.145, -0.423, 0.812, ..., 0.267] (384 dimensions)
```

#### 2. Cosine Similarity Computation
```python
# Compute pairwise similarities between all topic embeddings
similarity_matrix = visualizer.compute_similarity_matrix()
```

**Mathematical Foundation:**
The cosine similarity between two vectors A and B is calculated as:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Where:
- `A · B` is the dot product of vectors A and B
- `||A||` and `||B||` are the magnitudes (lengths) of the vectors

**Properties:**
- **Range**: -1 to +1 (but typically 0 to 1 for text embeddings)
- **1.0**: Identical/highly similar content
- **0.0**: Completely unrelated content
- **Values between 0-1**: Varying degrees of similarity

#### 3. Similarity Matrix Structure
The resulting matrix is symmetric and shows all pairwise relationships:

```
           Topic A  Topic B  Topic C  Topic D
Topic A    1.000    0.234    0.567    0.123
Topic B    0.234    1.000    0.345    0.678
Topic C    0.567    0.345    1.000    0.456
Topic D    0.123    0.678    0.456    1.000
```

**Matrix Properties:**
- **Diagonal = 1.0**: Each topic is identical to itself
- **Symmetric**: similarity(A,B) = similarity(B,A)
- **Size**: N×N where N is the number of topics

#### 4. Practical Example with Real Content

Consider these mathematics topics:

```python
topics = {
    'derivatives_basic': "Basic differentiation rules including power rule, product rule, quotient rule",
    'derivatives_applications': "Applications of derivatives for optimization and curve sketching",
    'integrals_basic': "Basic integration techniques and antiderivatives",
    'vectors_addition': "Vector addition using component-wise and geometric methods"
}
```

**Expected Similarity Scores:**
- derivatives_basic ↔ derivatives_applications: **~0.75** (both about derivatives)
- derivatives_basic ↔ integrals_basic: **~0.65** (related calculus concepts)
- derivatives_basic ↔ vectors_addition: **~0.25** (different mathematical areas)

### Applications in the System

#### 1. Topic Recommendations
```python
recommendations = visualizer.get_recommendations('derivatives_basic', n_recommendations=3)
# Returns: [('derivatives_applications', 0.75), ('integrals_basic', 0.65), ...]
```

**How it works:**
1. Find the row in the similarity matrix for the given topic
2. Sort all other topics by their similarity scores
3. Return the top N most similar topics

#### 2. 2D Visualization Positioning
```python
visualizer.reduce_dimensions(method='umap', metric='cosine')
```

**How similarity affects positioning:**
- UMAP uses the similarity matrix (via cosine metric) to preserve relationships in 2D
- Topics with high similarity scores are positioned closer together
- Topics with low similarity scores are positioned farther apart
- The 2D layout maintains the high-dimensional similarity structure as much as possible

#### 3. Clustering and Grouping
The similarity matrix enables automatic discovery of topic clusters:
- **High similarity regions**: Related concepts (e.g., all calculus topics)
- **Low similarity regions**: Distinct areas (e.g., algebra vs. statistics)
- **Hierarchical patterns**: Parent-child relationships often show high similarity

### Quality Factors

#### What Makes Good Similarity Scores:
1. **Rich curriculum text**: Detailed descriptions → better embeddings → more accurate similarities
2. **Consistent terminology**: Using similar vocabulary for related concepts
3. **Appropriate level of detail**: Balanced descriptions across all topics
4. **Clear differentiation**: Distinct topics should have sufficiently different descriptions

#### Example of Good vs Poor Text Quality:

**✅ Good (produces meaningful similarities):**
```
"Derivatives as instantaneous rates of change, including power rule for polynomial functions, 
product rule for function products, quotient rule for ratios, and chain rule for compositions"
```

**❌ Poor (produces weak similarities):**
```
"Chapter 5 material on derivatives"
```

### Advanced Similarity Analysis

#### Similarity Threshold Interpretation:
- **0.8-1.0**: Very closely related (e.g., different aspects of same concept)
- **0.6-0.8**: Moderately related (e.g., concepts within same mathematical area)
- **0.4-0.6**: Somewhat related (e.g., concepts requiring similar prerequisite knowledge)
- **0.2-0.4**: Weakly related (e.g., same subject but different areas)
- **0.0-0.2**: Unrelated (e.g., completely different mathematical domains)

#### Export and Analysis:
```python
# Save similarity matrix for external analysis
np.save('similarity_matrix.npy', visualizer.similarity_matrix)

# Load and analyze
similarity_matrix = np.load('similarity_matrix.npy')
```

The similarity matrix serves as the foundation for all semantic operations in the system, enabling intelligent topic recommendations, meaningful spatial layouts, and automatic discovery of curriculum relationships.

## Dependencies

- **numpy**: Numerical computations and array operations
- **pandas**: Data manipulation and CSV handling  
- **plotly**: Interactive visualization and web-based plots
- **networkx**: Graph structures for hierarchy representation
- **scikit-learn**: Cosine similarity computation
- **umap-learn**: UMAP dimensionality reduction
- **sentence-transformers**: Text embedding generation
- **kaleido**: Static image export for Plotly (optional)

## Performance Notes

- **Embedding Computation**: Most time-consuming step, scales with number of topics and text length
- **Dimensionality Reduction**: UMAP typically faster than t-SNE for larger datasets
- **Visualization**: Plotly handles thousands of points well, but performance degrades with 10k+ points
- **Memory Usage**: Similarity matrices scale O(n²) with number of topics

For large taxonomies (>1000 topics), consider:
- Using lighter embedding models
- Implementing hierarchical clustering before visualization
- Chunked processing for very large datasets

## License

This project is provided as-is for educational and research purposes.