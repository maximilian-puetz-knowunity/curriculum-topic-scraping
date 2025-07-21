import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')


class TopicNode:
    """
    Represents a single topic in the taxonomy with hierarchical and semantic information.
    """
    def __init__(self, topic_id: str, name: str, parent_id: Optional[str] = None, 
                 country: Optional[str] = None, school_type: Optional[str] = None, 
                 grade: Optional[str] = None, curriculum_text: Optional[str] = None,
                 embedding: Optional[np.ndarray] = None, metadata: Optional[Dict] = None):
        self.topic_id = topic_id
        self.name = name
        self.parent_id = parent_id
        self.country = country
        self.school_type = school_type
        self.grade = grade
        self.curriculum_text = curriculum_text or name
        self.embedding = embedding
        self.metadata = metadata or {}
        self.x = None  # 2D coordinates for visualization
        self.y = None
        
    def to_dict(self) -> Dict:
        """Convert topic to dictionary representation."""
        return {
            'topic_id': self.topic_id,
            'name': self.name,
            'parent_id': self.parent_id,
            'country': self.country,
            'school_type': self.school_type,
            'grade': self.grade,
            'curriculum_text': self.curriculum_text,
            'x': self.x,
            'y': self.y,
            'metadata': self.metadata
        }


class SemanticTaxonomyVisualizer:
    """
    Main class for creating interactive semantic maps of educational topic taxonomies.
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.topics: Dict[str, TopicNode] = {}
        self.similarity_matrix: Optional[np.ndarray] = None
        self.embedding_model = SentenceTransformer(embedding_model)
        self.graph = nx.DiGraph()
        self.coordinates: Optional[pd.DataFrame] = None
        
    def load_topics_from_dict(self, topics_data: List[Dict]) -> None:
        """
        Load topics from a list of dictionaries.
        
        Args:
            topics_data: List of topic dictionaries with required fields
        """
        for topic_data in topics_data:
            topic = TopicNode(**topic_data)
            self.topics[topic.topic_id] = topic
            
    def load_topics_from_csv(self, csv_path: str) -> None:
        """
        Load topics from a CSV file.
        
        Args:
            csv_path: Path to the CSV file containing topic data
        """
        df = pd.read_csv(csv_path)
        # Convert NaN values to None for proper handling
        df = df.where(pd.notnull(df), None)
        topics_data = df.to_dict('records')
        self.load_topics_from_dict(topics_data)
        
    def compute_embeddings(self, force_recompute: bool = False) -> None:
        """
        Compute embeddings for all topics using the specified model.
        
        Args:
            force_recompute: Whether to recompute embeddings even if they exist
        """
        texts_to_embed = []
        topic_ids = []
        
        for topic_id, topic in self.topics.items():
            if topic.embedding is None or force_recompute:
                texts_to_embed.append(topic.curriculum_text)
                topic_ids.append(topic_id)
        
        if texts_to_embed:
            embeddings = self.embedding_model.encode(texts_to_embed)
            for i, topic_id in enumerate(topic_ids):
                self.topics[topic_id].embedding = embeddings[i]
                
    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix between all topics.
        
        Returns:
            Similarity matrix as numpy array
        """
        embeddings = np.array([topic.embedding for topic in self.topics.values()])
        self.similarity_matrix = cosine_similarity(embeddings)
        return self.similarity_matrix
        
    def reduce_dimensions(self, method: str = 'umap', **kwargs) -> pd.DataFrame:
        """
        Reduce embeddings to 2D coordinates for visualization.
        
        Args:
            method: Dimensionality reduction method ('umap' or 'tsne')
            **kwargs: Additional parameters for the reduction method
            
        Returns:
            DataFrame with topic_id, x, y coordinates
        """
        embeddings = np.array([topic.embedding for topic in self.topics.values()])
        topic_ids = list(self.topics.keys())
        
        if method.lower() == 'umap':
            reducer = umap.UMAP(
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1),
                metric=kwargs.get('metric', 'cosine'),
                random_state=kwargs.get('random_state', 42)
            )
        elif method.lower() == 'tsne':
            reducer = TSNE(
                n_components=2,
                perplexity=kwargs.get('perplexity', 30),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        coordinates_2d = reducer.fit_transform(embeddings)
        
        self.coordinates = pd.DataFrame({
            'topic_id': topic_ids,
            'x': coordinates_2d[:, 0],
            'y': coordinates_2d[:, 1]
        })
        
        # Update topic objects with coordinates
        for _, row in self.coordinates.iterrows():
            topic = self.topics[row['topic_id']]
            topic.x = row['x']
            topic.y = row['y']
            
        return self.coordinates
        
    def build_hierarchy_graph(self) -> nx.DiGraph:
        """
        Build NetworkX graph representing the hierarchical structure.
        
        Returns:
            NetworkX DiGraph with hierarchical edges
        """
        self.graph = nx.DiGraph()
        
        for topic_id, topic in self.topics.items():
            self.graph.add_node(topic_id, **topic.to_dict())
            
            if topic.parent_id and topic.parent_id in self.topics:
                self.graph.add_edge(topic.parent_id, topic_id)
                
        return self.graph
        
    def get_recommendations(self, topic_id: str, n_recommendations: int = 5,
                          same_curriculum_path: bool = True) -> List[Tuple[str, float]]:
        """
        Get recommended topics based on semantic similarity.
        
        Args:
            topic_id: ID of the current topic
            n_recommendations: Number of recommendations to return
            same_curriculum_path: Whether to filter by same country/school_type/grade
            
        Returns:
            List of (topic_id, similarity_score) tuples
        """
        if topic_id not in self.topics:
            return []
            
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
            
        topic_idx = list(self.topics.keys()).index(topic_id)
        similarities = self.similarity_matrix[topic_idx]
        
        current_topic = self.topics[topic_id]
        candidates = []
        
        for i, (other_id, other_topic) in enumerate(self.topics.items()):
            if other_id == topic_id:
                continue
                
            # Filter by curriculum path if requested
            if same_curriculum_path:
                if (other_topic.country != current_topic.country or
                    other_topic.school_type != current_topic.school_type or
                    other_topic.grade != current_topic.grade):
                    continue
                    
            candidates.append((other_id, similarities[i]))
            
        # Sort by similarity and return top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n_recommendations]
        
    def create_interactive_visualization(self, 
                                       title: str = "Topic Semantic Map",
                                       width: int = 1200, 
                                       height: int = 800,
                                       show_hierarchy_edges: bool = True) -> go.Figure:
        """
        Create interactive Plotly visualization of the topic map.
        
        Args:
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels
            show_hierarchy_edges: Whether to show hierarchical connections
            
        Returns:
            Plotly Figure object
        """
        if self.coordinates is None:
            raise ValueError("Must call reduce_dimensions() first")
            
        # Prepare data for plotting
        plot_data = []
        for topic_id, topic in self.topics.items():
            plot_data.append({
                'topic_id': topic_id,
                'name': topic.name,
                'x': topic.x,
                'y': topic.y,
                'country': topic.country or 'Unknown',
                'school_type': topic.school_type or 'Unknown',
                'grade': topic.grade or 'Unknown',
                'curriculum_text': topic.curriculum_text,
                'parent_id': topic.parent_id
            })
            
        df = pd.DataFrame(plot_data)
        
        # Create the main scatter plot
        fig = px.scatter(
            df, 
            x='x', 
            y='y',
            color='country',
            symbol='school_type',
            size_max=15,
            hover_data=['name', 'grade', 'curriculum_text'],
            title=title,
            width=width,
            height=height
        )
        
        # Add hierarchy edges if requested
        if show_hierarchy_edges:
            edge_x = []
            edge_y = []
            
            for topic_id, topic in self.topics.items():
                if topic.parent_id and topic.parent_id in self.topics:
                    parent = self.topics[topic.parent_id]
                    if parent.x is not None and parent.y is not None:
                        edge_x.extend([parent.x, topic.x, None])
                        edge_y.extend([parent.y, topic.y, None])
            
            if edge_x:
                fig.add_trace(go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.3)', width=1),
                    name='Hierarchy',
                    showlegend=True,
                    hoverinfo='none'
                ))
        
        # Update layout
        fig.update_layout(
            showlegend=True,
            hovermode='closest',
            xaxis=dict(title='Semantic Dimension 1'),
            yaxis=dict(title='Semantic Dimension 2'),
            font=dict(size=12)
        )
        
        return fig
        
    def highlight_recommendations(self, fig: go.Figure, topic_id: str, 
                                n_recommendations: int = 5) -> go.Figure:
        """
        Add recommendation highlights to an existing figure.
        
        Args:
            fig: Existing Plotly figure
            topic_id: Topic to get recommendations for
            n_recommendations: Number of recommendations to highlight
            
        Returns:
            Updated Plotly figure
        """
        recommendations = self.get_recommendations(topic_id, n_recommendations)
        
        if not recommendations:
            return fig
            
        # Get coordinates for recommended topics
        rec_data = []
        for rec_id, similarity in recommendations:
            topic = self.topics[rec_id]
            rec_data.append({
                'x': topic.x,
                'y': topic.y,
                'name': topic.name,
                'similarity': similarity
            })
            
        rec_df = pd.DataFrame(rec_data)
        
        # Add highlighted points for recommendations
        fig.add_trace(go.Scatter(
            x=rec_df['x'],
            y=rec_df['y'],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(width=2, color='darkred')
            ),
            name='Recommendations',
            text=rec_df['name'],
            hovertemplate='<b>Recommended:</b> %{text}<br>' +
                         '<b>Similarity:</b> %{customdata:.3f}<extra></extra>',
            customdata=rec_df['similarity']
        ))
        
        return fig
        
    def filter_by_criteria(self, country: Optional[str] = None,
                          school_type: Optional[str] = None,
                          grade: Optional[str] = None) -> List[str]:
        """
        Filter topics by hierarchical criteria.
        
        Args:
            country: Country to filter by
            school_type: School type to filter by  
            grade: Grade to filter by
            
        Returns:
            List of topic IDs matching the criteria
        """
        filtered_topics = []
        
        for topic_id, topic in self.topics.items():
            if country and topic.country != country:
                continue
            if school_type and topic.school_type != school_type:
                continue
            if grade and topic.grade != grade:
                continue
                
            filtered_topics.append(topic_id)
            
        return filtered_topics


def create_mock_data() -> List[Dict]:
    """
    Create mock taxonomy data for testing purposes.
    
    Returns:
        List of topic dictionaries
    """
    mock_topics = [
        # Mathematics topics
        {
            'topic_id': 'math_basic_arithmetic',
            'name': 'Basic Arithmetic',
            'parent_id': None,
            'country': 'USA',
            'school_type': 'Elementary',
            'grade': 'Grade 1',
            'curriculum_text': 'Addition and subtraction of single digit numbers'
        },
        {
            'topic_id': 'math_multiplication',
            'name': 'Multiplication',
            'parent_id': 'math_basic_arithmetic',
            'country': 'USA',
            'school_type': 'Elementary',
            'grade': 'Grade 3',
            'curriculum_text': 'Multiplication tables and basic multiplication concepts'
        },
        {
            'topic_id': 'math_fractions',
            'name': 'Fractions',
            'parent_id': 'math_multiplication',
            'country': 'USA',
            'school_type': 'Elementary',
            'grade': 'Grade 4',
            'curriculum_text': 'Understanding fractions, numerators, denominators, and basic operations'
        },
        {
            'topic_id': 'math_algebra_basics',
            'name': 'Basic Algebra',
            'parent_id': 'math_fractions',
            'country': 'USA',
            'school_type': 'Middle School',
            'grade': 'Grade 7',
            'curriculum_text': 'Variables, simple equations, and algebraic expressions'
        },
        
        # Science topics
        {
            'topic_id': 'science_plants',
            'name': 'Plant Biology',
            'parent_id': None,
            'country': 'USA',
            'school_type': 'Elementary',
            'grade': 'Grade 2',
            'curriculum_text': 'Parts of plants, photosynthesis, and plant life cycles'
        },
        {
            'topic_id': 'science_animals',
            'name': 'Animal Biology',
            'parent_id': None,
            'country': 'USA',
            'school_type': 'Elementary', 
            'grade': 'Grade 2',
            'curriculum_text': 'Animal habitats, food chains, and animal characteristics'
        },
        {
            'topic_id': 'science_ecosystems',
            'name': 'Ecosystems',
            'parent_id': 'science_animals',
            'country': 'USA',
            'school_type': 'Elementary',
            'grade': 'Grade 5',
            'curriculum_text': 'Interactions between organisms and their environment'
        },
        {
            'topic_id': 'science_chemistry_basics',
            'name': 'Basic Chemistry',
            'parent_id': None,
            'country': 'USA',
            'school_type': 'Middle School',
            'grade': 'Grade 8',
            'curriculum_text': 'Atoms, molecules, chemical reactions, and periodic table'
        },
        
        # Language Arts topics
        {
            'topic_id': 'lang_reading_comprehension',
            'name': 'Reading Comprehension',
            'parent_id': None,
            'country': 'USA',
            'school_type': 'Elementary',
            'grade': 'Grade 1',
            'curriculum_text': 'Understanding written text and extracting meaning'
        },
        {
            'topic_id': 'lang_creative_writing',
            'name': 'Creative Writing',
            'parent_id': 'lang_reading_comprehension',
            'country': 'USA',
            'school_type': 'Elementary',
            'grade': 'Grade 3',
            'curriculum_text': 'Writing stories, poems, and creative expression through text'
        },
        {
            'topic_id': 'lang_grammar',
            'name': 'Grammar and Syntax',
            'parent_id': 'lang_creative_writing',
            'country': 'USA',
            'school_type': 'Elementary',
            'grade': 'Grade 4',
            'curriculum_text': 'Parts of speech, sentence structure, and proper grammar usage'
        },
        
        # UK Curriculum examples
        {
            'topic_id': 'uk_maths_number',
            'name': 'Number and Place Value',
            'parent_id': None,
            'country': 'UK',
            'school_type': 'Primary',
            'grade': 'Year 2',
            'curriculum_text': 'Understanding numbers up to 100, place value, and counting'
        },
        {
            'topic_id': 'uk_science_materials',
            'name': 'Materials and Properties',
            'parent_id': None,
            'country': 'UK',
            'school_type': 'Primary',
            'grade': 'Year 2',
            'curriculum_text': 'Properties of everyday materials and how they can be changed'
        }
    ]
    
    return mock_topics


def run_example_pipeline():
    """
    Example pipeline demonstrating how to use the SemanticTaxonomyVisualizer.
    """
    # Initialize the visualizer
    visualizer = SemanticTaxonomyVisualizer()
    
    # Load mock data
    mock_data = create_mock_data()
    visualizer.load_topics_from_dict(mock_data)
    
    # Compute embeddings and similarity
    visualizer.compute_embeddings()
    visualizer.compute_similarity_matrix()
    
    # Reduce to 2D coordinates
    visualizer.reduce_dimensions(method='umap', n_neighbors=5, min_dist=0.3)
    
    # Build hierarchy graph
    visualizer.build_hierarchy_graph()
    
    # Create base visualization
    fig = visualizer.create_interactive_visualization(
        title="Educational Topic Semantic Map - Demo",
        show_hierarchy_edges=True
    )
    
    # Example: Highlight recommendations for basic arithmetic
    fig = visualizer.highlight_recommendations(fig, 'math_basic_arithmetic', n_recommendations=3)
    
    # Show the plot
    fig.show()
    
    # Example recommendation query
    recommendations = visualizer.get_recommendations('math_basic_arithmetic', n_recommendations=5)
    print(f"\nRecommendations for 'Basic Arithmetic':")
    for topic_id, similarity in recommendations:
        topic_name = visualizer.topics[topic_id].name
        print(f"  {topic_name} (similarity: {similarity:.3f})")
    
    return visualizer, fig


if __name__ == "__main__":
    run_example_pipeline()