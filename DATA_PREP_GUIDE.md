# Data Preparation Guide for Topic Semantic Map Visualization

This guide provides detailed instructions for preparing CSV data for the Topic Semantic Map Visualization system. Follow these specifications to ensure your curriculum data is properly processed and visualized.

## CSV Data Structure Requirements

### Required Columns

#### 1. `topic_id` (Required)
- **Type**: String
- **Purpose**: Unique identifier for each topic
- **Format**: Use descriptive IDs like `analysis_derivatives_basic` or `math_subject`
- **Rules**: 
  - Must be unique across all topics
  - No spaces (use underscores instead)
  - Alphanumeric characters and underscores only
  - Should be descriptive and hierarchical

**Examples:**
```
math_subject
analysis_topic
analysis_derivatives_basic
linear_algebra_vectors_dot_product
```

#### 2. `name` (Required)
- **Type**: String
- **Purpose**: Human-readable display name for the topic
- **Format**: Clear, concise names that will appear in the visualization
- **Rules**: 
  - Should be descriptive but not too long
  - Appears in hover text and legends
  - Use proper capitalization

**Examples:**
```
Mathematics
Analysis
Basic Differentiation Rules
Dot Product
```

#### 3. `curriculum_text` (Required for semantic analysis)
- **Type**: String
- **Purpose**: Detailed description used for generating semantic embeddings
- **Format**: Rich, descriptive text explaining the concept
- **Rules**: 
  - More detailed text = better semantic similarity detection
  - Should include key concepts, applications, and context
  - 50-500 words recommended
  - Include learning objectives, mathematical concepts, and applications

**Example:**
```
"Derivatives as rates of change, differentiation rules like power rule, product rule, quotient rule, and chain rule for finding instantaneous rates of change. Applications include optimization problems, curve sketching, and modeling dynamic processes in physics and economics."
```

### Hierarchical Structure Columns

#### 4. `parent_id` (Optional but recommended for hierarchy)
- **Type**: String or empty
- **Purpose**: References the `topic_id` of the parent topic
- **Format**: Must match exactly an existing `topic_id`, or leave empty for root topics
- **Rules**: 
  - Creates the hierarchical tree structure
  - Empty for root/top-level topics
  - Must reference valid existing `topic_id`
  - No circular references allowed

**Example Hierarchy:**
```
topic_id: math_subject, parent_id: (empty) - ROOT
topic_id: analysis_topic, parent_id: math_subject
topic_id: analysis_derivatives, parent_id: analysis_topic
topic_id: analysis_derivatives_basic, parent_id: analysis_derivatives
```

### Classification Columns

#### 5. `country` (Optional)
- **Type**: String
- **Purpose**: Groups topics by educational system
- **Format**: Country name like "Germany", "USA", "UK"
- **Effect**: Used for color coding in visualization
- **Rules**: Keep consistent spelling and capitalization

#### 6. `school_type` (Optional)
- **Type**: String
- **Purpose**: Educational level classification
- **Format**: "Elementary", "Primary", "Gymnasium", "High School", "Middle School"
- **Effect**: Used for symbol shapes in visualization
- **Rules**: Use consistent terminology within your dataset

#### 7. `grade` (Optional)
- **Type**: String
- **Purpose**: Grade level specification
- **Format**: "Grade 12", "Year 2", "Grades 10-12", "K-5"
- **Effect**: Used for filtering and hover information
- **Rules**: Use consistent format throughout your data

### Technical Columns

#### 8. `embedding` (Optional)
- **Type**: Empty (leave blank)
- **Purpose**: Pre-computed semantic embeddings
- **Format**: Leave empty - the system computes these automatically
- **Rules**: 
  - Only provide if you have pre-computed embeddings as numpy arrays
  - Most users should leave this empty
  - System will automatically generate using sentence transformers

#### 9. `metadata` (Optional)
- **Type**: Empty or JSON string
- **Purpose**: Additional structured data
- **Format**: Valid JSON object with extra information
- **Rules**: 
  - Must be valid JSON if provided
  - Can include additional fields like prerequisites, learning objectives, etc.

**Example:**
```json
{"prerequisites": ["Basic Algebra"], "difficulty": "Advanced", "time_hours": 15}
```

## Complete CSV Example

```csv
topic_id,name,parent_id,country,school_type,grade,curriculum_text,embedding,metadata
math_subject,Mathematics,,Germany,Gymnasium,Grades 10-12,"Core mathematics curriculum for German Gymnasium covering advanced mathematical concepts",,
analysis_topic,Analysis,math_subject,Germany,Gymnasium,Grade 12,"Mathematical analysis including functions, limits, derivatives, and integrals",,
analysis_functions,Functions,analysis_topic,Germany,Gymnasium,Grade 12,"Study of mathematical functions including domain, range, and behavior analysis",,
analysis_functions_polynomial,Polynomial Functions,analysis_functions,Germany,Gymnasium,Grade 12,"Polynomial functions of various degrees, their properties and graphical representations",,
analysis_derivatives,Differentiation,analysis_topic,Germany,Gymnasium,Grade 12,"Derivatives as rates of change, differentiation rules and applications",,
analysis_derivatives_basic,Basic Differentiation Rules,analysis_derivatives,Germany,Gymnasium,Grade 12,"Power rule, product rule, quotient rule, and chain rule for differentiation",,
```

## Data Processing Steps

### 1. CSV Loading and Preprocessing
The system automatically:
- Loads CSV using pandas
- Converts NaN values to None for proper hierarchy handling
- Validates data structure and required fields

### 2. Hierarchy Processing
- **Root Topics**: Topics with `parent_id = None` or empty
- **Child Topics**: Topics with valid `parent_id` referencing existing topics
- **Validation**: Checks that all `parent_id` values reference existing `topic_id` values
- **Cycle Detection**: Prevents circular references in the hierarchy

### 3. Semantic Embedding Generation
- **Text Source**: Uses `curriculum_text` field (falls back to `name` if empty)
- **Model**: Sentence-BERT transformer model (default: 'all-MiniLM-L6-v2')
- **Output**: 384-dimensional vectors for semantic similarity computation
- **Process**: Automatically computed during visualization creation

### 4. Dimensionality Reduction
- **Method**: UMAP (preferred) or t-SNE for fallback
- **Purpose**: Convert high-dimensional embeddings to 2D coordinates
- **Parameters**: Automatically adjusted based on dataset size and structure

## Best Practices for Data Structure

### Recommended Hierarchical Organization

```
Subject (Level 0)
├── Topic (Level 1)
│   ├── Subtopic (Level 2)
│   │   ├── Concept (Level 3)
│   │   └── Specific Detail (Level 4)
│   └── Another Subtopic (Level 2)
└── Another Topic (Level 1)
```

### Real Example Hierarchy

```
Mathematics (math_subject)
├── Analysis (analysis_topic)
│   ├── Functions (analysis_functions)
│   │   ├── Polynomial Functions (analysis_functions_polynomial)
│   │   ├── Rational Functions (analysis_functions_rational)
│   │   └── Trigonometric Functions (analysis_functions_trigonometric)
│   ├── Derivatives (analysis_derivatives)
│   │   ├── Basic Rules (analysis_derivatives_basic)
│   │   ├── Applications (analysis_derivatives_applications)
│   │   └── Implicit Differentiation (analysis_derivatives_implicit)
│   └── Integration (analysis_integrals)
│       ├── Basic Techniques (analysis_integrals_basic)
│       └── Applications (analysis_integrals_applications)
├── Linear Algebra (linear_algebra_topic)
│   ├── Vectors (linear_algebra_vectors)
│   │   ├── Vector Addition (linear_algebra_vectors_addition)
│   │   ├── Dot Product (linear_algebra_vectors_dot)
│   │   └── Cross Product (linear_algebra_vectors_cross)
│   └── Matrices (linear_algebra_matrices)
│       ├── Basic Operations (linear_algebra_matrices_basic)
│       └── Determinants (linear_algebra_matrices_determinant)
```

### Curriculum Text Quality Guidelines

#### ✅ Good curriculum text examples:

**Detailed and Context-rich:**
```
"Derivatives as rates of change, differentiation rules like power rule, product rule, quotient rule, and chain rule for finding instantaneous rates of change. Applications include optimization problems, related rates, curve sketching, and modeling dynamic processes in physics and economics."
```

**Application-focused:**
```
"Vector addition using parallelogram rule and component-wise methods. Geometric interpretation in 2D and 3D space, applications in physics for force combinations and displacement calculations."
```

**Comprehensive:**
```
"Matrix multiplication operations including conditions for multiplication, properties of matrix products, and applications in solving systems of linear equations and representing linear transformations."
```

#### ❌ Poor curriculum text examples:

**Too short:**
```
"Derivatives"
"Vectors"
"Math"
```

**Too vague:**
```
"Math stuff about functions"
"Some algebra concepts"
"Advanced topics"
```

**No context:**
```
"Rules and formulas"
"Chapter 5 material"
"Test content"
```

## Data Validation Requirements

### Essential Validation Checks

1. **Unique topic_ids**: No duplicate identifiers allowed
2. **Valid parent references**: All `parent_id` values must reference existing `topic_id` values
3. **No circular references**: Topic cannot be its own ancestor
4. **Root existence**: At least one topic with no parent (empty `parent_id`)
5. **Non-empty curriculum_text**: Required for meaningful semantic analysis
6. **Consistent hierarchy**: Well-formed tree structure

### Recommended Dataset Characteristics

#### Dataset Size Guidelines
- **Minimum**: 10+ topics for basic visualization
- **Sweet Spot**: 30-100 topics for rich clustering and meaningful relationships
- **Large Scale**: 100-500 topics (good performance with proper hierarchical structure)
- **Maximum**: 500+ topics (performance may degrade, consider chunking)

#### Hierarchy Depth
- **Recommended**: 3-5 levels deep
- **Example**: Subject → Topic → Subtopic → Concept → Detail
- **Avoid**: Too shallow (1-2 levels) or too deep (6+ levels)

#### Content Distribution
- **Balanced**: Roughly equal content across main topics
- **Detailed**: Rich curriculum text for better semantic analysis
- **Consistent**: Similar level of detail across similar hierarchy levels

## Common Data Preparation Errors

### 1. Hierarchy Issues
- **Circular references**: Topic A → Topic B → Topic A
- **Orphaned topics**: `parent_id` references non-existent topic
- **Multiple roots**: Several topics with empty `parent_id` when you want single root
- **Broken chains**: Missing intermediate levels in hierarchy

### 2. Content Issues
- **Empty curriculum_text**: Results in poor semantic similarity
- **Inconsistent detail levels**: Some topics very detailed, others very brief
- **Duplicate content**: Multiple topics with identical or nearly identical text
- **Generic descriptions**: Non-specific text that doesn't differentiate topics

### 3. Formatting Issues
- **Inconsistent naming**: Mixed capitalization, abbreviations
- **Special characters**: Using characters that break CSV parsing
- **Missing required fields**: Empty `topic_id` or `name` fields
- **Inconsistent hierarchical naming**: Not following consistent ID patterns

## Output and Visualization

### Generated Files
1. **HTML Visualization**: Interactive Plotly map with full functionality
2. **Coordinates CSV**: 2D positions for each topic for external use
3. **Similarity Matrix**: Pairwise semantic similarities between all topics
4. **Enriched JSON**: Complete topic data including computed embeddings

### Visual Elements in Generated Map
- **Position**: Based on semantic similarity (closer topics = more related content)
- **Color**: Grouped by country, main topic, or custom classification
- **Symbol Shape**: Differentiated by school_type or hierarchy level
- **Size**: Scaled by hierarchy level (larger markers = higher level concepts)
- **Connections**: Lines showing parent-child hierarchical relationships
- **Hover Information**: Detailed popup with topic information
- **Interactive Features**: Zoom, pan, filter, and exploration tools

## Getting Started Checklist

- [ ] Prepare CSV with all required columns (`topic_id`, `name`, `curriculum_text`)
- [ ] Design hierarchical structure with clear parent-child relationships
- [ ] Write detailed, context-rich curriculum descriptions
- [ ] Validate data structure (unique IDs, valid parent references)
- [ ] Add classification columns for enhanced visualization (country, school_type, grade)
- [ ] Test with small dataset first (10-20 topics)
- [ ] Iterate and expand based on initial results
- [ ] Document your topic ID naming conventions for consistency

This data preparation approach ensures optimal semantic mapping results with clear hierarchical clustering and meaningful visual relationships between curriculum topics. 