# Hate Speech Detection Models

A modular, composable framework for training and evaluating hate speech detection models with support for traditional ML, deep learning, and API-based approaches.

## 🎯 Quick Start

### Composable Model System
The easiest way to create and train models using our plug-and-play architecture:

```python
from model.pipeline import create_composable_model

# Traditional approach: BOW + Logistic Regression
model = create_composable_model(
    dataset_type='berkeley',
    feature_type='bow', 
    architecture_type='logistic',
    model_name='bow_classifier',
    dataset_config={'sample_size': 10000},
    feature_config={'max_features': 5000, 'ngram_range': [1, 2]},
    architecture_config={'C': 0.5}
)

# Sophisticated approach: HuggingFace + LightGBM
model = create_composable_model(
    dataset_type='berkeley',
    feature_type='huggingface',
    architecture_type='lightgbm', 
    model_name='hf_classifier',
    dataset_config={'sample_size': 10000},
    feature_config={'model_name': 'sentence-transformers/all-MiniLM-L6-v2'},
    architecture_config={'num_leaves': 31, 'learning_rate': 0.1}
)

# Train and predict
model.train()
probabilities = model.predict(['This is a test message'])
```

## 🏗️ Architecture

### Component Types

**Dataset Loaders:**
- `berkeley` - UC Berkeley Hate Speech Dataset via HuggingFace
- `csv` - Custom CSV files with flexible column mapping
- `synthetic` - Generated synthetic data for testing
- `memory` - In-memory data for quick experiments

**Feature Extractors:**
- `bow` - Bag of Words (Count/Binary vectorization)
- `tfidf` - TF-IDF weighted features
- `huggingface` - Transformer embeddings (sentence-transformers)
- `statistical` - Hand-crafted statistical features

**Model Architectures:**
- `logistic` - Logistic Regression
- `svm` - Support Vector Machine
- `random_forest` - Random Forest
- `lightgbm` - LightGBM Gradient Boosting
- `xgboost` - XGBoost Gradient Boosting

### Model Types

#### Basic Models (`model/basic/`)
- `BagOfWordsModel` - Count Vectorizer + Logistic Regression
- Traditional sklearn-based approaches optimized for speed

#### Sophisticated Models (`model/sophisticated/`)
- `HuggingFaceLightGBMModel` - Transformer embeddings + Gradient Boosting
- Deep learning approaches with semantic understanding

#### API Models (`model/api/`)
- `ClaudeHateSpeechModel` - Anthropic Claude zero-shot classification
- OpenAI GPT-based classification (credential management ready)
- External API integrations with secure credential handling

## 📊 Dataset Integration

### Flexible Column Mapping
The system automatically detects different column names across datasets:

```python
from model.base import DataProcessor

# Auto-detects columns (text/comment/content and label/hatespeech/class)
data = DataProcessor.load_evaluation_data("your_file.csv")

# Or specify explicitly
data = DataProcessor.load_evaluation_data("file.csv", text_column="message", label_column="hate")

# Use predefined configurations
data = DataProcessor.load_dataset_with_config("English_test.csv", "english_test")
```

**Supported Dataset Configurations:**
- `berkeley`: text → 'text', labels → 'hatespeech'
- `english_test`: text → 'text', labels → 'label'

### Berkeley Hate Speech Dataset
Primary training data loaded automatically:
```python
data = DataProcessor.load_berkeley_dataset(split='train')  # or 'validation', 'test'
texts = data['texts']
labels = data['labels']  # Binary: 0=not hate, 1=hate
```

## 🚀 Training Scripts

### Quick Model Comparison
Compare traditional vs sophisticated approaches on 5k samples:
```bash
python model/scripts/train_models_for_comparison.py
```

### Comprehensive Evaluation  
Full training on 20k samples + English test evaluation:
```bash
python model/scripts/train_and_evaluate.py
```

### Model Registry & Production
```python
from model.integration import ModelRegistry

registry = ModelRegistry()
registry.create_and_register_model('bow_classifier', 'berkeley', 'bow', 'logistic')
model = registry.get_model('bow_classifier')
```

## 🔐 API Credentials Setup

Set up credentials for Claude, OpenAI, and HuggingFace:
```bash
python model/scripts/setup_credentials.py
```

Creates `.env` file with:
```bash
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  
HUGGINGFACE_TOKEN=your_token_here
```

## 🧪 Testing & Evaluation

### Test All Model Types
```bash
python model/scripts/test_all_models.py
```

### Data Exploration
```bash
python model/scripts/data_explorer.py
```

### Goblin Engine Integration
Models integrate with the Goblin execution engine for A/B testing:
```bash
python model/scripts/test_goblin_integration.py
```

## 📈 Model Comparison Results

The framework automatically generates:
- Performance metrics (accuracy, precision, recall, F1, AUC)
- ROC curves and confusion matrices
- Error analysis with false positive/negative examples
- Optimal threshold analysis
- Feature importance (where applicable)

## 🛠️ Advanced Usage

### Manual Model Configuration
```python
from model.base import ModelConfig
from model.basic import BagOfWordsModel

config = ModelConfig(
    model_name="custom_bow",
    model_type="basic", 
    config={
        'max_features': 10000,
        'ngram_range': [1, 2],
        'C': 1.0,
        'min_df': 2
    }
)

model = BagOfWordsModel(config)
model.train(texts, labels)
probabilities = model.predict(test_texts)
```

### Custom Feature Extractors
```python
from model.components import get_feature_extractor

extractor = get_feature_extractor('huggingface', {
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'max_length': 256,
    'batch_size': 16
})
```

## 📋 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `scikit-learn` - Traditional ML models
- `lightgbm, xgboost` - Gradient boosting
- `transformers, torch` - HuggingFace models  
- `datasets` - Berkeley dataset loading
- `anthropic` - Claude API
- `openai` - OpenAI API

## 🎯 Data Format

**Input/Output:**
- **Input**: List of text strings
- **Output**: Probability scores (0-1) for hate speech detection  
- **Training Labels**: Binary (0=not hate speech, 1=hate speech)

**Supported File Formats:**
- CSV files with flexible column names
- HuggingFace datasets
- In-memory Python lists

## 🔄 Integration with Goblin Engine

Models automatically integrate with the Goblin execution engine for:
- A/B testing different model configurations
- Production deployment pipelines
- Automated model comparison workflows
- Result aggregation and reporting

See `goblin_backend/` for execution engine documentation.