# Hate Speech Detection Models

A unified framework for training and evaluating hate speech detection models across different complexity levels.

## Model Types

### Basic Models (`model/basic/`)
- Bag of Words + Logistic Regression
- TF-IDF + SVM
- Traditional sklearn-based approaches

### Sophisticated Models (`model/sophisticated/`)
- HuggingFace embeddings + LightGBM
- Fine-tuned BERT/RoBERTa
- Deep learning approaches

### API Models (`model/api/`)
- Claude zero-shot classification
- OpenAI GPT-based classification
- External API integrations

## Data Format

All models expect:
- **Input**: List of text strings
- **Output**: Probability scores (0-1) for hate speech detection
- **Training Labels**: Binary (0=not hate speech, 1=hate speech)

## Dataset Integration

### Berkeley Hate Speech Dataset
Primary training data loaded via:
```python
from model.base import DataProcessor
data = DataProcessor.load_berkeley_dataset()
```

### Custom Evaluation Data
Load CSV files with `text` column:
```python
data = DataProcessor.load_evaluation_data("your_file.csv")
```

## Usage

```python
from model.base import ModelConfig
from model.basic import BagOfWordsModel

# Configure model
config = ModelConfig(
    model_name="bow_classifier",
    model_type="basic",
    config={"max_features": 10000}
)

# Train model
model = BagOfWordsModel(config)
model.train(texts, labels)

# Make predictions
probabilities = model.predict(new_texts)
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Exploration

Explore the Berkeley dataset:
```bash
python model/scripts/data_explorer.py
```