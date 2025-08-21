# NER Testing and Fine-tuning for Organization Extraction
# This notebook tests different NER approaches and provides fine-tuning capabilities

# %% [markdown]
"""
# NER Testing and Fine-tuning Notebook

This notebook provides:
1. Testing different NER models (spaCy, Transformers, etc.)
2. Evaluation metrics for organization extraction
3. Fine-tuning pipelines for custom models
4. Comparison of model performance
5. Data preparation utilities

## Installation Requirements
```bash
pip install spacy transformers datasets torch evaluate seqeval
pip install spacy-transformers
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf
```
"""

# %% [markdown]
## 1. Setup and Imports

# %%
import spacy
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Transformers and datasets
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    pipeline
)
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader

# Evaluation
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

# spaCy training
from spacy.training.example import Example
from spacy.util import minibatch, compounding

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Setup complete!")

# %% [markdown]
## 2. Sample Data Creation and Loading

# %%
def create_sample_data():
    """Create sample data for testing and training"""
    
    # Sample texts with organizations
    sample_texts = [
        "Microsoft Corporation announced a partnership with OpenAI to integrate AI capabilities.",
        "Apple Inc. reported record quarterly earnings, outperforming Google LLC and Meta Platforms Inc.",
        "JPMorgan Chase & Co. is the largest bank in the United States, followed by Bank of America Corp.",
        "Tesla Inc. and General Motors Company are competing in the electric vehicle market.",
        "Amazon.com Inc. acquired Whole Foods Market for $13.7 billion.",
        "Salesforce Inc. is a leading cloud computing company based in San Francisco.",
        "Oracle Corporation provides database software and technology solutions.",
        "Netflix Inc. competes with Disney+ and HBO Max in the streaming market.",
        "IBM Corporation has been a technology leader for over a century.",
        "NVIDIA Corporation is known for its graphics processing units and AI chips.",
        "Cisco Systems Inc. provides networking hardware and software solutions.",
        "Intel Corporation manufactures semiconductors and microprocessors.",
        "Adobe Inc. develops creative software including Photoshop and Illustrator.",
        "PayPal Holdings Inc. facilitates online payments for millions of users.",
        "Goldman Sachs Group Inc. is a leading investment banking firm.",
        "Morgan Stanley provides wealth management and investment services.",
        "Wells Fargo & Company is one of the largest banks in the United States.",
        "Berkshire Hathaway Inc. is Warren Buffett's investment company.",
        "Johnson & Johnson develops pharmaceuticals and medical devices.",
        "Pfizer Inc. is a global pharmaceutical corporation.",
        "The Federal Reserve announced changes to interest rates affecting all major banks.",
        "Stanford University researchers collaborated with MIT on artificial intelligence.",
        "Harvard Business School published a study on corporate governance.",
        "The New York Stock Exchange saw heavy trading in technology stocks.",
        "BlackRock Inc. is the world's largest asset management firm."
    ]
    
    return sample_texts

def create_training_data():
    """Create training data in spaCy format"""
    
    training_data = [
        ("Microsoft Corporation announced a partnership with OpenAI.", 
         {"entities": [(0, 19, "ORG"), (48, 54, "ORG")]}),
        
        ("Apple Inc. reported record quarterly earnings, outperforming Google LLC.", 
         {"entities": [(0, 10, "ORG"), (59, 69, "ORG")]}),
        
        ("JPMorgan Chase & Co. is the largest bank in the United States.", 
         {"entities": [(0, 20, "ORG")]}),
        
        ("Tesla Inc. and General Motors Company are competing in the market.", 
         {"entities": [(0, 10, "ORG"), (15, 40, "ORG")]}),
        
        ("Amazon.com Inc. acquired Whole Foods Market for $13.7 billion.", 
         {"entities": [(0, 15, "ORG"), (25, 42, "ORG")]}),
        
        ("Salesforce Inc. is a leading cloud computing company.", 
         {"entities": [(0, 15, "ORG")]}),
        
        ("Oracle Corporation provides database software solutions.", 
         {"entities": [(0, 18, "ORG")]}),
        
        ("Netflix Inc. competes with Disney+ and HBO Max in streaming.", 
         {"entities": [(0, 12, "ORG"), (27, 34, "ORG"), (39, 46, "ORG")]}),
        
        ("IBM Corporation has been a technology leader for decades.", 
         {"entities": [(0, 15, "ORG")]}),
        
        ("NVIDIA Corporation is known for its graphics processing units.", 
         {"entities": [(0, 18, "ORG")]}),
    ]
    
    return training_data

# Create sample data
sample_texts = create_sample_data()
training_data = create_training_data()

print(f"Created {len(sample_texts)} sample texts")
print(f"Created {len(training_data)} training examples")

# %% [markdown]
## 3. Load and Test Different NER Models

# %%
class NERModelTester:
    """Class to test different NER models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def load_spacy_models(self):
        """Load available spaCy models"""
        spacy_models = {
            'en_core_web_sm': 'en_core_web_sm',
            'en_core_web_md': 'en_core_web_md', 
            'en_core_web_lg': 'en_core_web_lg',
            'en_core_web_trf': 'en_core_web_trf'
        }
        
        for name, model_name in spacy_models.items():
            try:
                self.models[name] = spacy.load(model_name)
                print(f"Loaded {name}")
            except OSError:
                print(f"Model {name} not available. Install with: python -m spacy download {model_name}")
    
    def load_transformer_models(self):
        """Load transformer-based NER models"""
        transformer_models = {
            'dbmdz-bert': 'dbmdz/bert-large-cased-finetuned-conll03-english',
            'dslim-bert': 'dslim/bert-base-NER',
            'microsoft-deberta': 'microsoft/deberta-v3-base'
        }
        
        for name, model_name in transformer_models.items():
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForTokenClassification.from_pretrained(model_name)
                self.models[name] = pipeline("ner", 
                                            model=model, 
                                            tokenizer=tokenizer, 
                                            aggregation_strategy="simple")
                print(f"Loaded {name}")
            except Exception as e:
                print(f"Failed to load {name}: {str(e)}")
    
    def extract_organizations_spacy(self, text, model):
        """Extract organizations using spaCy model"""
        doc = model(text)
        organizations = []
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                organizations.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0  # spaCy doesn't provide confidence scores
                })
        
        return organizations
    
    def extract_organizations_transformer(self, text, model):
        """Extract organizations using transformer model"""
        results = model(text)
        organizations = []
        
        for result in results:
            if 'ORG' in result['entity_group'] or 'ORGANIZATION' in result['entity_group']:
                organizations.append({
                    'text': result['word'],
                    'start': result['start'],
                    'end': result['end'],
                    'confidence': result['score']
                })
        
        return organizations
    
    def test_model_on_texts(self, model_name, texts):
        """Test a specific model on sample texts"""
        model = self.models[model_name]
        results = []
        
        for text in texts:
            if 'spacy' in model_name or isinstance(model, spacy.lang.en.English):
                orgs = self.extract_organizations_spacy(text, model)
            else:
                orgs = self.extract_organizations_transformer(text, model)
            
            results.append({
                'text': text,
                'organizations': orgs,
                'org_count': len(orgs)
            })
        
        return results
    
    def run_comprehensive_test(self, texts):
        """Run tests on all available models"""
        print("Running comprehensive NER tests...\n")
        
        for model_name in self.models.keys():
            print(f"Testing {model_name}...")
            try:
                results = self.test_model_on_texts(model_name, texts[:5])  # Test on first 5 texts
                self.results[model_name] = results
                
                # Print summary
                total_orgs = sum([r['org_count'] for r in results])
                print(f"  Total organizations found: {total_orgs}")
                
                # Show examples
                for i, result in enumerate(results[:2]):
                    print(f"  Text {i+1}: {result['text'][:50]}...")
                    for org in result['organizations']:
                        print(f"    - {org['text']} (confidence: {org['confidence']:.3f})")
                
                print()
                
            except Exception as e:
                print(f"  Error testing {model_name}: {str(e)}\n")

# Initialize and test models
tester = NERModelTester()
tester.load_spacy_models()
tester.load_transformer_models()

# Run tests
tester.run_comprehensive_test(sample_texts)

# %% [markdown]
## 4. Evaluation Metrics and Comparison

# %%
def create_evaluation_dataset():
    """Create a gold standard dataset for evaluation"""
    
    # Gold standard annotations (text, start, end, label)
    gold_data = [
        {
            'text': "Microsoft Corporation announced a partnership with OpenAI to integrate AI capabilities.",
            'entities': [
                {'start': 0, 'end': 19, 'label': 'ORG', 'text': 'Microsoft Corporation'},
                {'start': 48, 'end': 54, 'label': 'ORG', 'text': 'OpenAI'}
            ]
        },
        {
            'text': "Apple Inc. reported record quarterly earnings, outperforming Google LLC and Meta Platforms Inc.",
            'entities': [
                {'start': 0, 'end': 10, 'label': 'ORG', 'text': 'Apple Inc.'},
                {'start': 62, 'end': 72, 'label': 'ORG', 'text': 'Google LLC'},
                {'start': 77, 'end': 96, 'label': 'ORG', 'text': 'Meta Platforms Inc.'}
            ]
        },
        {
            'text': "JPMorgan Chase & Co. is the largest bank in the United States.",
            'entities': [
                {'start': 0, 'end': 20, 'label': 'ORG', 'text': 'JPMorgan Chase & Co.'}
            ]
        },
        {
            'text': "Tesla Inc. and General Motors Company are competing in the electric vehicle market.",
            'entities': [
                {'start': 0, 'end': 10, 'label': 'ORG', 'text': 'Tesla Inc.'},
                {'start': 15, 'end': 40, 'label': 'ORG', 'text': 'General Motors Company'}
            ]
        },
        {
            'text': "Amazon.com Inc. acquired Whole Foods Market for $13.7 billion.",
            'entities': [
                {'start': 0, 'end': 15, 'label': 'ORG', 'text': 'Amazon.com Inc.'},
                {'start': 25, 'end': 42, 'label': 'ORG', 'text': 'Whole Foods Market'}
            ]
        }
    ]
    
    return gold_data

def evaluate_model_performance(model_results, gold_data):
    """Evaluate model performance against gold standard"""
    
    evaluation_results = {}
    
    for model_name, results in model_results.items():
        if not results:
            continue
            
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for i, result in enumerate(results):
            if i >= len(gold_data):
                break
                
            predicted_orgs = set([(org['start'], org['end'], org['text']) for org in result['organizations']])
            gold_orgs = set([(ent['start'], ent['end'], ent['text']) for ent in gold_data[i]['entities']])
            
            # Calculate metrics
            tp = len(predicted_orgs.intersection(gold_orgs))
            fp = len(predicted_orgs - gold_orgs)
            fn = len(gold_orgs - predicted_orgs)
            
            true_positives += tp
            false_positives += fp
            false_negatives += fn
        
        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        evaluation_results[model_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    return evaluation_results

# Create evaluation dataset and run evaluation
gold_data = create_evaluation_dataset()
evaluation_results = evaluate_model_performance(tester.results, gold_data)

# Display results
print("Model Performance Evaluation")
print("=" * 50)

results_df = pd.DataFrame(evaluation_results).T
print(results_df.round(3))

# Visualize results
if len(results_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['precision', 'recall', 'f1']
    for i, metric in enumerate(metrics):
        if metric in results_df.columns:
            results_df[metric].plot(kind='bar', ax=axes[i], title=f'{metric.capitalize()} by Model')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
## 5. Fine-tuning spaCy Model

# %%
def prepare_spacy_training_data():
    """Prepare more comprehensive training data for spaCy"""
    
    extended_training_data = [
        ("Microsoft Corporation is a technology giant based in Redmond.", 
         {"entities": [(0, 19, "ORG")]}),
        
        ("Apple Inc. and Google LLC are major competitors in the tech industry.", 
         {"entities": [(0, 10, "ORG"), (15, 25, "ORG")]}),
        
        ("The partnership between Amazon.com Inc. and Whole Foods Market was announced yesterday.", 
         {"entities": [(24, 39, "ORG"), (44, 61, "ORG")]}),
        
        ("Tesla Inc., SpaceX, and Neuralink are all companies founded by Elon Musk.", 
         {"entities": [(0, 10, "ORG"), (12, 18, "ORG"), (24, 33, "ORG")]}),
        
        ("JPMorgan Chase & Co. reported strong quarterly earnings.", 
         {"entities": [(0, 20, "ORG")]}),
        
        ("Meta Platforms Inc. (formerly Facebook Inc.) focuses on virtual reality.", 
         {"entities": [(0, 19, "ORG"), (30, 44, "ORG")]}),
        
        ("Netflix Inc. competes with Disney+, HBO Max, and Amazon Prime Video.", 
         {"entities": [(0, 12, "ORG"), (27, 34, "ORG"), (36, 43, "ORG"), (49, 68, "ORG")]}),
        
        ("Oracle Corporation and SAP SE are enterprise software companies.", 
         {"entities": [(0, 18, "ORG"), (23, 29, "ORG")]}),
        
        ("IBM Corporation has been acquired by several smaller tech startups.", 
         {"entities": [(0, 15, "ORG")]}),
        
        ("NVIDIA Corporation and Advanced Micro Devices compete in the GPU market.", 
         {"entities": [(0, 18, "ORG"), (23, 46, "ORG")]}),
        
        ("Salesforce Inc. acquired Slack Technologies for $27.7 billion.", 
         {"entities": [(0, 15, "ORG"), (25, 42, "ORG")]}),
        
        ("PayPal Holdings Inc. spun off from eBay Inc. in 2015.", 
         {"entities": [(0, 20, "ORG"), (35, 44, "ORG")]}),
        
        ("Goldman Sachs Group Inc. and Morgan Stanley are investment banks.", 
         {"entities": [(0, 24, "ORG"), (29, 43, "ORG")]}),
        
        ("Bank of America Corp. and Wells Fargo & Company serve millions of customers.", 
         {"entities": [(0, 21, "ORG"), (26, 48, "ORG")]}),
        
        ("Berkshire Hathaway Inc. is led by Warren Buffett.", 
         {"entities": [(0, 23, "ORG")]}),
    ]
    
    return extended_training_data

def fine_tune_spacy_model():
    """Fine-tune a spaCy model for organization extraction"""
    
    print("Fine-tuning spaCy model...")
    
    # Load base model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Please install en_core_web_sm: python -m spacy download en_core_web_sm")
        return None
    
    # Get training data
    training_data = prepare_spacy_training_data()
    
    # Create training examples
    examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    
    # Get the NER component
    ner = nlp.get_pipe("ner")
    
    # Add the ORG label if it's not already there
    ner.add_label("ORG")
    
    # Disable other components during training
    pipe_exceptions = ["ner", "trf_wordpiecizer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    # Training loop
    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = nlp.resume_training()
        
        for iteration in range(30):  # Number of training iterations
            losses = {}
            
            # Shuffle the training data
            examples_shuffled = list(examples)
            np.random.shuffle(examples_shuffled)
            
            # Create batches
            batches = minibatch(examples_shuffled, size=compounding(4.0, 32.0, 1.001))
            
            for batch in batches:
                nlp.update(batch, drop=0.2, losses=losses, sgd=optimizer)
            
            if iteration % 5 == 0:
                print(f"Iteration {iteration}, Losses: {losses}")
    
    print("Fine-tuning complete!")
    return nlp

# Fine-tune the model
fine_tuned_model = fine_tune_spacy_model()

# Test the fine-tuned model
if fine_tuned_model:
    print("\nTesting fine-tuned model:")
    test_texts = [
        "Microsoft Corporation and Apple Inc. are leading technology companies.",
        "The merger between Disney+ and HBO Max was discussed by executives.",
        "JPMorgan Chase & Co. invested heavily in Goldman Sachs Group Inc."
    ]
    
    for text in test_texts:
        doc = fine_tuned_model(text)
        print(f"\nText: {text}")
        print("Organizations found:")
        for ent in doc.ents:
            if ent.label_ == "ORG":
                print(f"  - {ent.text} ({ent.start_char}-{ent.end_char})")

# %% [markdown]
## 6. Fine-tuning Transformer Models

# %%
def prepare_transformer_training_data():
    """Prepare training data in transformer format (IOB tagging)"""
    
    def create_iob_tags(text, entities):
        """Convert entity annotations to IOB tags"""
        tokens = text.split()
        tags = ['O'] * len(tokens)
        
        # Simple tokenization - in practice, use proper tokenizer
        char_to_token = {}
        current_pos = 0
        
        for i, token in enumerate(tokens):
            start_pos = text.find(token, current_pos)
            end_pos = start_pos + len(token)
            
            for char_pos in range(start_pos, end_pos):
                char_to_token[char_pos] = i
            
            current_pos = end_pos
        
        # Apply entity tags
        for entity in entities:
            start_char, end_char, label = entity
            
            start_token = char_to_token.get(start_char)
            end_token = char_to_token.get(end_char - 1)
            
            if start_token is not None and end_token is not None:
                tags[start_token] = f'B-{label}'
                for token_idx in range(start_token + 1, end_token + 1):
                    if token_idx < len(tags):
                        tags[token_idx] = f'I-{label}'
        
        return tokens, tags
    
    # Convert spaCy training data to IOB format
    iob_data = []
    training_data = prepare_spacy_training_data()
    
    for text, annotations in training_data:
        entities = [(ent[0], ent[1], ent[2]) for ent in annotations['entities']]
        tokens, tags = create_iob_tags(text, entities)
        
        iob_data.append({
            'tokens': tokens,
            'ner_tags': tags
        })
    
    return iob_data

def create_transformer_dataset():
    """Create dataset for transformer training"""
    
    # Get IOB formatted data
    iob_data = prepare_transformer_training_data()
    
    # Create label mappings
    all_tags = set()
    for example in iob_data:
        all_tags.update(example['ner_tags'])
    
    tag_to_id = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    id_to_tag = {idx: tag for tag, idx in tag_to_id.items()}
    
    print(f"Tags: {list(tag_to_id.keys())}")
    
    # Convert to dataset format
    dataset_dict = {
        'tokens': [example['tokens'] for example in iob_data],
        'ner_tags': [[tag_to_id[tag] for tag in example['ner_tags']] for example in iob_data]
    }
    
    # Split into train/validation
    split_idx = int(0.8 * len(dataset_dict['tokens']))
    
    train_dataset = Dataset.from_dict({
        'tokens': dataset_dict['tokens'][:split_idx],
        'ner_tags': dataset_dict['ner_tags'][:split_idx]
    })
    
    val_dataset = Dataset.from_dict({
        'tokens': dataset_dict['tokens'][split_idx:],
        'ner_tags': dataset_dict['ner_tags'][split_idx:]
    })
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    }), tag_to_id, id_to_tag

def tokenize_and_align_labels(examples, tokenizer, tag_to_id):
    """Tokenize and align labels for transformer training"""
    
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def fine_tune_transformer_model():
    """Fine-tune a transformer model for NER"""
    
    print("Preparing transformer fine-tuning...")
    
    # Create dataset
    datasets, tag_to_id, id_to_tag = create_transformer_dataset()
    
    if len(datasets['train']) < 5:
        print("Not enough training data for transformer fine-tuning. Need more examples.")
        return None
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(tag_to_id),
        id2label=id_to_tag,
        label2id=tag_to_id
    )
    
    # Tokenize datasets
    tokenized_datasets = datasets.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, tag_to_id),
        batched=True
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Metric computation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [id_to_tag[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id_to_tag[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Starting transformer training...")
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model("./fine_tuned_ner_model")
    
    print("Transformer fine-tuning complete!")
    
    return trainer, tokenizer, id_to_tag

# Fine-tune transformer model (commented out by default due to computational requirements)
# transformer_trainer, transformer_tokenizer, transformer_id_to_tag = fine_tune_transformer_model()

print("Transformer fine-tuning code ready (uncomment to run)")

# %% [markdown]
## 7. Model Comparison and Recommendations

# %%
def create_model_comparison_report():
    """Create a comprehensive comparison report"""
    
    print("=" * 60)
    print("MODEL COMPARISON REPORT")
    print("=" * 60)
    
    # Performance summary
    if evaluation_results:
        print("\n1. PERFORMANCE METRICS")
        print("-" * 30)
        
        performance_df = pd.DataFrame(evaluation_results).T
        performance_df = performance_df.round(3)
        print(performance_df)
        
        # Best performing model
        if 'f1' in performance_df.columns:
            best_model = performance_df['f1'].idxmax()
            best_f1 = performance_df.loc[best_model, 'f1']
            print(f"\nBest performing model: {best_model} (F1: {best_f1:.3f})")
    
    print("\n2. MODEL CHARACTERISTICS")
    print("-" * 30)
    
    model_info = {
        'en_core_web_sm': {
            'size': 'Small (~15MB)',
            'speed': 'Fast',
            'accuracy': 'Good',
            'use_case': 'Production, real-time processing'
        },
        'en_core_web_lg': {
            'size': 'Large (~750MB)',
            'speed': 'Medium',
            'accuracy': 'Better',
            'use_case': 'Batch processing, higher accuracy needs'
        },
        'en_core_web_trf': {
            'size': 'Very Large (~500MB)',
            'speed': 'Slow',
            'accuracy': 'Best',
            'use_case': 'Offline analysis, maximum accuracy'
        },
        'dbmdz-bert': {
            'size': 'Large (~400MB)',
            'speed': 'Slow',
            'accuracy': 'Very Good',
            'use_case': 'Research, high-quality extraction'
        },
        'dslim-bert': {
            'size': 'Medium (~400MB)',
            'speed': 'Medium',
            'accuracy': 'Very Good',
            'use_case': 'Balanced performance'
        }
    }
    
    for model_name, info in model_info.items():
        if model_name in tester.models:
            print(f"\n{model_name}:")
            for key, value in info.items():
                print(f"  {key.capitalize()}: {value}")
    
    print("\n3. RECOMMENDATIONS")
    print("-" * 30)
    
    recommendations = """
    For Production Use:
    - en_core_web_sm: Best for real-time applications with good accuracy
    - Fine-tuned spaCy model: Optimal for domain-specific organization extraction
    
    For Research/Analysis:
    - en_core_web_trf: Maximum accuracy for offline batch processing
    - Fine-tuned transformer: Best accuracy but computationally expensive
    
    For Your Market Research Firm:
    - Start with en_core_web_sm + fine-tuning for speed and accuracy balance
    - Consider en_core_web_lg for higher accuracy on important documents
    - Use transformer models for critical analysis where accuracy is paramount
    """
    
    print(recommendations)
    
    print("\n4. FINE-TUNING RECOMMENDATIONS")
    print("-" * 30)
    
    finetuning_advice = """
    Data Requirements:
    - Minimum 100-200 annotated examples for basic improvement
    - 500-1000 examples for significant performance gains
    - Include diverse document types from your domain
    
    Annotation Guidelines:
    - Focus on organization types relevant to your research
    - Include edge cases (subsidiaries, partnerships, etc.)
    - Maintain consistent annotation standards
    
    Training Strategy:
    - Start with spaCy fine-tuning (faster, easier)
    - Move to transformer fine-tuning for maximum accuracy
    - Use active learning to iteratively improve model
    """
    
    print(finetuning_advice)

# Generate comparison report
create_model_comparison_report()

# %% [markdown]
## 8. Advanced Testing and Evaluation Tools

# %%
def advanced_error_analysis(model_results, gold_data):
    """Perform detailed error analysis"""
    
    print("ADVANCED ERROR ANALYSIS")
    print("=" * 50)
    
    error_types = {
        'missed_entities': [],
        'false_positives': [],
        'boundary_errors': [],
        'type_errors': []
    }
    
    for model_name, results in model_results.items():
        if not results:
            continue
            
        print(f"\nAnalyzing {model_name}:")
        model_errors = {
            'missed_entities': [],
            'false_positives': [],
            'boundary_errors': []
        }
        
        for i, result in enumerate(results):
            if i >= len(gold_data):
                break
                
            text = result['text']
            predicted_orgs = {(org['start'], org['end'], org['text']) for org in result['organizations']}
            gold_orgs = {(ent['start'], ent['end'], ent['text']) for ent in gold_data[i]['entities']}
            
            # Find missed entities
            missed = gold_orgs - predicted_orgs
            for start, end, text_span in missed:
                model_errors['missed_entities'].append({
                    'text': text,
                    'missed_entity': text_span,
                    'context': text[max(0, start-20):end+20]
                })
            
            # Find false positives
            false_pos = predicted_orgs - gold_orgs
            for start, end, text_span in false_pos:
                model_errors['false_positives'].append({
                    'text': text,
                    'false_entity': text_span,
                    'context': text[max(0, start-20):end+20]
                })
            
            # Find boundary errors (same text, different boundaries)
            gold_texts = {text_span for _, _, text_span in gold_orgs}
            pred_texts = {text_span for _, _, text_span in predicted_orgs}
            
            for gold_text in gold_texts:
                for pred_text in pred_texts:
                    if gold_text in pred_text or pred_text in gold_text:
                        if gold_text != pred_text:
                            model_errors['boundary_errors'].append({
                                'gold': gold_text,
                                'predicted': pred_text,
                                'text': text
                            })
        
        # Print error summary
        print(f"  Missed entities: {len(model_errors['missed_entities'])}")
        print(f"  False positives: {len(model_errors['false_positives'])}")
        print(f"  Boundary errors: {len(model_errors['boundary_errors'])}")
        
        # Show examples
        if model_errors['missed_entities']:
            print(f"  Example missed: {model_errors['missed_entities'][0]['missed_entity']}")
        if model_errors['false_positives']:
            print(f"  Example false positive: {model_errors['false_positives'][0]['false_entity']}")

# Run advanced error analysis
if tester.results and gold_data:
    advanced_error_analysis(tester.results, gold_data)

# %% [markdown]
## 9. Custom Evaluation Metrics for Organization Extraction

# %%
def organization_specific_metrics(predictions, gold_standard):
    """Calculate metrics specific to organization extraction"""
    
    metrics = {
        'exact_match_accuracy': 0,
        'partial_match_accuracy': 0,
        'entity_level_precision': 0,
        'entity_level_recall': 0,
        'token_level_f1': 0,
        'organization_type_accuracy': {}
    }
    
    # Organization type patterns
    org_types = {
        'corporation': [r'corp\.?, r'corporation, r'inc\.?],
        'llc': [r'llc, r'l\.l\.c\.?],
        'company': [r'company, r'co\.?],
        'bank': [r'bank, r'banking],
        'group': [r'group, r'holdings?],
        'university': [r'university, r'college, r'institute]
    }
    
    def get_org_type(org_name):
        org_lower = org_name.lower()
        for org_type, patterns in org_types.items():
            for pattern in patterns:
                if re.search(pattern, org_lower):
                    return org_type
        return 'other'
    
    # Calculate metrics
    total_exact_matches = 0
    total_partial_matches = 0
    total_predictions = 0
    total_gold = 0
    
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred_list, gold_list in zip(predictions, gold_standard):
        pred_orgs = {org['text'].lower() for org in pred_list['organizations']}
        gold_orgs = {ent['text'].lower() for ent in gold_list['entities']}
        
        # Exact matches
        exact_matches = len(pred_orgs.intersection(gold_orgs))
        total_exact_matches += exact_matches
        
        # Partial matches (fuzzy)
        partial_matches = 0
        for pred_org in pred_orgs:
            for gold_org in gold_orgs:
                if pred_org in gold_org or gold_org in pred_org:
                    partial_matches += 1
                    break
        total_partial_matches += partial_matches
        
        total_predictions += len(pred_orgs)
        total_gold += len(gold_orgs)
        
        # Type-specific accuracy
        for gold_ent in gold_list['entities']:
            org_type = get_org_type(gold_ent['text'])
            type_stats[org_type]['total'] += 1
            
            if gold_ent['text'].lower() in pred_orgs:
                type_stats[org_type]['correct'] += 1
    
    # Calculate final metrics
    metrics['exact_match_accuracy'] = total_exact_matches / total_gold if total_gold > 0 else 0
    metrics['partial_match_accuracy'] = total_partial_matches / total_gold if total_gold > 0 else 0
    metrics['entity_level_precision'] = total_exact_matches / total_predictions if total_predictions > 0 else 0
    metrics['entity_level_recall'] = total_exact_matches / total_gold if total_gold > 0 else 0
    
    # Type-specific accuracy
    for org_type, stats in type_stats.items():
        metrics['organization_type_accuracy'][org_type] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    return metrics

# Calculate organization-specific metrics
if tester.results and gold_data:
    print("\nORGANIZATION-SPECIFIC METRICS")
    print("=" * 50)
    
    for model_name, results in tester.results.items():
        if results:
            metrics = organization_specific_metrics(results[:len(gold_data)], gold_data)
            
            print(f"\n{model_name}:")
            print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
            print(f"  Partial Match Accuracy: {metrics['partial_match_accuracy']:.3f}")
            print(f"  Entity-Level Precision: {metrics['entity_level_precision']:.3f}")
            print(f"  Entity-Level Recall: {metrics['entity_level_recall']:.3f}")
            
            if metrics['organization_type_accuracy']:
                print("  Organization Type Accuracy:")
                for org_type, accuracy in metrics['organization_type_accuracy'].items():
                    print(f"    {org_type}: {accuracy:.3f}")

# %% [markdown]
## 10. Data Augmentation and Active Learning

# %%
def generate_synthetic_training_data():
    """Generate synthetic training data for organization extraction"""
    
    # Organization templates
    org_templates = [
        "{company} {suffix}",
        "The {company} {suffix}",
        "{company} {type}",
        "{adjective} {company} {suffix}"
    ]
    
    # Company name components
    company_names = [
        "Global", "International", "National", "United", "American", "European",
        "Tech", "Data", "Smart", "Digital", "Innovation", "Future", "Advanced",
        "Capital", "Financial", "Investment", "Banking", "Securities", "Holdings"
    ]
    
    company_bases = [
        "Solutions", "Systems", "Technologies", "Dynamics", "Industries", "Services",
        "Partners", "Associates", "Ventures", "Enterprises", "Communications"
    ]
    
    suffixes = ["Inc.", "Corporation", "Corp.", "LLC", "Ltd.", "Company", "Co."]
    types = ["Bank", "Group", "Holdings", "Partners", "Associates"]
    adjectives = ["Leading", "Premier", "Top", "Major", "Key"]
    
    # Sentence templates
    sentence_templates = [
        "{org} announced strong quarterly results.",
        "The partnership between {org1} and {org2} was finalized.",
        "{org} acquired {org2} for ${amount} billion.",
        "Investors are bullish on {org} stock.",
        "{org} reported revenue growth of {percent}%.",
        "The merger of {org1} and {org2} creates a market leader.",
        "{org} expanded its operations to {location}.",
        "Analysts upgraded {org} to a buy rating.",
        "{org} launched a new product line.",
        "{org} signed a strategic agreement with {org2}."
    ]
    
    synthetic_data = []
    
    # Generate synthetic organizations
    synthetic_orgs = []
    for _ in range(50):
        template = np.random.choice(org_templates)
        
        if "{company}" in template:
            company = np.random.choice(company_names) + " " + np.random.choice(company_bases)
        else:
            company = np.random.choice(company_bases)
        
        org_name = template.format(
            company=company,
            suffix=np.random.choice(suffixes),
            type=np.random.choice(types),
            adjective=np.random.choice(adjectives)
        )
        
        synthetic_orgs.append(org_name)
    
    # Generate synthetic sentences
    for _ in range(100):
        template = np.random.choice(sentence_templates)
        
        # Select organizations for the sentence
        org1 = np.random.choice(synthetic_orgs)
        org2 = np.random.choice(synthetic_orgs)
        
        # Generate sentence
        sentence = template.format(
            org=org1,
            org1=org1,
            org2=org2,
            amount=np.random.randint(1, 100),
            percent=np.random.randint(1, 50),
            location=np.random.choice(["Asia", "Europe", "North America", "South America"])
        )
        
        # Find entity positions
        entities = []
        for org in [org1, org2]:
            if org in sentence:
                start = sentence.find(org)
                end = start + len(org)
                entities.append((start, end, "ORG"))
        
        # Remove duplicates
        entities = list(set(entities))
        
        synthetic_data.append((sentence, {"entities": entities}))
    
    return synthetic_data

def active_learning_selection(model, unlabeled_texts, n_samples=10):
    """Select most informative samples for annotation using active learning"""
    
    scores = []
    
    for text in unlabeled_texts:
        if hasattr(model, 'predict'):
            # For transformer models
            predictions = model(text)
            confidence_scores = [pred['score'] for pred in predictions]
        else:
            # For spaCy models
            doc = model(text)
            confidence_scores = []
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    # spaCy doesn't provide confidence, use length as proxy
                    confidence_scores.append(1.0 / len(ent.text))
        
        # Calculate uncertainty (lower average confidence = higher uncertainty)
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            uncertainty = 1 - avg_confidence
        else:
            uncertainty = 1.0  # No entities found = high uncertainty
        
        scores.append((text, uncertainty))
    
    # Sort by uncertainty (descending) and return top samples
    scores.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in scores[:n_samples]]

# Generate synthetic data
print("Generating synthetic training data...")
synthetic_training_data = generate_synthetic_training_data()
print(f"Generated {len(synthetic_training_data)} synthetic examples")

# Show examples
print("\nSynthetic data examples:")
for i, (text, annotations) in enumerate(synthetic_training_data[:3]):
    print(f"{i+1}. {text}")
    for start, end, label in annotations['entities']:
        print(f"   Entity: {text[start:end]} ({label})")

# Active learning example
if tester.models:
    print("\nActive Learning Example:")
    
    # Create some unlabeled texts
    unlabeled_texts = [
        "The acquisition by TechCorp Industries was completed last month.",
        "Global Finance LLC reported exceptional performance this quarter.",
        "Innovation Partners announced a strategic partnership yesterday.",
        "DataSys Corporation expanded their cloud infrastructure.",
        "The investment from Venture Holdings exceeded expectations."
    ]
    
    # Use first available model for active learning
    model_name = list(tester.models.keys())[0]
    model = tester.models[model_name]
    
    selected_samples = active_learning_selection(model, unlabeled_texts, n_samples=3)
    
    print("Most informative samples for annotation:")
    for i, sample in enumerate(selected_samples):
        print(f"{i+1}. {sample}")

# %% [markdown]
## 11. Production Deployment Considerations

# %%
def create_production_pipeline():
    """Create a production-ready NER pipeline"""
    
    class ProductionNERPipeline:
        def __init__(self, model_path=None, model_type='spacy'):
            self.model_type = model_type
            self.model = None
            self.preprocessing_steps = []
            self.postprocessing_steps = []
            self.confidence_threshold = 0.8
            
        def load_model(self, model_path):
            """Load the trained model"""
            if self.model_type == 'spacy':
                try:
                    self.model = spacy.load(model_path)
                except:
                    # Fallback to default model
                    self.model = spacy.load("en_core_web_sm")
            elif self.model_type == 'transformer':
                # Load transformer model
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForTokenClassification.from_pretrained(model_path)
                self.pipeline = pipeline("ner", 
                                        model=self.model, 
                                        tokenizer=self.tokenizer,
                                        aggregation_strategy="simple")
        
        def add_preprocessing_step(self, step_function):
            """Add preprocessing step"""
            self.preprocessing_steps.append(step_function)
        
        def add_postprocessing_step(self, step_function):
            """Add postprocessing step"""
            self.postprocessing_steps.append(step_function)
        
        def preprocess_text(self, text):
            """Apply preprocessing steps"""
            for step in self.preprocessing_steps:
                text = step(text)
            return text
        
        def postprocess_entities(self, entities):
            """Apply postprocessing steps"""
            for step in self.postprocessing_steps:
                entities = step(entities)
            return entities
        
        def extract_organizations(self, text):
            """Extract organizations from text"""
            # Preprocess
            processed_text = self.preprocess_text(text)
            
            # Extract entities
            if self.model_type == 'spacy':
                entities = self._extract_spacy(processed_text)
            else:
                entities = self._extract_transformer(processed_text)
            
            # Postprocess
            entities = self.postprocess_entities(entities)
            
            return entities
        
        def _extract_spacy(self, text):
            """Extract using spaCy model"""
            doc = self.model(text)
            entities = []
            
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities.append({
                        'text': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 1.0,  # spaCy doesn't provide confidence
                        'label': ent.label_
                    })
            
            return entities
        
        def _extract_transformer(self, text):
            """Extract using transformer model"""
            results = self.pipeline(text)
            entities = []
            
            for result in results:
                if 'ORG' in result['entity_group'] and result['score'] >= self.confidence_threshold:
                    entities.append({
                        'text': result['word'],
                        'start': result['start'],
                        'end': result['end'],
                        'confidence': result['score'],
                        'label': result['entity_group']
                    })
            
            return entities
        
        def batch_process(self, texts, batch_size=32):
            """Process multiple texts in batches"""
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_results = []
                
                for text in batch:
                    entities = self.extract_organizations(text)
                    batch_results.append({
                        'text': text,
                        'entities': entities,
                        'entity_count': len(entities)
                    })
                
                results.extend(batch_results)
            
            return results
    
    # Define preprocessing functions
    def clean_text(text):
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\&\-\(\)]', '', text)
        return text.strip()
    
    def normalize_quotes(text):
        """Normalize quotation marks"""
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'['']', "'", text)
        return text
    
    # Define postprocessing functions
    def filter_short_entities(entities):
        """Remove very short entities"""
        return [ent for ent in entities if len(ent['text']) > 2]
    
    def deduplicate_entities(entities):
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity['text'].lower(), entity['start'], entity['end'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def merge_overlapping_entities(entities):
        """Merge overlapping entities"""
        if not entities:
            return entities
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        merged = [sorted_entities[0]]
        
        for entity in sorted_entities[1:]:
            last_entity = merged[-1]
            
            # Check for overlap
            if entity['start'] <= last_entity['end']:
                # Merge entities
                merged_text = entity['text'] if len(entity['text']) > len(last_entity['text']) else last_entity['text']
                merged_entity = {
                    'text': merged_text,
                    'start': last_entity['start'],
                    'end': max(entity['end'], last_entity['end']),
                    'confidence': max(entity['confidence'], last_entity['confidence']),
                    'label': entity['label']
                }
                merged[-1] = merged_entity
            else:
                merged.append(entity)
        
        return merged
    
    # Create production pipeline
    pipeline_instance = ProductionNERPipeline(model_type='spacy')
    
    # Add preprocessing steps
    pipeline_instance.add_preprocessing_step(clean_text)
    pipeline_instance.add_preprocessing_step(normalize_quotes)
    
    # Add postprocessing steps
    pipeline_instance.add_postprocessing_step(filter_short_entities)
    pipeline_instance.add_postprocessing_step(deduplicate_entities)
    pipeline_instance.add_postprocessing_step(merge_overlapping_entities)
    
    return pipeline_instance

# Create production pipeline
print("Creating production NER pipeline...")
production_pipeline = create_production_pipeline()

# Load model (using default for demo)
if tester.models:
    model_name = 'en_core_web_sm' if 'en_core_web_sm' in tester.models else list(tester.models.keys())[0]
    production_pipeline.model = tester.models[model_name]
    production_pipeline.model_type = 'spacy'

# Test production pipeline
test_texts = [
    "Microsoft Corporation and Apple Inc. announced a joint venture yesterday.",
    "The acquisition of LinkedIn Corp. by Microsoft was completed in 2016.",
    "JPMorgan Chase & Co. reported strong quarterly earnings beating expectations."
]

print("\nTesting production pipeline:")
results = production_pipeline.batch_process(test_texts)

for i, result in enumerate(results):
    print(f"\nText {i+1}: {result['text']}")
    print(f"Organizations found: {result['entity_count']}")
    for entity in result['entities']:
        print(f"  - {entity['text']} (confidence: {entity['confidence']:.3f})")

# %% [markdown]
## 12. Model Performance Monitoring and Logging

# %%
import logging
from datetime import datetime
import json

def setup_performance_monitoring():
    """Setup performance monitoring and logging"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ner_performance.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('NER_Performance')
    
    class PerformanceMonitor:
        def __init__(self, logger):
            self.logger = logger
            self.metrics_history = []
            self.processing_times = []
            self.error_count = 0
            self.total_processed = 0
        
        def log_processing(self, text, entities, processing_time, model_name):
            """Log processing results"""
            self.total_processed += 1
            self.processing_times.append(processing_time)
            
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'model': model_name,
                'text_length': len(text),
                'entities_found': len(entities),
                'processing_time': processing_time,
                'entities': [{'text': e['text'], 'confidence': e.get('confidence', 1.0)} for e in entities]
            }
            
            self.logger.info(f"Processed text: {json.dumps(log_data)}")
        
        def log_error(self, error_message, text=None):
            """Log processing errors"""
            self.error_count += 1
            
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'error': str(error_message),
                'text_preview': text[:100] if text else None
            }
            
            self.logger.error(f"Processing error: {json.dumps(error_data)}")
        
        def calculate_performance_metrics(self):
            """Calculate performance metrics"""
            if not self.processing_times:
                return {}
            
            metrics = {
                'total_processed': self.total_processed,
                'error_rate': self.error_count / self.total_processed if self.total_processed > 0 else 0,
                'avg_processing_time': np.mean(self.processing_times),
                'median_processing_time': np.median(self.processing_times),
                'max_processing_time': np.max(self.processing_times),
                'min_processing_time': np.min(self.processing_times),
                'throughput_per_second': 1 / np.mean(self.processing_times) if np.mean(self.processing_times) > 0 else 0
            }
            
            return metrics
        
        def generate_performance_report(self):
            """Generate performance report"""
            metrics = self.calculate_performance_metrics()
            
            report = f"""
PERFORMANCE MONITORING REPORT
Generated: {datetime.now().isoformat()}
{'='*50}

Processing Statistics:
- Total documents processed: {metrics.get('total_processed', 0)}
- Error rate: {metrics.get('error_rate', 0):.3f}
- Average processing time: {metrics.get('avg_processing_time', 0):.3f}s
- Median processing time: {metrics.get('median_processing_time', 0):.3f}s
- Throughput: {metrics.get('throughput_per_second', 0):.2f} docs/second

Performance Trends:
- Recent processing times: {self.processing_times[-10:] if len(self.processing_times) >= 10 else self.processing_times}
"""
            
            return report
    
    return PerformanceMonitor(logger)

# Setup monitoring
monitor = setup_performance_monitoring()

# Example usage with timing
import time

def timed_extraction(pipeline, text, model_name):
    """Extract entities with timing"""
    start_time = time.time()
    
    try:
        entities = pipeline.extract_organizations(text)
        processing_time = time.time() - start_time
        
        monitor.log_processing(text, entities, processing_time, model_name)
        return entities
    
    except Exception as e:
        processing_time = time.time() - start_time
        monitor.log_error(e, text)
        return []

# Test monitoring
if hasattr(production_pipeline, 'model') and production_pipeline.model:
    test_texts_monitoring = [
        "Apple Inc. and Microsoft Corporation are technology leaders.",
        "JPMorgan Chase & Co. acquired a fintech startup.",
        "Google LLC announced new AI capabilities."
    ]
    
    print("Testing performance monitoring:")
    for text in test_texts_monitoring:
        entities = timed_extraction(production_pipeline, text, "production_spacy")
        print(f"Found {len(entities)} entities in: {text[:50]}...")

# Generate performance report
print("\n" + monitor.generate_performance_report())

# %% [markdown]
## Summary and Next Steps

print("""
NOTEBOOK SUMMARY AND NEXT STEPS
================================

This notebook has provided:

1. ✅ Comprehensive NER model testing framework
2. ✅ Performance evaluation and comparison tools
3. ✅ Fine-tuning pipelines for both spaCy and transformer models
4. ✅ Production-ready deployment pipeline
5. ✅ Performance monitoring and logging system
6. ✅ Data augmentation and active learning strategies

RECOMMENDED NEXT STEPS FOR YOUR MARKET RESEARCH FIRM:

1. Data Collection & Annotation:
   - Collect 500-1000 documents from your domain
   - Create annotation guidelines for organization types
   - Use active learning to select most informative samples

2. Model Selection:
   - Start with en_core_web_sm for speed
   - Fine-tune with your domain-specific data
   - Consider en_core_web_lg for higher accuracy needs

3. Production Deployment:
   - Implement the production pipeline with monitoring
   - Set up automated retraining with new data
   - Create API endpoints for integration

4. Continuous Improvement:
   - Monitor performance metrics regularly
   - Collect feedback from publishing team
   - Retrain models with new domain-specific data
   - A/B test different models in production

5. Integration:
   - Build API for the Streamlit app
   - Create batch processing capabilities
   - Integrate with your existing publishing workflow
   - Set up automated quality checks

PERFORMANCE BENCHMARKS FROM THIS NOTEBOOK:
- Model comparison across different architectures
- Error analysis identifying improvement areas
- Processing speed benchmarks for production planning
- Accuracy metrics for different organization types

FILES TO SAVE:
- Fine-tuned models (spacy_org_model/, transformer_org_model/)
- Performance logs (ner_performance.log)
- Evaluation results (evaluation_results.json)
- Training data (annotated_org_data.json)

NEXT JUPYTER NOTEBOOKS TO CREATE:
1. data_annotation_tools.ipynb - Tools for team annotation
2. model_deployment.ipynb - Production deployment guide
3. performance_analysis.ipynb - Ongoing performance monitoring
4. domain_adaptation.ipynb - Adapting to market research terminology
"""

print("Notebook complete! All components ready for production implementation.")

# %% [markdown]
## Appendix: Quick Start Commands

# %%
print("""
QUICK START COMMANDS FOR YOUR TEAM:
===================================

# 1. Install Dependencies
pip install spacy transformers datasets torch evaluate seqeval
pip install spacy-transformers matplotlib seaborn plotly
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

# 2. Load and Test Models
tester = NERModelTester()
tester.load_spacy_models()
tester.run_comprehensive_test(your_sample_texts)

# 3. Fine-tune spaCy Model
fine_tuned_model = fine_tune_spacy_model()

# 4. Create Production Pipeline
pipeline = create_production_pipeline()
results = pipeline.batch_process(your_texts)

# 5. Monitor Performance
monitor = setup_performance_monitoring()
entities = timed_extraction(pipeline, text, "model_name")

# 6. Generate Reports
evaluation_results = evaluate_model_performance(model_results, gold_data)
create_model_comparison_report()
""")