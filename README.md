# TellAScam

TellAScam is a machine-learning-powered scam and spam detection project designed to analyze suspicious text messages and provide a risk signal through a REST API.

The current version uses natural language processing to classify SMS messages as spam or legitimate. The project focuses not only on model performance, but also on building a reproducible ML pipeline, analyzing model failures, and exposing predictions through a software interface that can later support web or mobile clients.

## Features

- Text classification using TF-IDF and Logistic Regression
- Unigram and bigram text features
- Class weighting to address dataset imbalance
- Stratified train/test splitting
- 5-fold stratified cross-validation
- Precision, recall, F1, and confusion-matrix evaluation
- False-positive and false-negative analysis
- Serialized model pipeline for reusable inference
- FastAPI REST API
- Automatic request validation with Pydantic
- Automated inference tests with pytest

## Architecture

```text
                     TellAScam

                        Message
                           |
                           v
                    FastAPI REST API
                           |
                           v
                  Saved ML Pipeline
                           |
                +----------+----------+
                |                     |
                v                     v
         TF-IDF Vectorizer     Logistic Regression
         Unigrams + Bigrams     Balanced Weights
                |                     |
                +----------+----------+
                           |
                           v
                  Prediction + Score
```

Training and inference are kept separate. The model is trained offline and serialized to disk. The API loads the trained pipeline once at startup and reuses it for incoming prediction requests.

## Machine Learning Approach

### Dataset

The initial model uses the UCI SMS Spam Collection, containing 5,572 labeled SMS messages. The dataset is provided by the UCI Machine Learning Repository and is licensed under CC BY 4.0, which is not included in the repository and must downloaded separately.

During exploratory data analysis:

- 4,825 messages were labeled legitimate (`ham`)
- 747 messages were labeled `spam`
- No missing labels or messages were found
- 403 exact duplicate rows were identified and removed
- No remaining duplicate messages or conflicting labels were found

The resulting dataset contained 5,169 unique messages.

The dataset is imbalanced, with legitimate messages significantly outnumbering spam messages. This made accuracy alone an insufficient evaluation metric.

### Preprocessing

The project intentionally avoids aggressive text cleaning in the initial model. Information such as numbers, unusual spelling, and other textual patterns may contain useful signals for spam detection.

Exact duplicate messages are removed before data splitting to reduce the risk of identical observations appearing in both training and evaluation data.

### Feature Extraction

TellAScam uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features.

TF-IDF assigns weights based on both:

- how frequently a term occurs within a message
- how common or rare that term is across the training corpus

The final configuration includes both unigrams and bigrams:

```python
TfidfVectorizer(ngram_range=(1, 2))
```

This allows the model to represent individual terms as well as two-word sequences that provide additional local context.

### Classification

Logistic Regression is used as the initial classifier.

The baseline model showed high precision but lower recall for spam messages:

| Metric | Baseline |
| --- | ---: |
| Accuracy | 96.42% |
| Precision | 96.08% |
| Recall | 74.81% |
| F1 | 84.12% |

The baseline confusion matrix was:

```text
[[899   4]
 [ 33  98]]
```

This showed that the classifier produced few false alarms but failed to identify 33 of the 131 spam messages in the held-out split.

Because spam represented the minority class, balanced class weighting was evaluated.

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
```

On the same held-out split, the balanced model produced:

| Metric | Balanced |
| --- | ---: |
| Accuracy | 97.39% |
| Precision | 88.24% |
| Recall | 91.60% |
| F1 | 89.89% |

Its confusion matrix was:

```text
[[887  16]
 [ 11 120]]
```

False negatives decreased from 33 to 11, while false positives increased from 4 to 16.

This represents the expected precision-recall tradeoff from making the classifier more sensitive to the minority spam class.

## Cross-Validation

Further model selection was performed using 5-fold stratified cross-validation on the training data.

Two TF-IDF configurations were compared while keeping the classifier and class weighting constant:

| Features | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Unigrams | 89.51% | 94.25% | 91.79% |
| Unigrams + Bigrams | **91.70%** | **94.63%** | **93.12%** |

Adding bigrams improved mean precision, recall, and F1 across the validation folds.

The selected MVP configuration therefore uses:

```text
TF-IDF
ngram_range=(1, 2)

        +

Logistic Regression
class_weight="balanced"
```

## Error Analysis

Model development included analysis of false positives and false negatives rather than relying only on aggregate metrics.

Observed failure patterns included:

- legitimate messages containing spam-associated vocabulary
- legitimate security warnings
- short messages involving calls or SMS
- unusually abbreviated spam messages
- phone-number-heavy promotional messages
- very short advertisements
- messages requiring more contextual understanding than individual words provide

This analysis motivated evaluating bigram features to provide additional local context.

## API

TellAScam exposes predictions through FastAPI.

### Run the server

After training the model:

```bash
python -m uvicorn tellascam.api:app --reload
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Prediction Endpoint

`POST /predict`

Example request:

```json
{
  "message": "URGENT: You have won a cash prize. Claim it now."
}
```

Example response:

```json
{
  "prediction": "spam",
  "spam_probability": 0.91
}
```

The returned probability is the model's estimated probability for the `spam` class. It should not be interpreted as a general probability that a message is fraudulent.

## Installation

Clone the repository:

```bash
git clone https://github.com/Under-1evel/TellAScam.git
cd TellAScam
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Training

Place the SMS Spam Collection dataset in:

```text
data/raw/SMSSpamCollection
```

Then run:

```bash
python -m tellascam.train
```

The trained model is saved locally to:

```text
models/tellascam_model.joblib
```

Datasets and generated model artifacts are intentionally excluded from Git version control.

## Testing

Run the automated test suite with:

```bash
python -m pytest
```

Current tests verify core inference behavior, including valid classification labels and probability bounds.

## Project Structure

```text
TellAScam/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   └── tellascam/
│       ├── __init__.py
│       ├── api.py
│       ├── evaluate.py
│       ├── predict.py
│       ├── preprocessing.py
│       └── train.py
├── tests/
│   └── test_predict.py
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Limitations

TellAScam v1 is an experimental spam-classification system and should not be treated as a production security tool.

The initial model is trained on an older SMS spam dataset and does not represent the full range of modern phishing, impersonation, social engineering, or other scams.

TF-IDF and Logistic Regression are effective lightweight text-classification methods, but they have limited understanding of semantic context. The current system also does not inspect URLs, domains, sender identity, attachments, images, or external threat intelligence.

The model's `spam_probability` therefore represents confidence in the learned spam classification rather than a comprehensive scam-risk score.

## Future Work

TellAScam is intended to evolve into a multi-signal scam detection system. Potential future improvements include:

- phishing-specific and more recent training datasets
- URL and domain risk analysis
- sender and message metadata analysis
- transformer-based NLP model comparison
- model probability calibration and threshold tuning
- screenshot and image analysis
- multi-signal risk scoring
- explainable warnings showing why a message was considered suspicious
- web interface
- containerization and cloud deployment
- model and API monitoring

## License

This project is licensed under the MIT License.

The dataset used for model development is distributed separately under its own license and is not included directly in this repository.