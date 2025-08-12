# semi-supervised-text-anomaly-detection
This repository contains the code and resources for the paper "Anomaly Detection in Text Data: A Semi-Supervised Approach Applied to the Portuguese Domain". The research focuses on applying state-of-the-art anomaly detection techniques to Portuguese text corpora, specifically addressing challenges in tasks like hate speech detection and sentiment analysis.

Key Features:
- Semi-Supervised Anomaly Detection: Implements a modified version of the DevNet model, introducing a flexible loss function that balances regular and anomalous samples based on contamination levels.
- Text Representations: Utilizes two pre-trained BERT models, BERTimbau (for Portuguese) and multilingual SBERT, comparing their performance in anomaly detection tasks.
- Flexible Loss Function: Allows dynamic adjustment for different contamination levels using the parameter η, ensuring optimal detection across varying datasets.
- Experiments: Tested on two Brazilian datasets: Told-Br for hate speech detection and UTLC-Movies for sentiment analysis, achieving promising results with minimal labeled data.

Keywords: Anomaly detection, Text data, Semi-supervised learning, Portuguese text, Deep learning, BERTimbau, SBERT, DevNet, Sentiment analysis, Hate speech detection, Natural Language Processing (NLP), Machine learning, Text embeddings, Outlier detection.
