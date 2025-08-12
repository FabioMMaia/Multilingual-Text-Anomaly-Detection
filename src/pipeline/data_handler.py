from datasets import load_dataset, concatenate_datasets, Dataset
from kagglehub import KaggleDatasetAdapter
import kagglehub
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import json

def parse_wikinews_to_parquet_auto(parquet_output_path):
    import os
    import bz2
    import re
    import requests
    import mwparserfromhell
    import xml.etree.ElementTree as ET
    from tqdm import tqdm

    # Dump config
    dump_url = "https://dumps.wikimedia.org/ptwikinews/20250420/ptwikinews-20250420-pages-meta-current.xml.bz2"
    dump_dir = "wikinews_dataset"
    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(dump_dir, os.path.basename(dump_url))

    # Download if needed
    if not os.path.exists(dump_path):
        print("Downloading WikiNews dump...")
        with requests.get(dump_url, stream=True) as r:
            r.raise_for_status()
            with open(dump_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")

    # Helpers
    def is_redirect(text):
        return bool(text and re.search(r"#redirect \[\[.*?\]\]", text, re.IGNORECASE))

    def extract_categories(text):
        pattern = re.compile(r"\[\[Categoria:([^\]]*)\]\]", re.IGNORECASE)
        return pattern.findall(text), pattern.sub('', text)

    def extract_dates(text):
        pattern = re.compile(r"\{\{data\|([^\}]*)\}\}", re.IGNORECASE)
        return pattern.findall(text), pattern.sub('', text)

    def drop_sources(text):
        return re.sub(r"\{\{Fonte\|[^\}]*\}\}", '', text, flags=re.IGNORECASE)

    # Category priority
    priority = ["Política", "Economia", "Saúde", "Ciência e tecnologia", "Cultura"]

    def pick_priority(cats):
        for p in priority:
            for c in cats:
                if p.lower() in c.lower():
                    return p
        return None

    # Parse the dump
    print("Parsing and extracting content...")
    entries = []
    with bz2.open(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("start", "end"))
        context = iter(context)
        _, root = next(context)
        current = {}
        for event, elem in tqdm(context, desc="Parsing"):
            tag = elem.tag.lower()
            if event == "end":
                if tag.endswith("title"):
                    current["title"] = elem.text or ""
                elif tag.endswith("text"):
                    current["text"] = elem.text or ""
                elif tag.endswith("id") and "id" not in current:
                    current["pageid"] = elem.text or ""
                elif tag.endswith("ns"):
                    current["ns"] = elem.text
                elif tag.endswith("page"):
                    if current.get("ns") != "0":
                        current = {}
                        root.clear()
                        continue

                    raw_text = current.get("text", "")
                    if is_redirect(raw_text):
                        current = {}
                        root.clear()
                        continue

                    cats, raw_text = extract_categories(raw_text)
                    main_cat = pick_priority(cats)
                    if not main_cat:
                        current = {}
                        root.clear()
                        continue

                    raw_text = drop_sources(raw_text)
                    raw_text = extract_dates(raw_text)[1]
                    clean_text = mwparserfromhell.parse(raw_text).strip_code().strip()

                    entries.append({
                        "text": clean_text,
                        "label": main_cat
                    })

                    current = {}
                    root.clear()

    print(f"Extracted {len(entries)} valid articles with main categories.")

    # Save only 'text' and 'label'
    df = pd.DataFrame(entries)[["text", "label"]]
    df = df[df["text"].str.strip().astype(bool)]  # remove empty texts
    df.to_parquet(parquet_output_path, index=False)
    print(f"Saved to: {parquet_output_path}")


def process_dataset(dataset_name, subset_name, model_name, encoder_class, source, project_path=".", max_samples=60_000, sampling_strategy="stratified"):
    print(f"\n🔄 Processing: {dataset_name} ({subset_name}) with {model_name}")

    assert sampling_strategy in ["stratified", "head"], f"Unknown sampling strategy: {sampling_strategy}"

    model_short = model_name.split("/")[-1]
    dataset_short = dataset_name.split("/")[-1]

    # Define file paths
    data_dir = os.path.join(project_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    path_texts = os.path.join(data_dir, f"texts_{dataset_short}.parquet")
    path_labels = os.path.join(data_dir, f"labels_{dataset_short}.parquet")
    path_embeddings = os.path.join(data_dir, f"embeddings_{dataset_short}_{model_short}.parquet")

    # Skip if embeddings already exist
    if os.path.exists(path_embeddings):
        print(f"Embeddings already exist at {path_embeddings}. Skipping.")
        return

    # Load preprocessed text/label if available
    if os.path.exists(path_texts) and os.path.exists(path_labels):
        print(f"Loading texts and labels from cached .parquet files.")
        texts = pd.read_parquet(path_texts)["text"].tolist()
        labels = pd.read_parquet(path_labels)["label"].to_numpy()

    else:
        # Load dataset based on the source
        if source == "hf":
            try:
                dataset = load_dataset(dataset_name, name=subset_name)
            except:
                dataset = load_dataset(dataset_name, name=subset_name, trust_remote_code=True)

            if dataset_name == "tweets-hate-speech-detection/tweets_hate_speech_detection":
                full_dataset = concatenate_datasets([dataset[k] for k in dataset.keys() if k in ["train"]]) # for this specific dataset, labels are only available in the train split
                print("Renaming columns for tweets_hate_speech_detection: 'tweet' >> 'text'")
                full_dataset = full_dataset.rename_column("tweet", "text")
            else:
                full_dataset = concatenate_datasets([dataset[k] for k in dataset.keys() if k in ["train", "test", "validation"]])

        elif source == "kaggle":
            df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset_name, subset_name)
            if dataset_name == "augustop/portuguese-tweets-for-sentiment-analysis":
                df = df.rename(columns={"tweet_text": "text", "sentiment": "label"})
                df = df[["text", "label"]]
            full_dataset = Dataset.from_pandas(df)

        elif source == "wikinews_dump":
            if not os.path.exists(subset_name):
                print(f"{subset_name} not found. Downloading and parsing WikiNews...")
                parse_wikinews_to_parquet_auto(subset_name)

            df = pd.read_parquet(subset_name)
            df = df.rename(columns={"body": "text", "main_category": "label"})
            df = df[["text", "label"]]
            full_dataset = Dataset.from_pandas(df)

        else:
            raise ValueError(f"Unknown source: {source}")

        # Detect text and label columns
        colnames = full_dataset.column_names
        text_key = "text" if "text" in colnames else colnames[0]
        label_key = "label" if "label" in colnames else colnames[-1]

        # Extract and format text
        first_item = full_dataset[text_key][0]
        if isinstance(first_item, list):
            texts = [" ".join(map(str, x)) for x in full_dataset[text_key]]
        else:
            texts = list(map(str, full_dataset[text_key]))

        labels = np.array(full_dataset[label_key])

        # Optional sample limit
        if len(texts) > max_samples:
            print(f"Original size: {len(texts)}")

            df_temp = pd.DataFrame({'text': texts, 'label': labels})

            if sampling_strategy == "head":
                print(f"Truncating by taking the first {max_samples} samples.")
                texts = texts[:max_samples]
                labels = labels[:max_samples]


            elif sampling_strategy == "stratified":
                print(f"Truncating with stratified sampling to {max_samples} samples.")
                df_sampled, _ = train_test_split(
                    df_temp,
                    train_size=max_samples,
                    stratify=df_temp["label"],
                    random_state=42
                )
                texts = df_sampled["text"].tolist()
                labels = df_sampled["label"].to_numpy()
 
        # Save raw text and labels
        pd.DataFrame(texts, columns=["text"]).to_parquet(path_texts)
        pd.DataFrame(labels, columns=["label"]).to_parquet(path_labels)

    # Encode and save embeddings
    encoder = encoder_class(model_name)
    embeddings = encoder.encode_texts(texts)
    pd.DataFrame(embeddings).to_parquet(path_embeddings)

    assert len(embeddings) == len(labels) == len(texts)
    print(f"✅ Saved embeddings for {dataset_short} with {model_short}")




