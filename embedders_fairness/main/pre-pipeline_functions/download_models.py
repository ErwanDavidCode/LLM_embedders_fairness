from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import os

# Target folder where to store the models
BASE_DIR = "../../models"

# Must be coherent with the json of the model we use.
models = {
    # "MPNet": "sentence-transformers/all-mpnet-base-v2",
    # "MiniLM-L12": "sentence-transformers/all-MiniLM-L12-v2",
    # "DistilRoBERTa": "sentence-transformers/all-distilroberta-v1",
    # "MiniLM-L12-paraphrase": "sentence-transformers/paraphrase-MiniLM-L12-v2",
    # "MiniLM-L12-multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # "DistilRoBERTa-paraphrase": "sentence-transformers/paraphrase-distilroberta-base-v1",
    # "MPNet-paraphrase": "sentence-transformers/paraphrase-mpnet-base-v2",
    # "e5-large-v2": "intfloat/e5-large-v2",
    # "stella-base-en-v2": "infgrad/stella-base-en-v2",
    # "UAE-Large-V1": "WhereIsAI/UAE-Large-V1",
    # "gte-large": "thenlper/gte-large",
    # "sup-simcse-bert-base": "princeton-nlp/sup-simcse-bert-base-uncased",
    # "ember-v1": "llmrails/ember-v1",

    # "SFR-Embedding-Mistral": "Salesforce/SFR-Embedding-Mistral",
    # "gtr-t5-large": "sentence-transformers/gtr-t5-large",
    # "sentence-t5-large": "sentence-transformers/sentence-t5-large",
    # "average_word_embeddings_glove": "sentence-transformers/average_word_embeddings_glove.6B.300d",
    # "LaBSE": "sentence-transformers/LaBSE",

    # "BERT-NLI": "sentence-transformers/bert-base-nli-mean-tokens", #to delete
    # "DistilBERT-NLI": "sentence-transformers/distilbert-base-nli-stsb-mean-tokens", #to delete

}

# Create the base folder if it does not exist
os.makedirs(BASE_DIR, exist_ok=True)

# Download
for name, hf_path in models.items():
    print(f"⏬ Downloading {name} from {hf_path} ...")
    model_path = os.path.join(BASE_DIR, f"models--{hf_path.replace('/', '--')}")
    snapshot_download(repo_id=hf_path, local_dir=model_path, local_dir_use_symlinks=False)
    print(f"✅ {name} saved in {model_path}")
