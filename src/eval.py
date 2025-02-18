import numpy as np
import torch
from bert_score import score
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import argparse
from tqdm import tqdm
from readmepp import ReadMe
from utils import *
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
corpus_lang = {
    "cefrsp": ["en"],
    "readme": ["ar", "en", "fr", "hi", "ru"],
    "cefrsp_whole": ["en"]
}
cefrmap = {
    1: "A1", 2: "A2", 3: "B1"
}


def clean_llm_generation(generation):
    sentlist = generation.split("\n")
    for sent in sentlist:
        sent = sent.strip()
        if sent != "":
            return sent
    return None


def fluency_classify(input1, model, tokenizer, lang="en"):
    inputs = tokenizer(input1, return_tensors='pt', padding = True)
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction_labels = torch.argmax(logits, dim=1).cpu().numpy()
    # label 0 means acceptable. Need to inverse - for 2020 EMNLP
    if lang == "en": prediction_labels = 1 - prediction_labels
    return prediction_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'evaluation param')
    parser.add_argument("--file_path", type = str, help = 'Evaluation file path', default = "LLMGeneration/CEFR-SP/en_few-shot_29.csv")
    parser.add_argument("--lang", type = str, help = 'Language', default = "en")
    parser.add_argument("--cola", type = str, help = 'Cola assessment path', default = "roberta-large-cola-krishna2020")
    parser.add_argument("--STS_model", type = str, help = 'STS model path', default = "all-MiniLM-L6-v2")
    parser.add_argument("--BS_model", type = str, help = 'BertScore model path', default = "roberta-large")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.file_path)
    Sentence_Source = []  # original sentence
    llm_gene_CEFR = []  # LLM generation for desired CEFR, from A1->B1
    CEFR_Label = []  # desired CEFR label
    for cefr in range(1, 4):
        src_sent_ = df["Sentence"].tolist()
        llm_gene_ = [clean_llm_generation(sent) for sent in df[f"llm_gene_CEFR{cefr}"].tolist()]
        Sentence_Source += src_sent_
        llm_gene_CEFR += llm_gene_
        CEFR_Label += [cefr for _ in range(len(Sentence_Source))]

    """ calculating CEFR level """
    readmepp_predictor = ReadMe(lang=args.lang)
    readmepp_predictor.model.to(device)

    CEFR_Label = np.array(CEFR_Label)
    CEFR_Pred = np.array([readmepp_predictor.predict(sent) for sent in tqdm(llm_gene_CEFR)])
    exaacc = exact_accuracy(CEFR_Label, CEFR_Pred)
    adjacc = adjacency_accuracy(CEFR_Label, CEFR_Pred)
    rmse = np.sqrt(np.mean((CEFR_Label - CEFR_Pred) ** 2))
    correlation, p_value = spearmanr(CEFR_Label, CEFR_Pred)

    """ calculating Bertscore and STS """
    sts_model = SentenceTransformer(args.STS_model)
    sts_model.to(device)
    STS = []
    for i, srcsent in tqdm(enumerate(Sentence_Source)):
        gensent = llm_gene_CEFR[i]
        embeddings = sts_model.encode([srcsent, gensent])
        sem_sim = float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))
        STS.append(sem_sim)
    STS = np.mean(STS)

    P, R, F1 = score(
        cands=llm_gene_CEFR,
        refs=Sentence_Source,
        model_type=args.BS_model,
        lang=args.lang,
        verbose=True,
        device=args.device,
        rescale_with_baseline=True,
        batch_size=32,
    )
    P = np.mean(P)
    R = np.mean(R)
    F1 = np.mean(F1)

    """ calculating COLA(fluency) score """
    BatchSize = 16
    model = AutoModelForSequenceClassification.from_pretrained(args.cola)
    tokenizer = AutoTokenizer.from_pretrained(args.cola)
    model.to(device)
    COLA = []
    for idx, _ in tqdm(enumerate(llm_gene_CEFR[::BatchSize])):
        sent = llm_gene_CEFR[idx * BatchSize:(idx + 1) * BatchSize]
        flu = fluency_classify(sent, model, tokenizer).tolist()
        COLA += flu
    COLA = np.mean(COLA)

    print({
        "ρ": correlation,
        "adjacc": adjacc,
        "exaacc": exaacc,
        "rmse": rmse,
        "STS": STS,
        "BS_F1": F1,
        "BS_R": R,
        "BS_P": P,
        "Cola": COLA
    })




