# 👋simplify-step-by-step
This repository is used to reproduce the experimental results of
_**Let’s Simplify Step by Step: Guiding LLM Towards Multilingual Zero-shot Readability-Controlled Sentence Simplification**_, 
and meanwhile, it supplements the details that are not described clearly in the paper. I hope it can help and inspire you.

### Research Framework
<img src="img/research_framework.jpg" alt="" width="100%" style="max-width: 100%; height: auto; display: block; margin: 0 auto;"> 

<br>

> **Abstract** 
> - **Motivation**: Large language models (LLMs) have shown promise in text simplification tasks that adapt content for different reading levels. **However, these models struggle when asked to make substantial adjustments in complexity**, particularly when simplifying from advanced to elementary reading levels (e.g., from CEFR level C2 to A1). 
> - **Methodology**: To address this challenge, we propose a novel framework that combines dynamic path planning with semantic-aware exemplar selection and Chain-of-Thought generation to break down complex simplification tasks. Our approach separates readability path planning from semantic preservation, enabling focused optimization of each objective. 


## 🥇0. Our main results:
> - The following are the heatmap displays of our model and the one-step LLM in terms of three metrics, namely _Adjacent Accuracy_, _Exact Accuracy_, and _RMSE_, on two benchmarks, CEFR-SP and README.  
> - We achieve up to 26\% improvement in adjacent accuracy while maintaining semantic fidelity. 


<style>
    .image-container {
        display: flex;
        flex-wrap: wrap;
    }

    .image-container img {
        width: 33.33%;
        max-width: 100%;
        height: auto;
        box-sizing: border-box;
        padding: 2px;  
    }
</style>


- _**CEFR-SP_Partial**_
<div class="image-container">
    <img src="img/CEFR-SP/CEFR-SP_Partial_dp-cot_adjacc.jpg" alt="">
    <img src="img/CEFR-SP/CEFR-SP_Partial_dp-cot_exaacc.jpg" alt="">
    <img src="img/CEFR-SP/CEFR-SP_Partial_dp-cot_rmse.jpg" alt="">
</div>

- _**CEFR-SP_Whole**_
<div class="image-container">
    <img src="img/CEFR-SP/CEFR-SP_Whole_dp-cot_adjacc.jpg" alt="">
    <img src="img/CEFR-SP/CEFR-SP_Whole_dp-cot_exaacc.jpg" alt="">
    <img src="img/CEFR-SP/CEFR-SP_Whole_dp-cot_rmse.jpg" alt="">
</div>

- _**README_Arabic**_
<div class="image-container">
    <img src="img/README/ar/README-ar_dp-cot_adjacc.jpg" alt="">
    <img src="img/README/ar/README-ar_dp-cot_exaacc.jpg" alt="">
    <img src="img/README/ar/README-ar_dp-cot_rmse.jpg" alt="">
</div>

- _**README_English**_
<div class="image-container">
    <img src="img/README/en/README-en_dp-cot_adjacc.jpg" alt="">
    <img src="img/README/en/README-en_dp-cot_exaacc.jpg" alt="">
    <img src="img/README/en/README-en_dp-cot_rmse.jpg" alt="">
</div>

- _**README_French**_
<div class="image-container">
    <img src="img/README/fr/README-fr_dp-cot_adjacc.jpg" alt="">
    <img src="img/README/fr/README-fr_dp-cot_exaacc.jpg" alt="">
    <img src="img/README/fr/README-fr_dp-cot_rmse.jpg" alt="">
</div>

- _**README_Hindi**_
<div class="image-container">
    <img src="img/README/hi/README-hi_dp-cot_adjacc.jpg" alt="">
    <img src="img/README/hi/README-hi_dp-cot_exaacc.jpg" alt="">
    <img src="img/README/hi/README-hi_dp-cot_rmse.jpg" alt="">
</div>

- _**README_Russian**_
<div class="image-container">
    <img src="img/README/ru/README-ru_dp-cot_adjacc.jpg" alt="">
    <img src="img/README/ru/README-ru_dp-cot_exaacc.jpg" alt="">
    <img src="img/README/ru/README-ru_dp-cot_rmse.jpg" alt="">
</div>

## 1. project structure
```
simplify-step-by-step
├─ data  # storing raw data which includes at least two columns named "Rating" and "Sentence". 
│  ├─ ar
│  ├─ en   # include CEFR-SP_partial
│  ├─ fr
│  ├─ hi
│  ├─ ru
│  ├─ prompt_expert  # Semantic prompt used in paper
├─ img   # visual auto-evaluation metric (Acc, RMSE) comparison between one-step LLM and our DP-planner+CoT
│  ├─ CEFR-SP
│  └─ README
│     ├─ ar
│     ├─ en
│     ├─ fr
│     ├─ hi
│     └─ ru
├─ README.md
├─ src
│  ├─ LLMGeneration  # llm generation for desired cefr-level
│  │  ├─ CEFR-SP
│  │  └─ README
│  │     ├─ ar
│  │     ├─ en
│  │     ├─ fr
│  │     ├─ hi
│  │     └─ ru
│  ├─ experiment_analysis.ipynb  # analysis llm generation, which mainly includes calculating ρ, adj_acc, exa_acc and rmse. 
│  ├─ cefr_estimator_choose.ipynb  # choosing optimal cefr-estimator for CEFR-SP corpus
│  ├─ llm_infer_zero-shot_dp-planner_CoT.py  # llm inference codes using policy planned by dp and CoT generation using semantic-aware exemplar selection
│  ├─ utils.py  # prompt description and dp-algorithm
```

## 2. Run Inference using DP-planner+CoT generation
```python
python src/llm_infer_zero-shot_dp-planner_CoT.py --infer_bs 5 --case_num 3 --model_name /path/to/Llama-3.1-8B-Instruct --save_dir zero-shot_cefrsp --corpus CEFR-SP
```
The sentences simplified by the LLM will be saved in [LLMGeneration](src%2FLLMGeneration). Among them, `llm_gene_CEFR1, llm_gene_CEFR2 and llm_gene_CEFR3` represent the generations specified for the `A1, A2 and B1` CEFR-levels. 


## 3. Auto-Evaluation


