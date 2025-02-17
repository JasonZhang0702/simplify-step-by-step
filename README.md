# simplify-step-by-step
This repository is used to reproduce the experimental results of
_**Let’s Simplify Step by Step: Guiding LLM Towards Multilingual Zero-shot Readability-Controlled Sentence Simplification**_, 
and meanwhile, it supplements the details that are not described clearly in the paper. I hope it can help and inspire you.

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




