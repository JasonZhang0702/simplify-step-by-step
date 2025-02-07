# simplify-step-by-step
following information is used for learning about *simplify-step-by-step* project

### project structure
```
simplify-step-by-step
├─ data  # storing raw data which labeled cefr-level
├─ img  # visual autto-evaluation metric comparison
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
│  ├─ experiment_analysis.ipynb  # analysis llm generation
│  ├─ llm_infer_few-shot_cefrsp_dpagent_expert.py  # llm inference codes using agent policy and expert guided prompt
│  ├─ utils.py  # 
```
