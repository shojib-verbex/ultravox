# Evaluation Datasets Integration TODO List

This document tracks the sequential addition of speech LLM evaluation datasets to the Ultravox evaluation framework.

**Scope**: Freely accessible datasets only (no LDC membership, no registration-required datasets)

---

## Phase 1: Verify Existing Datasets

- [ ] **1.1 VoiceBench** - Verify all 11 subsets are configured
  - Path: `hlt-lab/voicebench`
  - Config: `ultravox/data/configs/voicebench.py`
  - Subsets: bbh, mmsu, openbookqa, sd-qa, advbench, alpacaeval, alpacaeval-full, alpacaeval-speaker, commoneval, ifeval, mtbench, wildvoice

- [ ] **1.2 Big Bench Audio** - Verify configuration
  - Path: `fixie-ai/big_bench_audio`
  - Config: `ultravox/data/configs/bigbenchaudio.py`

- [ ] **1.3 AMI Meeting Corpus** - Verify configuration
  - Path: Check existing config
  - Config: `ultravox/data/configs/ami.py`

- [ ] **1.4 AudioBench** - Verify all available subsets
  - Path: `AudioLLMs/*` organization
  - Config: `ultravox/data/configs/audiobench.py`
  - Note: 50+ datasets available, currently only 4 configured

---

## Phase 2: Add Fully Open Datasets (No Registration)

- [ ] **2.1 HeySQuAD**
  - HuggingFace: `yijingwu/HeySQuAD_human`, `yijingwu/HeySQuAD_machine`
  - License: CC-BY 4.0
  - Samples: 76k human, 97k machine
  - Task: Spoken QA (SQuAD-based)
  - Metric: F1 / Exact Match
  - Config to create: `ultravox/data/configs/heysquad.py`

- [ ] **2.2 MMAU (Massive Multi-Task Audio Understanding)**
  - HuggingFace: `AudioLLMs/MMAU-mini`
  - License: Apache 2.0
  - Samples: 1k (mini) with answers
  - Tasks: 27 skills (speech, sound, music)
  - Metric: Task accuracy
  - Config to create: `ultravox/data/configs/mmau.py`

- [ ] **2.3 Dynamic-SUPERB** (selected tasks)
  - HuggingFace: `DynamicSuperb/*` organization
  - License: CC-BY 4.0
  - Tasks: Select from 180 available tasks
  - Config to create: `ultravox/data/configs/dynamic_superb.py`

---

## Phase 3: Add Open Datasets with NC Restriction

- [ ] **3.1 AIR-Bench**
  - HuggingFace: `qyang1021/AIR-Bench-Dataset`
  - License: CC-BY-NC-4.0
  - Tasks: 19 foundation + 2k open-ended QA
  - Config to create: `ultravox/data/configs/airbench.py`

- [ ] **3.2 SD-Eval**
  - HuggingFace: `amphion/SD-Eval`
  - License: CC-BY-NC-4.0
  - Samples: 7,303 utterances
  - Tasks: Emotion, accent, age, background sound
  - Config to create: `ultravox/data/configs/sdeval.py`

- [ ] **3.3 SLURP**
  - HuggingFace: `qmeeus/slurp`
  - License: CC-BY-4.0 (text), CC-BY-NC-4.0 (audio)
  - Samples: 72k utterances, 18 domains
  - Task: Spoken language understanding
  - Config to create: `ultravox/data/configs/slurp.py`

- [ ] **3.4 SpokenWOZ**
  - HuggingFace: `ssz1111/SpokenWOZ-Train-Audio`
  - License: CC-BY-NC-4.0
  - Samples: 249 hours, 5.7k dialogues
  - Task: Task-oriented dialogue
  - Config to create: `ultravox/data/configs/spokenwoz.py`

---

## Phase 4: Metrics Implementation (As Needed)

- [ ] **4.1 Implement F1 / Exact Match metric** for HeySQuAD
- [ ] **4.2 Implement intent/slot accuracy** for SLURP
- [ ] **4.3 Implement emotion accuracy** for SD-Eval
- [ ] **4.4 Implement multi-task metrics** for MMAU, Dynamic-SUPERB

---

## Phase 5: Update Evaluation Config

- [ ] **5.1 Update `eval_config.yaml`** with new dataset entries
- [ ] **5.2 Create specialized eval configs** (e.g., `eval_config_comprehensive.yaml`)

---

## Implementation Template

For each dataset, follow these steps:

### Step 1: Create Config File
```python
# ultravox/data/configs/{dataset_name}.py
from ultravox.data import types

CONFIG = types.DatasetConfig(
    name="{dataset-name}",
    path="{huggingface-path}",
    subset="{optional-subset}",
    splits=[
        types.DatasetSplitConfig(name="test", num_samples=N, split=types.DatasetSplit.TEST),
    ],
    user_template=types.QA_USER_TEMPLATE,
    transcript_template="{{audio_field}}",
    assistant_template="{{answer_field}}",
    eval_config=types.EvalConfig(metric="{metric_name}"),
)

configs = [CONFIG]
```

### Step 2: Register Dataset
```python
# ultravox/data/registry.py
from ultravox.data.configs import {dataset_name}
# ...
register_datasets({dataset_name}.configs)
```

### Step 3: Add to Eval Config
```yaml
# ultravox/evaluation/configs/eval_config.yaml
eval_sets:
  - name: {dataset-name}
```

### Step 4: Verify
```bash
# Test loading
python -c "from ultravox.data import registry; ds = registry.create_dataset('{dataset-name}', registry.types.EvalDatasetArgs()); print(len(ds))"

# Run small eval
python -m ultravox.evaluation.eval --config-path ultravox/evaluation/configs/eval_config.yaml --eval-dataset-args.max_samples 10
```

---

## Notes

### Datasets to Add (All Freely Accessible)

| Priority | Dataset | HuggingFace Path | Samples |
|----------|---------|------------------|---------|
| High | HeySQuAD | `yijingwu/HeySQuAD_human` | 76k |
| High | MMAU | `AudioLLMs/MMAU-mini` | 1k |
| High | Dynamic-SUPERB | `DynamicSuperb/*` | varies |
| Medium | AIR-Bench | `qyang1021/AIR-Bench-Dataset` | varies |
| Medium | SD-Eval | `amphion/SD-Eval` | 7.3k |
| Medium | SLURP | `qmeeus/slurp` | 72k |
| Medium | SpokenWOZ | `ssz1111/SpokenWOZ-Train-Audio` | 203k turns |

### Datasets Already in Codebase
- VoiceBench (12 subsets) - `hlt-lab/voicebench`
- Big Bench Audio - `fixie-ai/big_bench_audio`
- AudioBench (4 subsets) - `fixie-ai/*`
- AMI - Meeting corpus
- LibriSpeech - ASR
- CommonVoice - ASR (multiple languages)
- CoVoST2 - Speech translation
- FLEURS - Multilingual
- GigaSpeech - Large-scale ASR
- VoxPopuli - Multilingual speech
- WeNetSpeech - Chinese ASR
