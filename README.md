# Multi Media Analytics - UvA 2025

## Snellius
### Connect to Snellius
```
ssh <user>@snellius.surf.nl 
```

### Setup project
```
cd MMA
python -m venv .venv
source .venv/bin/activate
pip install -r ../all_requirements.txt
```


### Access compute node
```
srun --partition=gpu_a100 --gpus=4 --ntasks=1 --cpus-per-task=72 --time=00:20:00 --pty bash -i
```

### Run server on Snellius
On the root directory of the project run:
```
python app/main.py
```

### Connect to server on your local machine
```
ssh -L 8050:127.0.0.1:8050 -J <user>@snellius.surf.nl <user>@<node hostname>
```


After the Dash server is running open http://127.0.0.1:8050/ on your browser.


## Plotly and Dash tutorials
- Dash in 20 minutes: https://dash.plotly.com/tutorial
- Plotly plots gallery: https://plotly.com/python/

## Snellius tutorials
- Basics: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html
- Resources: https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting


## MMA Metrics Data Cheatsheet

#### Setup

Make sure you have imported and connected:

```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'app')))

print(sys.path)
from db.database import Database
import uuid

db = Database()
db.connect()
```

---

#### Metric Overview

| Metric Type       | Table Name              | What It Measures                                    |
| ----------------- | ----------------------- | --------------------------------------------------- |
| **BERTScore**     | `bertscore_metrics`     | Novelty of current prompt vs previous               |
| **Guidance**      | `guidance_metrics`      | Prompt/image guidance values used during generation |
| **LPIPS**         | `lpips_metrics`         | Visual change between images                        |
| **Functionality** | `functionality_metrics` | Suggestion/enhancement usage stats                  |
| **Prompt Words**  | `prompt_word_metrics`   | Word-level analysis of prompts                      |

---

#### Fetch All Metrics

```python
df_bert = db.fetch_all_bertscore_metrics()
df_guidance = db.fetch_all_guidance_metrics()
df_lpips = db.fetch_all_lpips_metrics()
df_func = db.fetch_all_functionality_metrics()
df_words = db.fetch_all_prompt_word_metrics()
```

---

#### Fetch by User

```python
user_id = 1

df_bert = db.fetch_bertscore_by_user(user_id)
df_guidance = db.fetch_guidance_by_user(user_id)
df_lpips = db.fetch_lpips_by_user(user_id)
df_func = db.fetch_functionality_by_user(user_id)
df_words = db.fetch_prompt_word_metrics_by_user(user_id)
```

---

#### Fetch by Chat

```python
chat_id = 2

df_bert = db.fetch_bertscore_by_chat(chat_id)
df_guidance = db.fetch_guidance_by_chat(chat_id)
df_lpips = db.fetch_lpips_by_chat(chat_id)
df_func = db.fetch_functionality_by_chat(chat_id)
df_words = db.fetch_prompt_word_metrics_by_chat(chat_id)
```

---

#### Column Hints

##### `bertscore_metrics`

* `bert_novelty`: float (0–1)
* `prompt_id`, `previous_prompt_id`

##### `guidance_metrics`

* `prompt_guidance`, `image_guidance`: floats
* `depth`, `prompt_id`

##### `lpips_metrics`

* `lpips`: float (0–1, lower = more similar)
* `image_id`, `previous_image_id`

##### `functionality_metrics`

* `used_suggestion_pct`, `used_enhancement_pct`, `used_both_pct`, `no_ai_pct`: % as floats
* One row per (user, chat)

##### `prompt_word_metrics`

* `full_text`: full prompt
* `relevant_words`: comma-separated words
* `word_count`: number of words after stopword removal
