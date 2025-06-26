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
pip install -r all_requirements.txt
```

To use the models from the Hugging Face Hub, create a token in your account, then create a `.env` file in the root directory of the project with the following content:
```
HUGGINGFACE_TOKEN=<your_huggingface_token>
```

### Access compute node

##### For the A100
```
srun --partition=gpu_a100 --gpus=4 --ntasks=1 --cpus-per-task=72 --time=00:30:00 --pty bash -i
```

##### For the H100
```
srun --partition=gpu_h100 --gpus=4 --ntasks=1 --cpus-per-task=64 --time=00:30:00 --pty bash -i
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

## Information regarding Database and the Metrics used

You can check out the database using the `understanding_database.ipynb` jupyter file.


#### Metric Overview

| Metric Type       | Table Name              | What It Measures                                    |
| ----------------- | ----------------------- | --------------------------------------------------- |
| **BERTScore**     | `bertscore_metrics`     | Novelty of current prompt vs previous               |
| **Guidance**      | `guidance_metrics`      | Prompt/image guidance values used during generation |
| **LPIPS**         | `lpips_metrics`         | Visual change between images                        |
| **Functionality** | `functionality_metrics` | Suggestion/enhancement usage stats                  |
| **Prompt Words**  | `prompt_word_metrics`   | Word-level analysis of prompts                      |

#### What is stored in the columns?

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