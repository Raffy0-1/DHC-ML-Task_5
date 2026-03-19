# DHC-ML-Task_5
> Mental Health Chatbot with DistilGPT2: A fine-tuned DistilGPT2 model that acts as an empathetic mental health chatbot, generating responses to user inputs categorized by emotion. Built using the Hugging Face Transformers library and Empathetic Dialogues dataset.

---

# Mental Health Support Chatbot

## 1. Task Objective

The primary objective of this project is to develop and fine-tune an empathetic conversational AI chatbot capable of providing initial support and listening to users experiencing various mental health challenges. The chatbot aims to engage in natural language conversations, understand user emotions, and respond with empathetic and supportive messages. This project serves as a foundational step towards creating more accessible and immediate mental health resources.

---

## 2. Dataset Used

The project utilizes a refined subset of the **Empathetic Dialogues Dataset**.

### Original Dataset Characteristics:
- Contains dialogues categorized by various emotions.
- Provides a rich source of human-bot interactions focusing on empathetic responses.

### Preprocessing and Filtering:
To ensure the dataset is highly relevant for mental health support and to improve model performance, a rigorous cleaning and filtering pipeline was applied:

- **Emotion Filtering**: Only dialogues expressing mental health-relevant emotions such as `sad`, `lonely`, `afraid`, `terrified`, `anxious`, `devastated`, `apprehensive`, `ashamed`, `guilty`, `disappointed`, `angry`, `furious`, `embarrassed`, and `annoyed` were retained.
- **Turn Extraction**: The first turn of each conversation (user's initial message and the bot's first reply) was extracted to focus on initial empathetic responses.
- **Quality Control - Empathy Scoring**: Bot replies were analyzed for the presence of a curated list of empathy-indicating keywords (e.g., `sorry`, `understand`, `feel`, `difficult`, `support`, `listen`). Replies with at least one empathy keyword were prioritized.
- **Quality Control - Bad Phrases**: Bot replies containing common casual phrases (e.g., `lol`, `wtf`, `huh?`) were filtered out to maintain a professional and empathetic tone.
- **Length Filtering**: Both user messages and bot replies were filtered based on minimum character and word counts to ensure sufficient context and meaningful responses.

After preprocessing, the dataset was reduced to **2,530 clean samples**, each containing a user query and a corresponding empathetic bot reply, formatted for direct model training.

---

## 3. Models Applied

### Base Model: `distilgpt2`

The project employs `distilgpt2`, a distilled version of GPT-2, as its base language model. `distilgpt2` offers a good balance between performance and computational efficiency, making it suitable for fine-tuning on custom datasets with limited resources.

### Fine-tuning Approach:
- **Causal Language Modeling (CLM)**: The model was fine-tuned for CLM, where it learns to predict the next token in a sequence — ideal for generative conversational agents.
- **Tokenizer**: The `AutoTokenizer` from Hugging Face was used with `distilgpt2`, and the `pad_token` was explicitly set to `eos_token` for consistent handling of variable-length sequences.
- **Training Parameters (`TrainingArguments`)**:
  - `per_device_train_batch_size=2`, `per_device_eval_batch_size=2`
  - `num_train_epochs=2`
  - `learning_rate=5e-5`, `weight_decay=0.01`, `warmup_steps=100`
  - `eval_strategy="steps"`, `eval_steps=200`, `save_steps=200`
  - `save_total_limit=2`, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`
- **Data Collator**: `DataCollatorForLanguageModeling` with `mlm=False`.
- **Early Stopping**: `EarlyStoppingCallback` with `patience=3` to prevent overfitting.

### Response Generation Configuration:
Custom `GenerationConfig` parameters were applied to control the quality and style of generated responses:

| Parameter | Value |
|---|---|
| `max_new_tokens` | 60 |
| `do_sample` | True |
| `temperature` | 0.6 |
| `top_k` | 40 |
| `top_p` | 0.85 |
| `repetition_penalty` | 1.4 |
| `no_repeat_ngram_size` | 3 |

---

## 4. Project Structure

```
DHC-ML-Task_5/
│
├── 📓 Mental_Health_Chatbot_Main.ipynb   # Main notebook — includes data preprocessing,
│                                         # model training, CLI chatbot & Streamlit UI
├── 📓 Failed_Mental_Health_Chatbot.ipynb # Earlier iteration — kept for transparency
│
├── 📁 Results/                           # Training results and outputs
├── 📄 requirements.txt                   # Python dependencies
├── 📄 README.md
├── 📄 .gitignore
└── 📄 LICENSE
```

> ⚠️ **Note:** Model files (`model.safetensors`, `config.json`, `tokenizer.json`, etc.) are **not included** in this repository due to their large size. The model is saved externally on Google Drive.

---

## 5. How to Run

Everything is contained inside `Mental_Health_Chatbot_Main.ipynb`. Open it in **Google Colab** and run the sections you need.

### Prerequisites
Make sure your fine-tuned model is saved in Google Drive at:
```
/content/drive/MyDrive/MentalHealthBot/model_final
```

### Install Dependencies
```bash
pip install streamlit transformers torch pyngrok
```

### Option A — Streamlit Web UI (Recommended)
Run the **Streamlit UI** section of the notebook. It will:
1. Mount your Google Drive and load the model
2. Launch a Streamlit app on port 8501
3. Use `pyngrok` to generate a public URL you can open in any browser

```python
from pyngrok import ngrok
import subprocess

ngrok.set_auth_token("your_ngrok_token_here")  # get free token at ngrok.com
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])

public_url = ngrok.connect(8501)
print(f"App is live at: {public_url}")
```

### Option B — Command Line Interface (CLI)
Run the **CLI** section of the notebook. Type your message and press Enter to chat. Type `quit` or `exit` to end the session.

---

## 6. Key Results and Findings

### Model Performance:
- The fine-tuned `distilgpt2` model successfully learned to generate contextually relevant and empathetic responses.
- Training metrics (loss, evaluation loss) indicated effective learning and convergence during fine-tuning.

### Chatbot Capabilities:
- **Empathetic Responding**: Acknowledges user emotions and offers supportive statements aligned with the training data.
- **Coherent Conversation**: Maintains reasonable coherence and addresses the user's input directly.
- **Reduced Repetition**: Generation parameters (`repetition_penalty`, `no_repeat_ngram_size`) produce more varied replies.

### Observed Challenges and Future Work:
- **Generality of Responses**: Responses can sometimes be generic, indicating a need for more nuanced understanding.
- **Depth of Empathy**: Limited by model size; a larger model (e.g., DialoGPT-medium) could provide richer responses.
- **Lack of Proactive Support**: The current model is reactive. Future iterations could explore proactive suggestions or resource recommendations.
- **Data Diversity**: Expanding the training dataset with more diverse mental health dialogues could significantly improve robustness.

### Conclusion:
This project successfully establishes a functional empathetic chatbot as a proof-of-concept for AI in mental health support. The results highlight the potential of fine-tuned language models while pointing to clear avenues for further improvement in response sophistication, emotional intelligence, and data enrichment.

---

## 7. Dependencies

```
streamlit
transformers
torch
pyngrok
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 8. Disclaimer

> This chatbot is a research/learning project and is **not a substitute for professional mental health care**. If you or someone you know is in crisis, please reach out to a licensed mental health professional or a crisis helpline.
