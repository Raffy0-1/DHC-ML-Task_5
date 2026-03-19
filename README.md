# DHC-ML-Task_5
"Mental Health Chatbot with DistilGPT2: A fine-tuned DistilGPT2 model that acts as an empathetic mental health chatbot, generating responses to user inputs categorized by emotion. Built using the Hugging Face Transformers library and Empathetic Dialogues dataset."
# Mental Health Support Chatbot

## 1. Task Objective

The primary objective of this project is to develop and fine-tune an empathetic conversational AI chatbot capable of providing initial support and listening to users experiencing various mental health challenges. The chatbot aims to engage in natural language conversations, understand user emotions, and respond with empathetic and supportive messages. This project serves as a foundational step towards creating more accessible and immediate mental health resources.

## 2. Dataset Used

The project utilizes a refined subset of the **Empathetic Dialogues Dataset**.

### Original Dataset Characteristics:
*   Contains dialogues categorized by various emotions.
*   Provides a rich source of human-bot interactions focusing on empathetic responses.

### Preprocessing and Filtering:
To ensure the dataset is highly relevant for mental health support and to improve model performance, a rigorous cleaning and filtering pipeline was applied:
*   **Emotion Filtering**: Only dialogues expressing mental health-relevant emotions such as 'sad', 'lonely', 'afraid', 'terrified', 'anxious', 'devastated', 'apprehensive', 'ashamed', 'guilty', 'disappointed', 'angry', 'furious', 'embarrassed', and 'annoyed' were retained.
*   **Turn Extraction**: The first turn of each conversation (user's initial message and the bot's first reply) was extracted to focus on initial empathetic responses.
*   **Quality Control - Empathy Scoring**: Bot replies were analyzed for the presence of a curated list of empathy-indicating keywords (e.g., 'sorry', 'understand', 'feel', 'difficult', 'support', 'listen'). Replies with at least one empathy keyword were prioritized.
*   **Quality Control - Bad Phrases**: Bot replies containing common 'bad' or casual phrases (e.g., 'lol', 'wtf', 'huh?') were filtered out to maintain a professional and empathetic tone.
*   **Length Filtering**: Both user messages and bot replies were filtered based on minimum character and word counts to ensure sufficient context and meaningful responses.

After preprocessing, the dataset was reduced to **2530 clean samples**, each containing a user query and a corresponding empathetic bot reply, formatted for direct model training.

## 3. Models Applied

### Base Model: `distilgpt2`

The project employs `distilgpt2`, a distilled version of GPT-2, as its base language model. `distilgpt2` offers a good balance between performance and computational efficiency, making it suitable for fine-tuning on custom datasets with limited resources.

### Fine-tuning Approach:
*   **Causal Language Modeling (CLM)**: The model was fine-tuned for CLM, where it learns to predict the next token in a sequence. This is ideal for generative conversational agents.
*   **Tokenizer**: The `AutoTokenizer` from Hugging Face was used with `distilgpt2`, and the `pad_token` was explicitly set to `eos_token` for consistent handling of variable-length sequences.
*   **Training Parameters (`TrainingArguments`)**:
    *   `per_device_train_batch_size=2`, `per_device_eval_batch_size=2`
    *   `num_train_epochs=2`
    *   `learning_rate=5e-5`, `weight_decay=0.01`, `warmup_steps=100`
    *   `eval_strategy="steps"`, `eval_steps=200`, `save_steps=200`
    *   `save_total_limit=2`, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`
*   **Data Collator**: `DataCollatorForLanguageModeling` with `mlm=False` was used.
*   **Early Stopping**: An `EarlyStoppingCallback` with `patience=3` was implemented to prevent overfitting and optimize training duration.

### Response Generation Configuration:
Custom `GenerationConfig` parameters were applied to control the quality and style of generated responses, including `max_new_tokens`, `do_sample`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, and `no_repeat_ngram_size`.

## 4. Key Results and Findings

### Model Performance:
*   The fine-tuned `distilgpt2` model successfully learned to generate contextually relevant and empathetic responses, demonstrating its ability to act as a conversational agent.
*   Training metrics (loss, evaluation loss) indicated effective learning and convergence during the fine-tuning process.

### Chatbot Capabilities:
*   **Empathetic Responding**: The chatbot can generate responses that acknowledge user emotions and offer supportive statements, aligned with the empathetic nature of the training data.
*   **Coherent Conversation**: It maintains a reasonable degree of coherence, following the conversation flow and addressing the user's input.
*   **Reduced Repetition**: Configuration of generation parameters, particularly `repetition_penalty` and `no_repeat_ngram_size`, helped to produce more varied and less repetitive replies.

### Observed Challenges and Future Work:
*   **Generality of Responses**: At times, the chatbot's responses can be somewhat generic or lead to simple clarification questions, indicating a need for more nuanced understanding and response generation.
*   **Depth of Empathy**: While empathetic, the model's ability to deeply understand complex emotional states and provide truly personalized support is limited by its size and the scope of the training data.
*   **Lack of Proactive Support**: The current model is reactive. Future iterations could explore mechanisms for proactive support or resource recommendation.
*   **Data Diversity**: Expanding the training dataset with more diverse and complex mental health dialogues could significantly enhance the bot's capabilities and robustness.

### Conclusion:
This project successfully established a functional empathetic chatbot, providing a strong proof-of-concept for AI in mental health support. The results highlight the potential of fine-tuned language models while also pointing to clear avenues for further improvement in areas such as response sophistication, emotional intelligence, and data enrichment.
