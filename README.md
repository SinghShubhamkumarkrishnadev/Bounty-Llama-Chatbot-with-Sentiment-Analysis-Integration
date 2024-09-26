# Bounty-Llama-Chatbot-with-Sentiment-Analysis-Integration

---

### **📚 Educational Tutor Chatbot with Sentiment Analysis**
---

#### **💡 Project Overview**

This project demonstrates a chatbot designed to interact with users in an educational context. The chatbot uses **Llama 2** for generating conversational responses and incorporates **DistilBERT** to analyze the sentiment of user input. Based on the sentiment analysis, the chatbot adjusts its responses, offering helpful, positive, or encouraging feedback, enhancing the user's learning experience.

The goal is to create a tutor that dynamically adapts its teaching style based on the user's mood, providing personalized support. The chatbot is designed to generate responses in a coherent, diverse, and sentiment-aware manner, ensuring a more engaging and productive learning environment.

## **Code Structure:**

The project is divided into several key sections:
1. **Installation of Dependencies**
2. **Model Loading**
3. **Response Generation**
4. **Sentiment-based Response Adjustment**
5. **Response Cleaning**
6. **Integration and Gradio Interface**
7. **Launching the Chatbot**

---

## 📌 visit this link to go to my google colab 
https://colab.research.google.com/drive/1nMgsrk6bOrfO2INxTxcdeNUawMD0MKuL?usp=sharing


## Must Watch Demo Video 
<a href="https://youtu.be/WDKuvBKNj-g">
    <img src="https://github.com/SinghShubhamkumarkrishnadev/Bounty-Llama-Chatbot-with-Sentiment-Analysis-Integration/blob/main/videos.png" width="400" height="300" alt="Watch the video">
</a>


## **1. Installation of Dependencies**
This section installs all the necessary libraries such as `torch`, `transformers`, `gradio`, and others required for model loading, text generation, and sentiment analysis.

```python
!pip install torch torchvision torchaudio transformers sentencepiece gradio sentence-transformers accelerate
```

---

## **2. Model Loading**
We load two models:
- **Sentiment Analysis Model**: `DistilBERT` to classify the sentiment of user input as positive or negative.
- **Llama 2 Model**: `Llama-2-7b-chat-hf` for generating tutor responses.

```python
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Load sentiment analysis model
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("Sentiment analysis model loaded successfully.")
except Exception as e:
    print(f"Error loading sentiment analysis model: {str(e)}")

# Load Llama 2 model and tokenizer
try:
    model_id = "NousResearch/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    llama_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        torch_dtype=torch.float16, device_map="auto",
        max_length=3072, 
        do_sample=True, temperature=0.6, top_p=0.9
    )
    print("Llama 2 model loaded successfully.")
except Exception as e:
    print(f"Error loading Llama model: {str(e)}")
```

---

## **3. Response Generation**
This function uses the Llama 2 model to generate a response based on the user's input. We adjust the response length dynamically to provide more detailed answers when needed.

```python
# Function to generate Llama responses
def generate_llama_response(user_input):
    try:
        input_length = len(user_input.split())
        adjusted_max_length = min(3072, input_length * 6 + 150)
        response = llama_pipeline(user_input, max_length=adjusted_max_length, do_sample=True)[0]['generated_text']
        return response.strip()
    except Exception as e:
        return f"Error generating Llama response: {str(e)}"
```

---

## **4. Sentiment-based Response Adjustment**
We use sentiment analysis to tailor responses:
- If the user expresses negative sentiment, the chatbot offers a more encouraging or simplified explanation.
- If the sentiment is positive, the chatbot offers positive reinforcement.

```python
# Expanded sentiment-based response function with nuanced feedback
def sentiment_based_tutor_response(user_input):
    try:
        sentiment = sentiment_pipeline(user_input)[0]
        label = sentiment['label']
        score = sentiment['score']

        # More nuanced responses
        if label == "NEGATIVE":
            if score > 0.9:
                pre_message = "It seems you're feeling quite frustrated. Let me take a different approach."
            else:
                pre_message = "It seems you're having some difficulty. Let me try explaining it more simply."
        elif label == "POSITIVE":
            if score > 0.9:
                pre_message = "Great! You're really on top of this. Let's dive deeper."
            else:
                pre_message = "You're doing well. Let's move to the next step."
        else:
            pre_message = "Let's take it step by step and figure it out together."

        return pre_message
    except Exception as e:
        return f"Error in sentiment analysis: {str(e)}"
```

---

## **5. Response Cleaning**
The chatbot's response might include redundant phrases. This section filters out these phrases and removes sentences that are too similar to the user’s input to ensure a cleaner, more useful reply.

```python
import difflib

# Clean redundant or overly similar responses
def clean_response(user_input, llama_response):
    redundant_phrases = ["Let me try explaining it more simply.", "It seems you're having some difficulty."]
    for phrase in redundant_phrases:
        llama_response = llama_response.replace(phrase, "").strip()

    response_sentences = llama_response.split('. ')
    cleaned_sentences = []
    for sentence in response_sentences:
        similarity = difflib.SequenceMatcher(None, user_input, sentence).ratio()
        if similarity < 0.7 and len(sentence.split()) > 3:
            cleaned_sentences.append(sentence)

    cleaned_response = '. '.join(cleaned_sentences).strip()
    return cleaned_response if len(cleaned_response) > 50 else llama_response
```

---

## **6. Integration of Sentiment and Llama Responses**
This function integrates sentiment-based adjustment and the response generation. It combines the sentiment analysis and the generated response into a single cohesive output.

```python
# Main function: sentiment + chatbot responses
def tutor_with_sentiment(user_input):
    sentiment_message = sentiment_based_tutor_response(user_input)
    llama_response = generate_llama_response(user_input)
    cleaned_response = clean_response(user_input, llama_response)
    return f"{sentiment_message} {cleaned_response}".strip()
```

---

## **7. Gradio Interface for User Interaction**
A Gradio interface is used to interact with the chatbot. It includes a feature for clearing chat history when the user types "reset".

```python
import gradio as gr

# Edge-case handling: Function to handle non-queries or random input
def handle_edge_cases(user_input):
    if len(user_input.strip()) == 0:
        return "It looks like you haven't entered anything. Could you please provide more details?"

    if len(user_input.split()) < 2:
        return "Can you elaborate a little more? I'm here to help with detailed questions."

    return None 

chat_history = []

def gradio_tutor_interface(question):
    try:
        if question.lower() == "reset":
            global chat_history
            chat_history = []  
            return "Chat history has been reset."
        
        edge_case_response = handle_edge_cases(question)
        if edge_case_response:
            return edge_case_response

        response = tutor_with_sentiment(question)
        chat_history.append(f"User: {question}\nBot: {response}")
        return "\n\n".join(chat_history)
    except Exception as e:
        return f"Error: {str(e)}"
```

---

## **8. Launching the Chatbot**
Finally, we launch the Gradio interface, allowing the chatbot to be accessed and tested interactively.

```python
# Gradio interface setup with reset and title
interface = gr.Interface(
    fn=gradio_tutor_interface,
    inputs="text",
    outputs="text",
    title="Educational Tutor Chatbot with Sentiment Analysis and Improved Handling",
    description="Ask a question, and the tutor adjusts its response based on your sentiment. Type 'reset' to clear chat history.",
    allow_flagging="never"
)

# Launch Gradio
interface.launch(debug=True)
```


#### **🔑 Objectives**
The primary objectives of this project are:
1. **Sentiment-Driven Response Adaptation:** 
   - The chatbot uses **sentiment analysis** from Hugging Face's `distilbert-base-uncased-finetuned-sst-2-english` model to determine if the user is positive, negative, or neutral.
   - Based on the sentiment, the chatbot adjusts its responses to ensure a more engaging, empathetic, and supportive interaction.

2. **Enhanced Educational Tutoring Experience:**
   - The chatbot is positioned as an educational tutor that helps users by providing responses tailored to their emotional state.
   - Negative sentiment results in simpler explanations, positive sentiment encourages further steps, and neutral sentiment invites collaborative learning.

3. **Seamless Integration of Llama 2 Model:**
   - The chatbot leverages the **Llama 2 language model** from Hugging Face (`NousResearch/Llama-2-7b-chat-hf`) for generating coherent, informative, and contextually appropriate responses to user queries.
   - The text generation pipeline is fine-tuned to provide **longer, more detailed answers** to complex questions.

4. **Interactive User Experience:**
   - The chatbot provides a smooth interaction experience through **Gradio**, a user-friendly interface that allows for real-time queries and responses. 
   - Users can ask any question, and the chatbot responds by adjusting its behavior based on the sentiment and context.

---

### **🧠 Models and Techniques Used**

1. **Sentiment Analysis Model:**
   - The **`distilbert-base-uncased-finetuned-sst-2-english`** model from Hugging Face is utilized to detect whether the user input is **positive**, **negative**, or **neutral**.
   - This model is a lightweight version of BERT, fine-tuned specifically for sentiment classification tasks.

2. **Text Generation (Llama 2 Model):**
   - The chatbot uses the **`NousResearch/Llama-2-7b-chat-hf`** model, which is a **powerful large language model** capable of generating human-like responses.
   - This model is integrated with a text generation pipeline that allows us to configure parameters such as **maximum length**, **temperature**, and **top-p** for more **diverse, coherent, and longer responses**.

3. **Response Cleaning:**
   - After generating the text, the response goes through a **cleaning process** where redundant or overly similar parts of the text are removed to maintain clarity.
   - The system ensures that no repeated phrases or overly generic statements are included in the response, keeping the interaction relevant and concise.

---

### **🛠 Implementation Details**

1. **Sentiment Analysis Integration:**
   - Every time a user inputs a question, the chatbot first performs **sentiment analysis** to gauge the user’s emotional tone. 
   - The sentiment labels are then used to adjust the response. For example:
     - **Negative sentiment:** The chatbot simplifies its explanations.
     - **Positive sentiment:** The chatbot offers positive reinforcement and encourages the user.
     - **Neutral sentiment:** The chatbot adopts a balanced, step-by-step approach to problem-solving.

2. **Dynamic Text Generation:**
   - Using the Llama 2 model, the chatbot generates a response based on the user’s input. 
   - The **length of the response** is dynamically adjusted based on the input size, ensuring that complex queries receive **longer, more detailed answers**.
   - We also tuned the generation parameters to produce **more coherent and diverse** responses.

3. **Interactive Gradio Interface:**
   - The chatbot is deployed using **Gradio**, which provides a **simple yet interactive interface** where users can enter questions and receive real-time responses.
   - Users can also type "reset" to clear the conversation history and start fresh.

---

### **🔄 How Sentiment Impacts Responses**

1. **Negative Sentiment:**
   - When negative sentiment is detected (e.g., frustration or confusion), the chatbot responds with:
     - **"It seems you're having some difficulty. Let me try explaining it more simply."**
   - This encourages the user while also providing a more accessible, simplified explanation to help them understand better.

2. **Positive Sentiment:**
   - When the user's input is positive, the chatbot reinforces this positivity by saying:
     - **"Great! You're doing well. Let's move to the next step."**
   - This helps keep the user motivated and encourages them to continue learning.

3. **Neutral Sentiment:**
   - For neutral sentiment, the chatbot keeps a collaborative tone:
     - **"Let's take it step by step and figure it out together."**
   - This ensures a balanced approach, guiding the user step-by-step through their query.

---

### **🚀 How to Use the Chatbot**

1. **Ask a Question:**
   - You can type any educational or general question in the chat box. The chatbot will analyze your sentiment and generate a response accordingly.

2. **Reset Chat History:**
   - If you wish to start a new conversation or clear the history, simply type **"reset"** in the input field.

3. **Real-Time Interaction:**
   - The chatbot will provide real-time responses based on the sentiment of your query, generating detailed, coherent answers using Llama 2's text generation capabilities.

---

### **📈 Real-World Applicability**

This chatbot has numerous potential applications in education and tutoring:
- **Personalized Learning:** By adjusting its behavior based on student sentiment, it offers a more personalized and engaging learning experience.
- **Supportive Tutoring:** The chatbot provides emotional support and encouragement when a student is struggling, helping to reduce frustration and boost motivation.
- **Adaptive Responses:** With its ability to generate longer, detailed explanations, it can assist in explaining complex topics, making it a useful tool for educational support.

---

## **Conclusion**
This chatbot system showcases how AI models such as Llama 2 and DistilBERT can be integrated to create a dynamic and responsive educational tutor. By incorporating sentiment analysis, the chatbot is more empathetic and can adapt its responses to the user's emotional state, providing a more personalized and effective learning experience.
