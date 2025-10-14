# final_project - Automatic Ticket Classification using Many-to-One RNN & Generative AI

## 📘 Project Overview
Organizations receive thousands of customer support tickets daily, making manual triage time-consuming and error-prone.  
This project automates the **classification** of customer support tickets into their respective **departments (queues)** using a **Many-to-One LSTM model**, and generates an **empathetic AI-powered reply** to the customer using **Google Gemini API**.

By combining **Deep Learning (RNN/LSTM)** with **Generative AI**, the system helps organizations route tickets faster and improve customer satisfaction through instant, polite responses.

---

## 🎯 Objectives
- **Automatically classify** support tickets into correct departments such as:
  - Billing and Payments  
  - Technical Support  
  - Customer Service  
  - Returns and Exchanges  
  - etc.
- **Generate automated responses** to acknowledge customer issues.
- **Improve operational efficiency** by reducing manual routing effort.

---

## 💡 Business Use Cases
| Use Case | Description |
|-----------|--------------|
| **Customer Support Automation** | Instantly routes customer queries to the right team. |
| **Reduced Resolution Time** | Automatically drafts acknowledgment responses. |
| **Cost Optimization** | Minimizes manual triage overhead. |
| **Customer Satisfaction** | Sends empathetic, AI-generated responses quickly. |

---

## 🧩 Technical Stack
| Category | Technology Used |
|-----------|-----------------|
| **Programming Language** | Python |
| **Libraries** | TensorFlow / Keras, NumPy, Pandas, Matplotlib, Scikit-learn |
| **Deep Learning** | Bidirectional LSTM (Many-to-One RNN) |
| **Generative AI** | Google Gemini API |
| **Dataset Source** | Hugging Face → Tobi-Bueck/customer-support-tickets |
| **IDE/Platform** | Jupyter Notebook / Google Colab |

---

## 📊 Dataset Details
**Source:** [Hugging Face – Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)

| Field | Description |
|--------|--------------|
| `body` | The main text content of the support ticket |
| `queue` | Department/Category label (target variable) |
| `priority`, `tags`, `subject` | Additional optional fields |

The project uses the **English-only subset** (`english_only_tickets.csv`) containing **33,000+** tickets.

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Load and clean text data  
- Remove rare classes (with <2 samples)  
- Tokenize and pad sequences  
- Label encode target classes  

### 2️⃣ Model Development
- Build a **Many-to-One LSTM model** using TensorFlow/Keras  
- Input: Customer ticket text  
- Output: Predicted department (queue)

### 3️⃣ Model Evaluation
- Metrics:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix
- Visualized training and validation performance

### 4️⃣ Generative AI Integration
- Once the ticket queue is predicted, the ticket text + predicted queue is passed to **Google Gemini API**.
- Gemini generates an **empathetic, context-aware reply** acknowledging the issue.

Example reply:

> _“Dear Customer,  
> Thank you for reaching out regarding your technical issue.  
> Our Technical Support team has received your ticket and will assist you shortly.  
> We appreciate your patience and understanding.”_

---

## 🧠 Model Architecture


Embedding (vocab_size=50,000, embedding_dim=128)
→ Bidirectional LSTM (units=128)
→ Dense (64, ReLU)
→ Dropout (0.3)
→ Dense (num_classes, Softmax)




---

## 📈 Results Summary
| Metric | Score |
|---------|--------|
| Training Accuracy | ~68% |
| Validation Accuracy | ~55% |
| Loss | ~1.53 |
| Classes | 10 (after filtering rare classes) |

> The model successfully distinguishes between major departments and provides a solid baseline for integrating NLP + GenAI.

---

## 🧮 Example Inference
```python
text = "I was charged twice for my subscription this month."
prediction = predict_queue(text)
print(prediction)

