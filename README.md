# EV ANN: Offline Transformer-based Chatbot (Eve)

This project implements **Eve**, an offline LLM-based chatbot optimized for edge computing on the **NVIDIA Jetson Nano**. It features real-time inference, custom persona loading, and advanced memory optimization.

Eve is an offline Transformer-based chatbot designed for local inference with privacy. 
It supports TTS using pyttsx3 and manages chat context efficiently. 
The project is CPU-friendly and can be extended for multilingual and emotion-aware responses.

## 🚀 Key Technical Features & Optimizations
The following optimizations were implemented:

### 1. CUDA-Accelerated Inference
* **Problem:** Standard CPU execution was too slow for real-time conversation.
* **Solution:** Integrated a CUDA-check mechanism to offload computations to the Jetson Nano’s Maxwell GPU.
* **Result:** Significant reduction in latency and smoother response generation.

### 2. Memory Optimization (4-bit/8-bit Precision)
* **Problem:** Large Language Models (LLMs) often exceed the 4GB/8GB RAM limits of edge devices.
* **Solution:** Implemented **Precision Mode Switching**. Used mixed-precision (4-bit quantization) to shrink the model's memory footprint without losing conversational quality.


### 3. Robust Character Management
* **Problem:** File path errors in loading the `Eve.json` persona file caused system crashes.
* **Solution:** Added a robust Exception Handling layer. If the character file is missing, the system automatically falls back to default persona values to ensure 100% uptime.

##  Tech Stack
* **Language:** Python 3.x
* **Frameworks:** PyTorch (v2.6.0+), Transformers
* **Optimization:** CUDA, 4-bit Quantization

## 📄 Documentation
For a deep dive into the debugging process and performance metrics, please refer to the:
👉 **[Full Error & Resolution Report](./SOLVE ERROR REPORT.pdf)**
