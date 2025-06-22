# AetherMindAI

**A Lightweight, Multifaceted AI Assistant for Google Colab**

## Overview

AetherMindAI is an open-source AI assistant designed to bring versatile capabilities to your Google Colab environment. Powered by lightweight, quantized language models like Microsoft's Phi-2, AetherMindAI offers a unique blend of conversational reasoning, advanced symbolic mathematics, and practical machine learning demonstrations.

Its modular architecture makes it easy to understand, extend, and integrate into educational projects, hobbyist explorations, or as a starting point for more complex AI applications. Whether you want to chat, solve complex equations, or experiment with ML model training, AetherMindAI provides a responsive and resource-friendly experience directly within your browser.

## Features

*   🧠 **Conversational AI:** Engage in natural language conversations, ask questions, and get informative responses powered by models like Phi-2.
*   🧮 **Symbolic Mathematics:** Solve algebraic equations, perform calculus (differentiation, integration, limits), and handle other symbolic math tasks via SymPy integration.
*   🤖 **Machine Learning Tasks:** Execute basic machine learning workflows, including classification, regression, and clustering, using scikit-learn with synthetic datasets.
*   🧱 **Modular Design:** Composed of distinct, understandable Python modules for chat, math, ML, and a central controller, promoting ease of modification and extension.
*   ☁️ **Colab Optimized:** Designed to run efficiently in Google Colab environments, leveraging its free compute resources and making AI experimentation accessible.
*   📝 **Task Routing:** Intelligently detects the type of user prompt (chat, math, or ML) and routes it to the appropriate specialized module.
*   💡 **Lightweight & Accessible:** Focuses on using smaller, quantized models to ensure it can run on standard Colab instances without requiring premium GPUs (for many tasks).
*   📚 **Educational & Hobbyist Friendly:** A great tool for learning about AI components, chatbot architecture, and integrating different AI functionalities.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/AetherMindAI/blob/main/AetherMindAI_Colab_Notebook.ipynb)
<!-- Replace YOUR_GITHUB_USERNAME and AetherMindAI_Colab_Notebook.ipynb with actual links once available -->

## Setup Instructions (Google Colab)

AetherMindAI is designed to be run primarily in a Google Colab environment.

1.  **Open in Colab:**
    *   Click the "Open In Colab" badge above.
    *   Alternatively, go to [Google Colab](https://colab.research.google.com/) and select `File > Open notebook > GitHub`, then enter the repository URL and choose the main project notebook (if one is provided, e.g., `AetherMindAI_Colab_Notebook.ipynb`) or upload individual `.py` files.

2.  **Clone the Repository (if not using a primary project notebook):**
    If you are setting up the environment manually in a new Colab notebook, you can clone the repository:
    ```bash
    !git clone https://github.com/YOUR_GITHUB_USERNAME/AetherMindAI.git
    %cd AetherMindAI
    ```
    *(Replace `YOUR_GITHUB_USERNAME` with the actual username or organization)*

3.  **Install Dependencies:**
    AetherMindAI relies on several Python libraries.
    *   If a `requirements.txt` file is provided in the repository:
        ```bash
        !pip install -r requirements.txt
        ```
    *   Alternatively, install them individually:
        ```bash
        !pip install transformers torch accelerate bitsandbytes sympy scikit-learn pandas numpy matplotlib
        ```
    *(Note: `bitsandbytes` is often used for quantization with Hugging Face models like Phi-2. `accelerate` helps with model loading on different devices.)*

4.  **Download Model (if not handled by scripts):**
    The `phi2_colab_runner.py` script attempts to download and load the Phi-2 model. Ensure your Colab instance has internet access.

5.  **Run AetherMindAI:**
    Once the setup is complete and all modules are in place, you can run the main controller:
    ```bash
    !python aethermind_main.py
    ```
    This will start the interactive chat loop in your Colab cell output.

## Core Modules

AetherMindAI is built with a modular approach. Here's a brief overview of the key components:

*   **`phi2_colab_runner.py`**:
    *   **Purpose:** Handles the core conversational AI capabilities.
    *   **Functionality:** Responsible for loading the pre-trained language model (e.g., Microsoft Phi-2) and its tokenizer. It includes functions to generate text responses based on user prompts and conversation history. This module is central to the chatbot's ability to understand and respond to natural language.

*   **`colab_math_solver.py`**:
    *   **Purpose:** Provides symbolic mathematics solving capabilities.
    *   **Functionality:** Integrates with the SymPy library to parse and solve mathematical queries expressed in natural language or as equations. It can handle tasks like solving algebraic equations, differentiation, integration, limits, and more.

*   **`colab_ml_tools.py`**:
    *   **Purpose:** Enables basic machine learning task execution.
    *   **Functionality:** Uses the scikit-learn library to perform common ML tasks such as classification (e.g., Logistic Regression), regression (e.g., Linear Regression), and clustering (e.g., KMeans). It operates on synthetically generated datasets, making it easy to demonstrate and experiment with ML concepts without needing external data. Includes options for basic plotting of results in Colab.

*   **`aethermind_main.py`**:
    *   **Purpose:** Acts as the central controller and task router.
    *   **Functionality:** Initializes the AetherMindAI application. It includes logic to detect the type of task a user is requesting (chat, math, or ML) based on their prompt. It then routes the request to the appropriate module (`phi2_colab_runner`, `colab_math_solver`, or `colab_ml_tools`) and presents the result to the user. It also manages the interactive chat loop and conversation history.

## Usage Examples

Once AetherMindAI is running, you can interact with it through the command-line interface in your Colab cell. Here are a few examples:

**1. General Conversation:**

```
You: Hello, AetherMind! Can you tell me about the Phi-2 language model?

[AetherMind Chat]: The Phi-2 language model is a Transformer-based model developed by Microsoft... (AetherMind provides a detailed explanation)
```

**2. Solving a Math Problem:**

```
You: Solve the equation x^2 + 2*x - 8 = 0 for x

[Math Solver]: The solutions for x in the equation x**2 + 2*x - 8 = 0 are: [-4, 2]
```

**3. Requesting a Machine Learning Task:**

```
You: Can you run a k-means clustering task for me?

[ML Tool: Clustering]: --- Clustering Task Summary ---
Data Parameters: {'n_samples': 150, 'random_state': 42, 'n_features': 2, 'centers': 3, 'cluster_std': 1.0}
Model Parameters: {'n_clusters': 3, 'random_state': 42}
Inertia: ...
A plot has been generated (if data dimensions were suitable).
(Note: Plotting might be disabled or text-based in some console versions; full plotting in `colab_ml_tools.py` if run directly)
```

**4. Asking for a different ML task (e.g., Classification):**

```
You: Perform a classification analysis.

[ML Tool: Classification]: --- Classification Task Summary ---
Data Parameters: {'n_samples': 150, 'random_state': 42, 'n_features': 4, 'n_classes': 2, 'n_informative': 2}
Model Parameters: {'random_state': 42}
Accuracy on Test Set: ...
A plot has been generated (if data dimensions were suitable).
```
*(Actual output for ML tasks will include metrics like accuracy, R-squared, inertia, etc., and may generate plots if enabled and data is suitable.)*

## Screenshot

<!--
[![AetherMindAI Screenshot](assets/aethermind_screenshot.png)](assets/aethermind_screenshot.png)
*A placeholder for a screenshot showcasing AetherMindAI in action within a Colab notebook.*
-->
*Coming soon: A visual glimpse of AetherMindAI!*

## Roadmap

AetherMindAI is an evolving project. Here are some potential directions and features planned for the future:

*   📄 **Retrieval Augmented Generation (RAG):** Integrate RAG capabilities to allow AetherMindAI to answer questions based on custom documents or knowledge bases.
*   🛠️ **Enhanced Agentic Behavior:** Develop more sophisticated agent-like functionalities, enabling AetherMindAI to perform multi-step tasks or use a wider array of tools.
*   🖼️ **Graphical User Interface (GUI):** Create a simple web-based GUI using Streamlit or Gradio for a more interactive user experience outside the direct Colab cell output.
*   🤖 **Support for More Models:** Add easy configuration to switch between different lightweight language models.
*   📊 **Advanced Data Analysis Tools:** Integrate tools for more complex data loading, preprocessing, and visualization, potentially linking with user-uploaded datasets.
*   🌐 **Web Interaction:** Allow the assistant to fetch information from the web to answer questions or complete tasks.
*   🧩 **Community-Driven Extensions:** Develop a framework for users to easily add new tools, capabilities, or specialized knowledge modules.
*   💾 **Persistent Memory:** Implement options for long-term memory or user profile storage.

## Contributing

Contributions are welcome and appreciated! Whether it's reporting a bug, suggesting a new feature, improving documentation, or submitting a pull request, your help makes AetherMindAI better.

Please follow these general guidelines:

1.  **Check for Existing Issues:** Before submitting a new issue or feature request, please check the [Issue Tracker](https://github.com/YOUR_GITHUB_USERNAME/AetherMindAI/issues) to see if it has already been reported or discussed.
2.  **Fork the Repository:** Create your own fork of the AetherMindAI repository.
3.  **Create a Feature Branch:** For new features or bug fixes, create a new branch from `main` (e.g., `git checkout -b feature/your-amazing-feature` or `bugfix/issue-description`).
4.  **Make Your Changes:** Implement your feature or fix the bug. Ensure your code is clear and well-commented where necessary.
5.  **Test Your Changes:** Verify that your changes work as expected and do not break existing functionality.
6.  **Submit a Pull Request:** Push your changes to your fork and submit a pull request to the `main` branch of the original AetherMindAI repository. Provide a clear description of your changes in the pull request.

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) [Year] [Your Name or Organization Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
*(You should replace `[Year]` and `[Your Name or Organization Name]` with the appropriate details. Consider also creating a separate `LICENSE` file in the repository containing this text.)*