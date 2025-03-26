# IMD3011 - Datacentric AI

## Learning with Limited Labels using Weak Supervision and Uncertainty-Aware Training

## [Dr. Elias Jacob de Menezes Neto](https://docente.ufrn.br/elias.jacob)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)

## Overview

This repository contains the code and resources for the course **IMD3011**, which focuses on advanced techniques for training machine learning models effectively when labeled data is scarce or noisy. We emphasize data-centric AI approaches, weak supervision methodologies, semi-supervised learning strategies, and annotation error detection mechanisms.

## Table of Contents

- [Overview](#overview)
- [Course Content](#course-content)
  - [Theoretical Introduction to Data-Centric AI and Weakly Supervised Learning](#theoretical-introduction-to-data-centric-ai-and-weakly-supervised-learning)
  - [Semi-supervised and Positive Unlabeled Learning](#semi-supervised-and-positive-unlabeled-learning)
  - [Weak Supervision Pipeline](#weak-supervision-pipeline)
  - [Advanced Topics in Weak Supervision - Named Entity Recognition](#advanced-topics-in-weak-supervision---named-entity-recognition)
  - [Annotation Error Detection](#annotation-error-detection)
  - [Confident Learning and Cleanlab](#confident-learning-and-cleanlab)
  - [Advanced Label Models](#advanced-label-models)
  - [Influence Functions](#influence-functions)
  - [Active Learning](#active-learning)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Using VS Code Dev Containers](#using-vs-code-dev-containers)
- [Teaching Approach](#teaching-approach)
- [Final Project](#final-project)
- [Contributing](#contributing)
- [Further Reading](#further-reading)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact](#contact)

## Course Content

Please refer to the [Syllabus](Programa_IMD3011.pdf) for a detailed overview of the course topics and schedule.

### Theoretical Introduction to Data-Centric AI and Weakly Supervised Learning
- Data-Centric AI paradigm
- Principles of Data-Centric AI
- Weak supervision techniques
- Types of weak supervision
- Aggregation of multiple labeling sources

### Semi-supervised and Positive Unlabeled Learning
- Semi-supervised learning approaches
- Self-training, co-training, and multi-view learning
- Label propagation
- Positive Unlabeled (PU) learning
- Elkan and Noto approach to PU learning

### Weak Supervision Pipeline
- Labeling Functions (LFs)
- Label Model
- Integration with Semi-Supervised Learning
- Snorkel Framework
- Evaluation metrics and comparison with fully supervised learning

### Advanced Topics in Weak Supervision - Named Entity Recognition
- Named Entity Recognition (NER) using weak supervision
- Skweak Framework
- Document-Level Labeling
- Transfer Learning in NER tasks
- Iterative refinement of labeling functions

### Annotation Error Detection
- Types of label noise
- Label noise transition matrix
- Retagging techniques
- Confident Learning for identifying mislabeled instances

### Confident Learning and Cleanlab
- Confident Learning methodology
- Cleanlab library
- Application to various data modalities
- Extension to multi-label classification
- Handling model miscalibration

### Advanced Label Models
- Snorkel MeTaL
- Generative Model
- Flying Squid
- Dawid-Skene
- Hyper Label Model
- CrowdLab

### Influence Functions
- Influence functions for model interpretation
- Source-Aware Influence Functions

### Active Learning
- Active Learning strategies
- Uncertainty sampling
- Query by Committee
- Diversity sampling

## Prerequisites

Ensure your system meets the following requirements:

- **Operating System:** Linux, macOS, or Windows 10+ (WSL recommended for Windows)
- **Python:** Version 3.12 or higher
- **Access to a Machine with a GPU:** Recommended for computationally intensive tasks; alternatively, use Google Colab.
- **Installation of [Poetry](https://python-poetry.org/docs/):** For managing Python dependencies.
- **Weights & Biases Account:** For experiment tracking and visualization. Sign up [here](https://wandb.ai/).
- **Ollama:** For downloading datasets and resources. Download it from [here](https://ollama.com/download).

## Installation

Follow these steps to set up your environment and install the dependencies:

1. **Clone the Repository:**

    ```shell
    git clone https://github.com/eliasjacob/imd3011-datacentric_ai.git
    cd imd3011-datacentric_ai
    ```

2. **Install Dependencies:**

   - **For GPU support:**
     
     ```shell
     poetry install --sync -E cuda --with cuda
     poetry shell
     ```

   - **For CPU-only support:**
     
     ```shell
     poetry install --sync -E cpu
     poetry shell
     ```

3. **Authenticate Weights & Biases:**

    ```shell
    wandb login
    ```

4. **Install Ollama:**

   Download and install Ollama from [here](https://ollama.com/download).

## Using VS Code Dev Containers

This repository is configured for Visual Studio Code Dev Containers, providing a consistent and isolated development environment:

1. **Install [Visual Studio Code](https://code.visualstudio.com/)** and the **[Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)**.
2. Clone this repository if you haven't already.
3. Open the repository in VS Code.
4. When prompted, click "Reopen in Container" or use the command palette (F1) and select "Remote-Containers: Reopen in Container".
5. VS Code will build the Docker container and set up the environment. (This may take several minutes on the first build.)
6. Once built, you'll have a fully configured development environment with all necessary dependencies installed.

## Teaching Approach

The course employs a **top-down** teaching method, differing from the traditional **bottom-up** approach.

- **Top-Down Method:** We begin with a high-level overview and practical applications, then delve into the details when needed. This helps maintain motivation and offers a clear view of how different components integrate.
- **Bottom-Up Method:** In contrast, this approach involves learning individual components in isolation, which can lead to a fragmented understanding.

### Example: Learning Baseball

Harvard Professor David Perkins, in his book [Making Learning Whole](https://www.amazon.com/Making-Learning-Whole-Principles-Transform/dp/0470633719), compares learning to playing baseball. Children start by playing the game and gradually learn the rules. Similarly, you will begin with practical applications and gradually uncover the underlying theories.

> **Important:** Initially, focus on what things do rather than all the details about how they work.

## Final Project

Your final project will be evaluated based on:

- **Technical Quality:** Robustness and efficacy of your implementation.
- **Creativity:** Originality of your approach.
- **Usefulness:** Practical applicability.
- **Presentation:** Effectiveness of your project showcase.
- **Documentation:** Clarity and thoroughness of your report.

### Project Guidelines

- **Individual Work:** The project must be completed individually.
- **Submission:** Provide a link to a GitHub repository or shared folder with your code, data, and report. Use virtual environments along with a `requirements.txt` file to ensure reproducibility.
- **Deadline:** Refer to the syllabus.
- **Presentation:** Prepare a 10-minute presentation to demonstrate your project.
- **Submission Platform:** Use the designated platform (e.g., SIGAA).

## Contributing

Contributions are welcome! To contribute:

1. **Fork the Repository:** Click on the "Fork" button at the top right of the repository page.
2. **Create a New Branch:**

    ```shell
    git checkout -b feature/YourFeature
    ```

3. **Implement Your Changes.**
4. **Commit Your Changes:**

    ```shell
    git commit -m 'Add some feature'
    ```

5. **Push to Your Branch:**

    ```shell
    git push origin feature/YourFeature
    ```

6. **Create a Pull Request:** Navigate to your fork on GitHub and click "New pull request."

## License

This course and its materials are a resource made possible by the investment of Brazilian taxpayers in public education. Your access to this educational opportunity is a direct result of their collective contribution.

As you learn and benefit from this course, please remember the broader community that supports institutions like our Federal University. I encourage you to honor this opportunity by using your education to contribute positively to society and to share your knowledge openly whenever possible.

This course is open-source and licensed under the MIT License. See the [LICENSE](LICENSE) file for details. Let's pay it forward by sharing what we learn!

## Contact

For any questions or feedback regarding the course materials or repository, please [contact me](mailto:elias.jacob@ufrn.br).

If you wish to provide anonymous feedback, use the [Google Forms feedback link](https://jacob.al/feedbacks).