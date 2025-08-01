# An AlphaZero-Inspired Chess Engine in Python (Ongoing)

[![Language](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-PyTorch-orange.svg)](https://pytorch.org/)

**This is an advanced chess engine built on the principles of Deep Reinforcement Learning, inspired by DeepMind's revolutionary AlphaZero. Instead of relying on handcrafted rules, this engine learns to play chess by training a deep neural network on vast datasets of chess games, guided by the strategic evaluations of a powerful traditional engine (Stockfish).**

The core of this project is the synergy between a dual-head convolutional neural network and a Monte Carlo Tree Search (MCTS) algorithm. The network learns to predict the outcome of games and the most promising moves, while MCTS uses these predictions to intelligently explore the game tree and select the optimal move. This repository serves as a comprehensive portfolio piece, demonstrating a full-cycle AI/ML project from data engineering to model deployment.

---

## 🌟 Project Architecture & Vision

TitanZero moves beyond traditional chess programming by treating chess mastery as a machine learning problem. The project is architected as a multi-stage pipeline, showcasing a robust, end-to-end implementation of a learning-based system.

![Architecture Diagram](https://placehold.co/800x400/2d3748/ffffff?text=Data+Pipeline+->+Neural+Network+->+MCTS+Engine)
*(A diagram illustrating the Data Pipeline -> Neural Network -> MCTS Engine flow.)*

The project's philosophy is to demonstrate proficiency in:
* **Data Engineering:** Building a scalable pipeline to process raw data (PGN files) into structured, ML-ready training tensors.
* **Deep Learning:** Designing, training, and validating a sophisticated neural network for a complex task.
* **Algorithm Design:** Integrating the trained model into a powerful search algorithm (MCTS) for high-performance inference.

---

## 🛠️ Technology Stack

* **Primary Language:** **Python 3.x**
* **Machine Learning Framework:** **PyTorch** - For building and training the neural network.
* **Core Libraries:**
    * **NumPy:** For high-performance numerical computation and tensor manipulation.
    * **python-chess:** For PGN parsing, board representation, and move validation.
* **Data Generation Engine:** **Stockfish** - Used as an oracle to generate high-quality evaluation labels (`policy` and `value`) for the training dataset.
* **Data Source:** **Lichess Database** - Provides millions of high-quality human and engine games in PGN format.

---

## 🗺️ Project Pipeline: A Step-by-Step Implementation

This project is broken down into three distinct, logical phases, mirroring a professional AI/ML workflow.

### **Phase 1: Data Engineering - The Supervised Learning Pipeline**
The foundation of any great model is great data. This phase focuses on creating a high-quality, large-scale dataset for training the neural network.

* [✔] **Module 1: Data Acquisition:** Systematically download and stream chess games in PGN format from the Lichess Open Database.
* [✔] **Module 2: Feature Extraction & Label Generation:**
    * Parse each game and process every board position.
    * For each position, use the **Stockfish engine** as a "teacher" to generate the ground-truth labels:
        1.  **Inputs (`X`):** Convert the board state into a `28x8x8` NumPy array. This multi-layered tensor represents piece locations, castling rights, en-passant squares, and other game state information, providing a rich feature set for the network.
        2.  **Policy Target (`y_policy`):** Generate a dictionary of the top moves and their win probabilities as evaluated by Stockfish. This serves as the target probability distribution for the network's policy head.
        3.  **Value Target (`y_value`):** Generate a single floating-point value in the range `[-1, 1]`, representing Stockfish's evaluation of the position (who is winning). This is the target for the network's value head.
* [✔] **Module 3: Data Serialization:** Save the processed `(inputs, policy, value)` tuples into an efficient format (`.npy` files) for fast loading during training.

### **🧠 Phase 2: Deep Learning - The Neural Network Brain**
This phase involves designing and training the core intelligence of the engine.

* [✔] **Module 4: Model Architecture:**
    * Implement a **dual-head convolutional neural network (CNN)**.
    * The network body consists of several residual blocks (ResNet architecture) to process the `28x8x8` spatial input.
    * The body feeds into two separate heads:
        1.  **Policy Head:** A fully connected layer followed by a softmax activation that outputs a probability distribution over all possible moves.
        2.  **Value Head:** A fully connected layer followed by a `tanh` activation that outputs a single scalar value between `[-1, 1]`.
* [✔] **Module 5: Model Training:**
    * Train the network on the dataset generated in Phase 1.
    * Use a combined loss function: `Loss = (Value_Loss) + (Policy_Loss)`. This is a combination of Mean Squared Error for the value head and Cross-Entropy for the policy head.
    * Implement a training loop with validation checks to prevent overfitting and save the best-performing model weights.

### **🚀 Phase 3: Inference - The MCTS Search Engine**
A trained network is not an engine. This phase integrates the network's "intuition" into a powerful, deliberate search algorithm.

* [✔] **Module 6: Monte Carlo Tree Search (MCTS) Implementation:**
    * Build an MCTS algorithm that uses the neural network to guide its search.
    * Each node in the search tree represents a board state and stores visit counts (`N`) and action-values (`Q`).
    * The search proceeds in four steps: **Selection, Expansion, Simulation, and Backpropagation**. The network's policy provides prior probabilities (`P`) for selection, and the value head provides the evaluation for new nodes.
* [✔] **Module 7: Final Move Selection:** After thousands of simulations, the engine selects the move corresponding to the most visited child of the root node.
* [ ] **Module 8: UCI Integration:** Wrap the MCTS engine in the Universal Chess Interface (UCI) protocol to allow it to be used in standard chess GUIs.

---

## 🔧 How to Build and Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Prajjwal-Vish/Chess_Engine_using_Neural_Net_-_MCTS.git](https://github.com/Prajjwal-Vish/Chess_Engine_using_Neural_Net_-_MCTS.git)
    cd Chess_Engine_using_Neural_Net_-_MCTS
    ```
2.  **Set up the environment:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download Dependencies:**
    * Place the **Stockfish engine executable** in the `/stockfish` directory.
    * Place a pre-trained model (e.g., `.pth` file) in the `/models` directory.
4.  **Run the engine (Example):**
    ```bash
    python -m codes.engine_match
    ```

## 🤝 How to Contribute

While this is primarily a solo portfolio project, suggestions, feedback, and discussions on model architecture or training techniques are highly welcome. Please feel free to open an issue to start a discussion.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
