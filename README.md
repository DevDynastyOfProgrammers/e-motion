# e-motion 🎭

e-motion is a survival roguelike game (inspired by *Vampire Survivors*) that adapts its difficulty in real-time based on the player's emotional state. By analyzing facial expressions via a webcam, the game dynamically adjusts enemy spawn rates, speed, and health to keep the player in the "Flow" state.

!Python
!Engine
!ML
!Manager

---

## 🌟 Key Features

*   Affective Computing: Real-time facial emotion recognition (CNN) using a dedicated worker thread to ensure smooth 60 FPS gameplay.
*   Dynamic Difficulty Adjustment (DDA): The Game Director system interpolates game parameters based on a "State Vector" derived from your emotions.
*   Custom ECS Architecture: A high-performance, Data-Driven Entity-Component-System engine written in pure Python.
*   Production-Ready ML: Optimized inference pipeline using NumPy vectorization (replacing heavy Pandas operations).

---

## 🛠️ Architecture Overview

The project is divided into three main layers:

1.  Core (`core/`): The game engine (ECS, Physics, Rendering, Event Bus).
2.  ML Inference (`ml/`): Optimized wrappers for Vision and State models.
3.  Research (`research/`): Data pipelines and training scripts (used for offline model generation).

---

## ⚙️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

### 1. Clone the Repository
git clone https://github.com/DevDynastyOfProgrammers/e-motion
cd e-motion

### 2. Install Dependencies
Ensure you have uv installed. Then sync the environment:
uv sync

### 3. Environment Setup
Create a .env file in the root directory. You can copy the example config:
cp .env.example .env

### 4. ⚠️ Import Vision Model Weights
The pre-trained weights for the Convolutional Neural Network (CNN) are excluded from the repository due to file size limits.

1.  Obtain the file `emotion_model.pth` (from the cloud link).
2.  Place it manually into the following directory:
        weights/emotion_model.pth
    
    *(Note: Ensure the filename is exactly `emotion_model.pth`).*

> Note on State Models: The State Director prototypes (`.npy` files) are lightweight and are already included in the repository. No generation steps are required.

---

## 🚀 How to Run

Once the dependencies are installed and the vision model is placed, start the game using:

uv run main.py

### Controls
*   W / A / S / D: Move Player.
*   Gameplay: Attacks are automatic. Survive as long as possible!

---

## 🧪 Technical Details for Reviewers

*   Multithreading: The ML inference runs in a separate BiofeedbackWorker thread (`core/ecs/systems/biofeedback.py`) to prevent the Global Interpreter Lock (GIL) from blocking the rendering loop.
*   Safety: The inference engine includes a "Safety Clamping" layer to ensure the neural network never outputs game-breaking multipliers (e.g., negative enemy health).
*   Optimization: Runtime state classification has been refactored from Pandas to pure NumPy, reducing inference latency significantly.

---

## 👥 Team & Roles

*   Game Architect & Team Lead: Dmitry Zharyj 
*   ML Engineer (Vision): Nikolai Smirnov 
*   ML Engineer (Game Director): Daniil Tropin 
*   Data Engineer: Artem Hazov