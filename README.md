# 🐍 RL Snake with PPO + Optuna

Este proyecto implementa el clásico juego de Snake entrenado con **Reinforcement Learning (RL)** usando **PPO (Proximal Policy Optimization)** y búsqueda de hiperparámetros con **Optuna**.

🚀 Entrenamiento en paralelo, búsqueda de hiperparámetros y soporte para visualizar el progreso con TensorBoard.

---

## 📂 Estructura del proyecto
rl-snake/
│── agents/ # Scripts de entrenamiento y evaluación (PPO)
│── envs/ # Entorno Snake en Gymnasium + Pygame
│── search/ # Búsqueda de hiperparámetros con Optuna
│── utils/ # Callbacks, wrappers, helpers
│── models/ # Modelos guardados (ej. final_model.zip)
│── requirements.txt # Dependencias
│── README.md # Este archivo

---

## ⚙️ Instalación

Clona el repo y crea un entorno virtual:

```bash
git clone https://github.com/<TU_USUARIO>/rl-snake.git
cd rl-snake
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```
