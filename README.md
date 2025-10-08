# 🐍 RL Snake with PPO + Optuna

Este proyecto implementa el clásico juego de Snake entrenado con **Reinforcement Learning (RL)** usando **PPO (Proximal Policy Optimization)** y búsqueda de hiperparámetros con **Optuna**.

🚀 Entrenamiento en paralelo, búsqueda de hiperparámetros y soporte para visualizar el progreso con TensorBoard.

---

## 📂 Estructura del proyecto
rl-snake/
agents/ # Scripts de entrenamiento y evaluación (PPO)

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
## 🏃‍♂️ Entrenamiento
🔍 Búsqueda de hiperparámetros con Optuna

Ejemplo de búsqueda en paralelo (3 terminales corriendo a la vez):

```bash
python -m search.optuna_search --n_trials 40 --timesteps 400000 --storage sqlite:///optuna.db --study_name snake_search --wide --best_out models/best_params.json
```

🤖 Entrenamiento robusto con los mejores parámetros

```bash
python -m agents.ppo_train_use_best --params models/best_params.json --timesteps 20000000 --n_envs 8 --grid_size 12 --max_no_food_steps 80 --tb runs --models models
```

📊 Visualización en TensorBoard

```bash
Durante o después del entrenamiento:
```

## 🎮 Evaluación
Jugar con el mejor modelo

```bash
python -m agents.eval_play --model models/best_model.zip --fps 30
```

Jugar con el modelo final

```bash
python -m agents.eval_play --model models/final_model.zi --fps 30
```
