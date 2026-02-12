# 📈 План Улучшения RL Trading Agent

## 🎯 Общая Стратегия

На основе анализа результатов тестирования выявлены критические проблемы балансировки стратегии и качества торговли. Данный план содержит пошаговые действия для улучшения показателей.

---

## 🔴 Критический Приоритет (Неделя 1-2)

### 1. Исправление Баланса Лонг/Шорт

**Проблема:** Агент выполняет 0 лонг-сделок и 2000 шорт-сделок (ratio: 0.00)

**Решения:**

#### A. Балансирование Функции Награды (train_rl_balanced.py)
```python
# Добавить в EnhancedTradingEnvironment

def calculate_balanced_reward(self, action, pnl_pct, position_type):
    """
    Расчет сбалансированной награды с учетом направления
    """
    base_reward = np.log1p(pnl_pct) * 100 if pnl_pct > -0.99 else -100
    
    # Штраф за дисбаланс направлений
    if hasattr(self, 'action_history'):
        recent_actions = self.action_history[-100:]  # Последние 100 действий
        long_count = sum(1 for a in recent_actions if a in [1, 2])
        short_count = sum(1 for a in recent_actions if a in [3, 4])
        total_trades = long_count + short_count
        
        if total_trades > 10:
            long_ratio = long_count / total_trades
            # Штраф за отклонение от 50/50
            balance_penalty = abs(long_ratio - 0.5) * 50  # 0-25
            base_reward -= balance_penalty
    
    return base_reward
```

#### B. Принудительное Вращение Действий
```python
# В TradingEnvironment добавить:

class DirectionalBalanceTracker:
    """Отслеживает и корректирует баланс направлений"""
    
    def __init__(self, window_size=50, target_ratio=0.5, tolerance=0.2):
        self.window_size = window_size
        self.target_ratio = target_ratio
        self.tolerance = tolerance
        self.direction_history = []
    
    def update(self, action):
        """Обновляет историю направлений"""
        if action in [1, 2]:  # Long
            self.direction_history.append('long')
        elif action in [3, 4]:  # Short
            self.direction_history.append('short')
        
        # Ограничиваем размер окна
        if len(self.direction_history) > self.window_size:
            self.direction_history.pop(0)
    
    def get_recommendation(self):
        """Рекомендует направление для восстановления баланса"""
        if len(self.direction_history) < 10:
            return None  # Недостаточно данных
        
        long_count = self.direction_history.count('long')
        short_count = self.direction_history.count('short')
        total = long_count + short_count
        
        if total == 0:
            return None
        
        long_ratio = long_count / total
        
        if long_ratio < self.target_ratio - self.tolerance:
            return 'long'  # Нужно больше лонгов
        elif long_ratio > self.target_ratio + self.tolerance:
            return 'short'  # Нужно больше шортов
        
        return None  # Баланс в норме
```

#### C. Корректировка Параметров Обучения
```python
# В train_rl_balanced.py:

# 1. Увеличить entropy coefficient для исследования
ent_coef=0.05  # Было 0.02

# 2. Уменьшить learning rate для стабильности
learning_rate=5e-5  # Было 1e-4

# 3. Добавить curiosity-driven exploration
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CuriosityFeatureExtractor(BaseFeaturesExtractor):
    """Извлекает признаки с учетом новизны состояний"""
    
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, features_dim)
        )
        # Сеть предсказания следующего состояния
        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(features_dim + 5, 256),  # features + action
            torch.nn.ReLU(),
            torch.nn.Linear(256, observation_space.shape[0])
        )
    
    def forward(self, observations):
        return self.net(observations)
```

---

## 🟠 Высокий Приоритет (Неделя 2-3)

### 2. Улучшение Win Rate и Profit Factor

**Проблема:** Win rate 46.7%, Profit factor 0.968 (< 1.0)

**Решения:**

#### A. Улучшенная Система Управления Рисками
```python
# В enhanced_trading_environment.py

class AdaptiveRiskManager:
    """Адаптивное управление рисками на основе волатильности"""
    
    def __init__(self, base_position_size=0.1, max_position_size=0.5):
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.volatility_window = []
        self.win_streak = 0
        self.loss_streak = 0
    
    def calculate_position_size(self, atr, current_price, balance):
        """
        Динамический расчет размера позиции
        """
        # Базовый размер на основе ATR
        volatility_factor = 1.0 / (1 + atr / current_price * 100)
        
        # Корректировка на основе серии
        streak_factor = 1.0
        if self.win_streak >= 3:
            streak_factor = 1.2  # Увеличиваем после побед
        elif self.loss_streak >= 2:
            streak_factor = 0.7  # Уменьшаем после поражений
        
        position_size = self.base_position_size * volatility_factor * streak_factor
        return min(position_size, self.max_position_size)
    
    def update_streak(self, pnl):
        """Обновляет серию побед/поражений"""
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
```

#### B. Улучшенные Условия Входа/Выхода
```python
# Добавить фильтры для входа:

def should_enter_long(self, obs):
    """Условия для входа в лонг"""
    indicators = self._extract_indicators(obs)
    
    # Минимум 2 из 3 условий:
    conditions = [
        indicators['rsi'] < 40,  # Перепроданность
        indicators['close'] < indicators['bb_lower'],  # Ниже нижней Боллинджера
        indicators['mfi'] < 30,  # MFI перепродан
        indicators['obv_slope'] > 0,  # Рост OBV
    ]
    
    return sum(conditions) >= 2

def should_enter_short(self, obs):
    """Условия для входа в шорт"""
    indicators = self._extract_indicators(obs)
    
    conditions = [
        indicators['rsi'] > 60,  # Перекупленность
        indicators['close'] > indicators['bb_upper'],  # Выше верхней Боллинджера
        indicators['mfi'] > 70,  # MFI перекуплен
        indicators['obv_slope'] < 0,  # Падение OBV
    ]
    
    return sum(conditions) >= 2
```

#### C. Динамические Стоп-лоссы и Тейк-профиты
```python
# ATR-based стопы

def calculate_dynamic_stops(self, entry_price, atr, direction):
    """
    Расчет динамических стопов на основе ATR
    """
    atr_multiplier_sl = 2.0  # Стоп-лосс = 2 * ATR
    atr_multiplier_tp = 3.0  # Тейк-профит = 3 * ATR (1:1.5 RR)
    
    if direction == 'long':
        stop_loss = entry_price - atr * atr_multiplier_sl
        take_profit = entry_price + atr * atr_multiplier_tp
    else:
        stop_loss = entry_price + atr * atr_multiplier_sl
        take_profit = entry_price - atr * atr_multiplier_tp
    
    return stop_loss, take_profit
```

---

## 🟡 Средний Приоритет (Неделя 3-4)

### 3. Оптимизация Гиперпараметров

#### A. Optuna для Поиска Оптимальных Параметров
```python
# hyperparameter_optimization.py

import optuna
from stable_baselines3 import PPO

def objective(trial):
    """Целевая функция для оптимизации"""
    
    # Параметры PPO
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_epochs = trial.suggest_int('n_epochs', 3, 20)
    gamma = trial.suggest_float('gamma', 0.90, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)
    ent_coef = trial.suggest_float('ent_coef', 0.001, 0.1, log=True)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 0.5)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
    
    # Создание модели с пробными параметрами
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        clip_range=clip_range,
        verbose=0
    )
    
    # Обучение
    model.learn(total_timesteps=50000)
    
    # Оценка
    mean_reward = evaluate_model(model, eval_env)
    
    return mean_reward

# Запуск оптимизации
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
```

#### B. Сеточный Поиск для Ключевых Параметров
```python
# Сетка для тестирования:
param_grid = {
    'ent_coef': [0.01, 0.02, 0.05, 0.1],  # Exploration
    'learning_rate': [3e-5, 5e-5, 1e-4, 3e-4],
    'n_steps': [512, 1024, 2048],
    'reward_scaling': [0.5, 1.0, 2.0],
    'position_size': [0.05, 0.1, 0.2],
}
```

---

## 🟢 Низкий Приоритет (Неделя 4+)

### 4. Улучшение Архитектуры Нейросети

#### A. Recurrent Policy для Учета Истории
```python
from stable_baselines3.common.policies import RecurrentActorCriticPolicy

model = PPO(
    "MlpLstmPolicy",  # LSTM для учета временных зависимостей
    env,
    policy_kwargs=dict(
        lstm_hidden_size=64,
        n_lstm_layers=1,
        enable_critic_lstm=True,
    ),
    verbose=1
)
```

#### B. Ensemble из Нескольких Моделей
```python
class EnsembleAgent:
    """Ансамбль из нескольких RL агентов"""
    
    def __init__(self, model_paths):
        self.models = [PPO.load(path) for path in model_paths]
    
    def predict(self, obs, deterministic=True):
        """Голосование агентов"""
        actions = []
        for model in self.models:
            action, _ = model.predict(obs, deterministic=deterministic)
            actions.append(action)
        
        # Мажоритарное голосование
        return max(set(actions), key=actions.count)
```

---

## 📋 План Реализации

### Неделя 1: Критические Исправления
- [ ] Добавить балансировку в функцию награды
- [ ] Реализовать DirectionalBalanceTracker
- [ ] Увеличить ent_coef до 0.05
- [ ] Запустить тестовое обучение (100k steps)
- [ ] Проверить балансировку

### Неделя 2: Улучшение Качества
- [ ] Внедрить AdaptiveRiskManager
- [ ] Добавить фильтры входа на основе индикаторов
- [ ] Реализовать ATR-based стопы
- [ ] Обучить модель с новыми параметрами (500k steps)

### Неделя 3: Оптимизация
- [ ] Настроить Optuna для поиска параметров
- [ ] Провести сеточный поиск
- [ ] Сравнить результаты

### Неделя 4: Полировка
- [ ] Внедрить Recurrent Policy (опционально)
- [ ] Создать ансамбль лучших моделей
- [ ] Провести финальное тестирование

---

## 📊 Метрики Успеха

| Метрика | Текущее | Целевое |
|---------|---------|---------|
| Balance Score | 0.000 | > 0.5 |
| Long/Short Ratio | 0.00 | 0.3 - 3.0 |
| Win Rate | 46.7% | > 55% |
| Profit Factor | 0.968 | > 1.3 |
| Sharpe Ratio | 9.335 | > 8.0 |
| Max Drawdown | 0.04% | < 5% |

---

## 🔄 Быстрые Правки (Immediate Fixes)

### Правка 1: Балансировка Награды (train_rl_balanced.py)

```python
# Найти класс EnhancedTradingEnvironment и добавить:

class EnhancedTradingEnvironment:
    def __init__(self, ...):
        # ... существующий код ...
        self.action_history = []
        self.long_count = 0
        self.short_count = 0
    
    def step(self, action):
        # ... существующий код ...
        
        # Обновляем счетчики
        self.action_history.append(action)
        if action in [1, 2]:
            self.long_count += 1
        elif action in [3, 4]:
            self.short_count += 1
        
        # Добавляем балансировку в награду
        total = self.long_count + self.short_count
        if total > 20:  # После 20 сделок начинаем балансировать
            long_ratio = self.long_count / total
            balance_bonus = -abs(long_ratio - 0.5) * 10  # -5 to 0
            reward += balance_bonus
        
        return obs, reward, terminated, truncated, info
```

### Правка 2: Принудительное Исследование

```python
# В train_rl_balanced.py:

model = PPO(
    # ... остальные параметры ...
    ent_coef=0.05,  # Увеличить с 0.02
    learning_rate=5e-5,  # Уменьшить для стабильности
    # ...
)
```

---

## 🎯 Следующие Действия

1. **Сегодня:** Применить Правку 1 и Правку 2
2. **Завтра:** Запустить обучение на 100k steps
3. **Через 2 дня:** Проверить балансировку с помощью analyze_strategy_balancing.py
4. **Если улучшение есть:** Продолжить обучение до 500k steps
5. **Если нет:** Внедрить AdaptiveRiskManager

---

*Составлено на основе анализа результатов тестирования от 07.02.2026*
