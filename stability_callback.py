import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class StabilityCallback(BaseCallback):
    """Callback для мониторинга и обеспечения стабильности обучения"""

    def __init__(self, check_freq=100, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.recent_rewards = []
        self.recent_kl_divs = []
        self.stability_threshold = 2.0  # Порог нестабильности
        self.kl_threshold = 0.1  # Порог KL-дивергенции

    def _on_step(self):
        """Проверка стабильности во время обучения"""
        # Получаем текущие метрики из логгера
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            logger_dict = self.model.logger.name_to_value
            
            # Проверяем KL-дивергенцию если доступна
            kl_div = None
            for key in logger_dict.keys():
                if 'kl_div' in key.lower() or 'approx_kl' in key.lower():
                    kl_div = logger_dict[key]
                    break
            
            if kl_div is not None:
                self.recent_kl_divs.append(kl_div)
                if len(self.recent_kl_divs) > 50:  # Хранить последние 50 значений
                    self.recent_kl_divs.pop(0)
        
        # Проверяем частоту вызова (каждые check_freq шагов)
        if self.n_calls % self.check_freq == 0:
            self._check_stability()
        
        return True

    def _check_stability(self):
        """Проверка стабильности обучения"""
        # Проверяем вариативность наград
        if hasattr(self, 'logger') and self.logger.name_to_value:
            if 'train/rollout/ep_rew_mean' in self.logger.name_to_value:
                current_reward = self.logger.name_to_value['train/rollout/ep_rew_mean']
                self.recent_rewards.append(current_reward)
                
                if len(self.recent_rewards) > 100:  # Хранить последние 100 наград
                    self.recent_rewards.pop(0)
                
                if len(self.recent_rewards) >= 10:
                    recent_mean = np.mean(self.recent_rewards[-10:])
                    recent_std = np.std(self.recent_rewards[-10:])
                    
                    # Проверяем коэффициент вариации
                    cv = recent_std / abs(recent_mean) if abs(recent_mean) > 0.01 else float('inf')
                    
                    if cv > self.stability_threshold:
                        print(f"⚠️  High reward volatility detected! CV: {cv:.2f}")
                        # Здесь можно добавить логику адаптации learning rate
                        self._adjust_learning_rate(0.9)  # Уменьшаем LR на 10%
                    
                    # Проверяем KL-дивергенцию
                    if len(self.recent_kl_divs) >= 5:
                        avg_kl = np.mean(self.recent_kl_divs[-5:])
                        if avg_kl < 0.001:  # Очень низкая KL-дивергенция
                            print(f"⚠️  Low KL divergence detected: {avg_kl:.4f}")
                            print("   This indicates insufficient policy updates")
                            self._adjust_learning_rate(1.1)  # Увеличиваем LR на 10%

    def _adjust_learning_rate(self, factor):
        """Адаптивное изменение learning rate"""
        if hasattr(self.model, 'lr_schedule') and hasattr(self.model.lr_schedule, 'initial_lr'):
            current_lr = self.model.learning_rate
            new_lr = current_lr * factor
            # Ограничиваем диапазон LR
            new_lr = np.clip(new_lr, 1e-6, 1e-3)
            print(f"   Adjusting learning rate: {current_lr:.6f} -> {new_lr:.6f}")
            
            # Обновляем learning rate (это может не работать в некоторых версиях SB3)
            try:
                self.model.learning_rate = new_lr
            except:
                print("   Could not adjust learning rate directly")

    def _on_training_end(self):
        """Финальная проверка стабильности"""
        if len(self.recent_rewards) >= 50:
            final_cv = np.std(self.recent_rewards[-50:]) / abs(np.mean(self.recent_rewards[-50:]))
            print(f"\n📊 Final stability check:")
            print(f"   Final coefficient of variation: {final_cv:.2f}")
            if final_cv < 0.5:
                print("   ✅ Good stability achieved!")
            else:
                print("   ⚠️  High variability in final rewards")


def get_adaptive_lr(initial_lr=3e-5, kl_threshold=0.01):
    """
    Функция для создания адаптивного learning rate
    
    Args:
        initial_lr: Начальное значение learning rate
        kl_threshold: Порог KL-дивергенции для адаптации
    """
    def schedule(progress_remaining):
        """
        Адаптивное расписание learning rate
        
        Args:
            progress_remaining: Оставшийся прогресс обучения (0-1)
        """
        # Базовое убывающее расписание
        base_lr = initial_lr * progress_remaining
        
        # Здесь можно добавить логику адаптации в зависимости от KL-дивергенции
        # Пока возвращаем просто базовое расписание
        return base_lr
    
    return schedule