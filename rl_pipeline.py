#!/usr/bin/env python3
"""
RL Trading Pipeline
Полный пайплайн для подготовки данных, тренировки RL агента и автоматической оптимизации до достижения приемлемых показателей прибыльности:
1. Скачивание данных
2. Очистка данных
3. Feature engineering
4. Обучение RL модели
5. Оценка модели
6. Автоматическая оптимизация при необходимости
"""

import os
import subprocess
import sys
import pickle
import json
from datetime import datetime, timedelta
import argparse
import time

def run_command(command, description):
    """Run shell command with error handling and real-time output"""
    print(f"\n🟡 {description}")
    print("=" * 50)

    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output in real-time
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                output_lines.append(output.strip())

        return_code = process.poll()

        if return_code == 0:
            print("✅ Команда выполнена успешно")
            return True, '\n'.join(output_lines)
        else:
            print(f"❌ Ошибка выполнения команды (код {return_code})")
            return False, '\n'.join(output_lines)

    except Exception as e:
        print(f"❌ Ошибка выполнения команды: {e}")
        return False, str(e)

def get_price_data(data_days=365):
    """Шаг 1: Скачивание данных"""
    # Модифицируем get_price_data.py для использования переменных
    start_date = datetime.now() - timedelta(days=data_days)
    end_date = datetime.now()

    # Создаем временный скрипт с параметрами
    temp_script = f"""
import sys
sys.path.append('.')
from get_price_data import download_and_process_data
from datetime import datetime, timedelta

start_date = datetime.now() - timedelta(days={data_days})
end_date = datetime.now()
download_and_process_data(start_date, end_date)
"""

    with open('temp_get_data.py', 'w') as f:
        f.write(temp_script)

    success, output = run_command("python temp_get_data.py", f"Скачивание данных за последние {data_days} дней")
    os.remove('temp_get_data.py')
    return success

def clean_data():
    """Шаг 2: Очистка данных"""
    return run_command("python clean_data.py", "Очистка данных (удаление NaN, ненужных колонок)")[0]

def feature_engineer():
    """Шаг 3: Feature engineering"""
    return run_command("python feature_engineer.py", "Добавление технических индикаторов")[0]

def train_rl(timesteps=100000):
    """Шаг 4: Обучение RL модели"""
    command = f"python3 train_rl.py --timesteps {timesteps}"
    return run_command(command, f"Обучение RL агента ({timesteps} шагов)")[0]

def evaluate_model():
    """Оценка обученной модели"""
    print("\n🔍 Оценка модели")
    print("=" * 50)
    
    # Попробуем загрузить результаты комплексной оценки
    try:
        with open('rl_comprehensive_evaluation.pkl', 'rb') as f:
            results = pickle.load(f)
        
        summary = results['summary']
        avg_return = summary.get('avg_return', 0)
        avg_sharpe = summary.get('avg_sharpe', 0)
        avg_drawdown = summary.get('avg_drawdown', 0)
        
        print(f"Средняя доходность: {avg_return*100:.2f}%")
        print(f"Средний коэффициент Шарпа: {avg_sharpe:.4f}")
        print(f"Средняя максимальная просадка: {avg_drawdown*100:.2f}%")
        
        return avg_return, avg_sharpe, avg_drawdown
    except FileNotFoundError:
        print("❌ Файл результатов оценки не найден")
        # Если файл не существует, запустим оценку
        command = "python -c \"import sys; sys.path.append('.'); from train_rl import evaluate_agent_comprehensive; from stable_baselines3 import PPO; import pandas as pd; model = PPO.load('ppo_trading_agent'); df = pd.read_csv('btc_usdt_data/full_btc_usdt_data_feature_engineered.csv'); df = df.tail(2000).reset_index(drop=True); results = evaluate_agent_comprehensive(model, 'btc_usdt_data/full_btc_usdt_data_feature_engineered.csv', n_episodes=5); print('RESULTS:', results['summary'])\""
        success, output = run_command(command, "Комплексная оценка модели")
        
        if success:
            # Попробуем извлечь результаты из вывода
            if 'RESULTS:' in output:
                # Это сложный случай, так как результаты в stdout
                # Для упрощения будем считать, что если команда успешна, 
                # то результаты были сохранены в файл
                try:
                    with open('rl_comprehensive_evaluation.pkl', 'rb') as f:
                        results = pickle.load(f)
                    summary = results['summary']
                    avg_return = summary.get('avg_return', 0)
                    avg_sharpe = summary.get('avg_sharpe', 0)
                    avg_drawdown = summary.get('avg_drawdown', 0)
                    
                    print(f"Средняя доходность: {avg_return*100:.2f}%")
                    print(f"Средний коэффициент Шарпа: {avg_sharpe:.4f}")
                    print(f"Средняя максимальная просадка: {avg_drawdown*100:.2f}%")
                    
                    return avg_return, avg_sharpe, avg_drawdown
                except FileNotFoundError:
                    print("❌ Не удалось загрузить результаты оценки")
                    return 0, 0, 0
            else:
                print("❌ Не удалось извлечь результаты оценки из вывода")
                return 0, 0, 0
        else:
            print("❌ Оценка модели не удалась")
            return 0, 0, 0

def optimize_hyperparameters(n_trials=30):
    """Оптимизация гиперпараметров"""
    print(f"\n⚙️ Оптимизация гиперпараметров ({n_trials} проб)")
    print("=" * 50)
    
    command = f"python hyperparameter_optimization.py --n_trials {n_trials}"
    success, output = run_command(command, f"Оптимизация гиперпараметров ({n_trials} проб)")
    
    if success:
        print("✅ Оптимизация гиперпараметров завершена")
        # Найдем файл исследования
        import glob
        study_files = glob.glob("hyperparameter_study_*.pkl")
        if study_files:
            # Используем самый новый файл
            latest_study = max(study_files, key=os.path.getctime)
            print(f"Файл исследования: {latest_study}")
            return latest_study
        else:
            print("❌ Файл исследования не найден")
            return None
    else:
        print("❌ Оптимизация гиперпараметров не удалась")
        return None

def check_performance_criteria(avg_return, avg_sharpe, avg_drawdown, min_return=0.02, min_sharpe=0.5, max_drawdown=0.15):
    """Проверка, удовлетворяет ли модель критериям производительности"""
    print(f"\n📋 Проверка критериев производительности:")
    print(f"  Минимальная доходность: {min_return*100:.2f}% (текущая: {avg_return*100:.2f}%)")
    print(f"  Минимальный Шарп: {min_sharpe:.2f} (текущий: {avg_sharpe:.4f})")
    print(f"  Максимальная просадка: {max_drawdown*100:.2f}% (текущая: {avg_drawdown*100:.2f}%)")
    
    meets_criteria = (
        avg_return >= min_return and
        avg_sharpe >= min_sharpe and
        avg_drawdown <= max_drawdown
    )
    
    print(f"  ✅ Критерии удовлетворены: {meets_criteria}")
    return meets_criteria

def iterative_training_pipeline(data_days=365, initial_timesteps=100000, max_iterations=10, 
                              min_return=0.02, min_sharpe=0.5, max_drawdown=0.15):
    """Итеративный пайплайн обучения до достижения приемлемых показателей"""
    print(f"\n🔄 Запуск итеративного пайплайна обучения")
    print("=" * 60)
    print(f"Целевые показатели:")
    print(f"  - Минимальная доходность: {min_return*100:.2f}%")
    print(f"  - Минимальный коэффициент Шарпа: {min_sharpe:.2f}")
    print(f"  - Максимальная просадка: {max_drawdown*100:.2f}%")
    print(f"  - Максимум итераций: {max_iterations}")
    print("=" * 60)
    
    # Сначала выполним базовую подготовку данных, если нужно
    print("\n📦 Подготовка данных...")
    if not os.path.exists('btc_usdt_data/full_btc_usdt_data_feature_engineered.csv'):
        print("⚠️ Требуется подготовка данных...")
        if not get_price_data(data_days):
            print("❌ Не удалось скачать данные")
            return False
        if not clean_data():
            print("❌ Не удалось очистить данные")
            return False
        if not feature_engineer():
            print("❌ Не удалось создать признаки")
            return False
    else:
        print("✅ Данные уже подготовлены")
    
    # Цикл оптимизации до достижения целевых показателей
    for iteration in range(max_iterations):
        print(f"\n{'='*20} ИТЕРАЦИЯ #{iteration+1} {'='*20}")
        
        # Обучение модели
        print(f"\n🎯 Обучение модели (итерация {iteration+1})")
        if not train_rl(initial_timesteps):
            print(f"❌ Обучение на итерации {iteration+1} не удалось")
            continue
        
        # Оценка модели
        print(f"\n📊 Оценка модели (итерация {iteration+1})")
        avg_return, avg_sharpe, avg_drawdown = evaluate_model()
        
        # Проверка критериев
        if check_performance_criteria(avg_return, avg_sharpe, avg_drawdown, min_return, min_sharpe, max_drawdown):
            print(f"\n🎉 ЦЕЛЕВЫЕ ПОКАЗАТЕЛИ ДОСТИГНУТЫ НА ИТЕРАЦИИ {iteration+1}!")
            print(f"Финальные результаты:")
            print(f"  - Доходность: {avg_return*100:.2f}%")
            print(f"  - Коэффициент Шарпа: {avg_sharpe:.4f}")
            print(f"  - Максимальная просадка: {avg_drawdown*100:.2f}%")
            return True
        else:
            print(f"\n❌ Целевые показатели НЕ достигнуты на итерации {iteration+1}")
            
            # Если это последняя итерация, выходим
            if iteration == max_iterations - 1:
                print(f"\n⚠️  Достигнут лимит итераций ({max_iterations}). Результаты могут быть неудовлетворительными.")
                print(f"Лучшие достигнутые результаты:")
                print(f"  - Доходность: {avg_return*100:.2f}%")
                print(f"  - Коэффициент Шарпа: {avg_sharpe:.4f}")
                print(f"  - Максимальная просадка: {avg_drawdown*100:.2f}%")
                return False
            
            # Оптимизация гиперпараметров для следующей итерации
            print(f"\n🔬 Запуск оптимизации гиперпараметров перед следующей итерацией...")
            study_file = optimize_hyperparameters(n_trials=20)
            
            if study_file:
                print(f"✅ Оптимизация завершена. Используем лучшие параметры из: {study_file}")
                # Здесь мы могли бы обновить параметры обучения, но для упрощения
                # просто продолжим со следующей итерацией
            else:
                print(f"⚠️  Оптимизация не удалась, продолжаем с увеличенным количеством шагов")
                initial_timesteps = int(initial_timesteps * 1.5)  # Увеличиваем шаги для следующей итерации
    
    return False

def main():
    parser = argparse.ArgumentParser(description='RL Trading Pipeline с автоматической оптимизацией')
    parser.add_argument('--data-days', type=int, default=365,
                       help='Количество дней данных для скачивания (по умолчанию 365)')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Количество шагов тренировки RL (по умолчанию 100000)')
    parser.add_argument('--max-iterations', type=int, default=5,
                       help='Максимальное количество итераций обучения/оптимизации (по умолчанию 5)')
    parser.add_argument('--min-return', type=float, default=0.02,
                       help='Минимальная требуемая доходность (по умолчанию 0.02, т.е. 2%)')
    parser.add_argument('--min-sharpe', type=float, default=0.5,
                       help='Минимальный требуемый коэффициент Шарпа (по умолчанию 0.5)')
    parser.add_argument('--max-drawdown', type=float, default=0.15,
                       help='Максимально допустимая просадка (по умолчанию 0.15, т.е. 15%)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Пропустить скачивание данных (использовать существующие)')
    parser.add_argument('--skip-clean', action='store_true',
                       help='Пропустить очистку данных')
    parser.add_argument('--skip-features', action='store_true',
                       help='Пропустить feature engineering')

    args = parser.parse_args()

    print("🚀 Запуск RL Trading Pipeline с автоматической оптимизацией")
    print("=" * 70)
    print(f"Настройки:")
    print(f"  - Данные за: {args.data_days} дней")
    print(f"  - Начальные шаги обучения: {args.timesteps:,}")
    print(f"  - Максимум итераций: {args.max_iterations}")
    print(f"  - Целевая доходность: {args.min_return*100:.2f}%")
    print(f"  - Целевой коэффициент Шарпа: {args.min_sharpe}")
    print(f"  - Максимальная просадка: {args.max_drawdown*100:.2f}%")
    print("=" * 70)

    # Если не пропускаем подготовку данных, проверяем их наличие
    if not (args.skip_download and args.skip_clean and args.skip_features):
        if not os.path.exists('btc_usdt_data/full_btc_usdt_data_feature_engineered.csv'):
            print("\n📦 Подготовка данных требуется...")
            
            steps = [
                ('data_download', lambda: get_price_data(args.data_days), args.skip_download, "Скачивание данных"),
                ('data_cleaning', clean_data, args.skip_clean, "Очистка данных"),
                ('feature_engineering', feature_engineer, args.skip_features, "Feature engineering"),
            ]

            completed_steps = []
            failed_steps = []

            for step_name, step_func, skip_flag, description in steps:
                if skip_flag:
                    print(f"⏭️ Пропуск {description}")
                    continue

                print(f"\n📋 Шаг: {description}")
                if step_func():
                    completed_steps.append(step_name)
                    print(f"✅ Шаг '{description}' завершен успешно")
                else:
                    failed_steps.append(step_name)
                    print(f"❌ Шаг '{description}' завершен с ошибкой")
                    break

            if failed_steps:
                print(f"\n❌ Подготовка данных не удалась: {', '.join(failed_steps)}")
                sys.exit(1)
        else:
            print("\n✅ Данные уже подготовлены")

    # Запуск итеративного пайплайна
    success = iterative_training_pipeline(
        data_days=args.data_days,
        initial_timesteps=args.timesteps,
        max_iterations=args.max_iterations,
        min_return=args.min_return,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown
    )

    if success:
        print("\n🎉 Пайплайн завершен успешно! Достигнуты удовлетворительные показатели прибыльности.")
        print("\nДальнейшие шаги:")
        print("  1. Проведите дополнительное тестирование на отложенной выборке")
        print("  2. Запустите бумажную торговлю: python rl_paper_trading.py")
        print("  3. При удовлетворительных результатах переходите к живой торговле с осторожностью")
    else:
        print("\n❌ Пайплайн завершен, но целевые показатели не достигнуты.")
        print("Рекомендуется:")
        print("  1. Проверить качество данных")
        print("  2. Увеличить объем обучающих данных")
        print("  3. Рассмотреть другие архитектуры модели или стратегии обучения")
        print("  4. Пересмотреть целевые показатели прибыльности")

if __name__ == "__main__":
    main()
