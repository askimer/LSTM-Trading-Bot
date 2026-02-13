#!/usr/bin/env python3
"""
Скрипт для запуска RL Trading Pipeline с автоматической установкой зависимостей
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def install_dependencies():
    """Установка зависимостей с помощью uv"""
    print("📦 Установка основных зависимостей с помощью uv...")
    
    try:
        # Установка зависимостей через uv
        result = subprocess.run(["uv", "sync"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Ошибка при установке зависимостей: {result.stderr}")
            return False
        else:
            print("✅ Основные зависимости установлены")
            return True
    except FileNotFoundError:
        print("❌ uv не найден. Установите uv сначала: pip install uv")
        return False
    except Exception as e:
        print(f"❌ Ошибка при установке зависимостей: {e}")
        return False

def check_torch_installed():
    """Проверка, установлен ли PyTorch"""
    try:
        import torch
        print(f"✅ PyTorch уже установлен (версия {torch.__version__})")
        return True
    except ImportError:
        print("⚠️ PyTorch не найден")
        return False

def install_torch():
    """Установка PyTorch с CPU поддержкой через uv"""
    print("📦 Установка PyTorch с CPU поддержкой через uv...")
    
    try:
        # Установка PyTorch с CPU поддержкой через uv
        result = subprocess.run([
            "uv", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Ошибка при установке PyTorch: {result.stderr}")
            return False
        else:
            print("✅ PyTorch установлен")
            return True
    except Exception as e:
        print(f"❌ Ошибка при установке PyTorch: {e}")
        return False

def check_sb3_installed():
    """Проверка, установлен ли Stable-Baselines3"""
    try:
        import stable_baselines3
        print(f"✅ Stable-Baselines3 уже установлен (версия {stable_baselines3.__version__})")
        return True
    except ImportError:
        print("⚠️ Stable-Baselines3 не найден")
        return False

def install_sb3():
    """Установка Stable-Baselines3 через uv"""
    print("📦 Установка Stable-Baselines3 через uv...")
    
    try:
        result = subprocess.run([
            "uv", "pip", "install", 
            "stable-baselines3[extra]"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Ошибка при установке Stable-Baselines3: {result.stderr}")
            return False
        else:
            print("✅ Stable-Baselines3 установлен")
            return True
    except Exception as e:
        print(f"❌ Ошибка при установке Stable-Baselines3: {e}")
        return False

def run_pipeline(data_days, timesteps, max_iterations, min_return, min_sharpe, max_drawdown):
    """Запуск пайплайна обучения с потоковым выводом"""
    print("🚀 Запуск RL Trading Pipeline...")

    try:
        # Start TensorBoard in background
        import subprocess
        import webbrowser
        import time
        import os

        # Check if tensorboard is installed
        try:
            import tensorboard
            print("📊 Запуск TensorBoard...")
            tensorboard_process = subprocess.Popen(
                ["tensorboard", "--logdir", "./rl_tensorboard/", "--port", "6006"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(3)  # Give TensorBoard time to start

            # Open browser
            print("🌐 Открытие TensorBoard в браузере...")
            webbrowser.open("http://localhost:6006/")
            print("📈 TensorBoard доступен по адресу: http://localhost:6006/")
        except ImportError:
            print("⚠️ TensorBoard не установлен, пропускаем запуск")
            tensorboard_process = None

        # Import and run pipeline directly instead of subprocess
        # This ensures proper output streaming
        print("🔧 Импорт и запуск пайплайна...")
        from rl_pipeline import main as pipeline_main
        import sys

        # Save original sys.argv
        original_argv = sys.argv[:]

        # Set new arguments for rl_pipeline.py
        sys.argv = [
            "rl_pipeline.py",
            "--data-days", str(data_days),
            "--timesteps", str(timesteps),
            "--max-iterations", str(max_iterations),
            "--min-return", str(min_return),
            "--min-sharpe", str(min_sharpe),
            "--max-drawdown", str(max_drawdown)
        ]

        try:
            # Run the pipeline
            pipeline_main()
            success = True
        except SystemExit as e:
            success = e.code == 0
        except Exception as e:
            print(f"❌ Ошибка выполнения пайплайна: {e}")
            success = False

        # Restore original sys.argv
        sys.argv = original_argv

        # Clean up TensorBoard process
        if tensorboard_process:
            print("\n🔴 Завершение TensorBoard...")
            tensorboard_process.terminate()
            try:
                tensorboard_process.wait(timeout=5)
                print("✅ TensorBoard успешно завершен")
            except subprocess.TimeoutExpired:
                print("⚠️ TensorBoard не завершился, принудительное завершение...")
                tensorboard_process.kill()
                tensorboard_process.wait()

        if success:
            print("✅ Пайплайн успешно завершен!")
            return True
        else:
            print("❌ Пайплайн завершился с ошибкой")
            return False
    except Exception as e:
        print(f"❌ Ошибка при запуске пайплайна: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Запуск RL Trading Pipeline с автоматической установкой зависимостей')
    parser.add_argument('--data-days', type=int, default=720, help='Количество дней данных (по умолчанию 720)')
    parser.add_argument('--timesteps', type=int, default=50000, help='Количество шагов обучения (по умолчанию 50000)')
    parser.add_argument('--max-iterations', type=int, default=3, help='Максимум итераций (по умолчанию 3)')
    parser.add_argument('--min-return', type=float, default=0.01, help='Минимальная доходность (по умолчанию 0.01, т.е. 1%)')
    parser.add_argument('--min-sharpe', type=float, default=0.3, help='Минимальный коэффициент Шарпа (по умолчанию 0.3)')
    parser.add_argument('--max-drawdown', type=float, default=0.2, help='Максимальная просадка (по умолчанию 0.2, т.е. 20%)')
    parser.add_argument('--skip-install', action='store_true', help='Пропустить установку зависимостей')
    
    args = parser.parse_args()
    
    print("🤖 RL Algorithmic Trading Bot - Автоматический запуск пайплайна")
    print("="*60)
    
    if not args.skip_install:
        # Установка основных зависимостей
        if not install_dependencies():
            print("❌ Не удалось установить основные зависимости")
            sys.exit(1)
        
        # Установка PyTorch если не установлен
        if not check_torch_installed():
            if not install_torch():
                print("❌ Не удалось установить PyTorch")
                sys.exit(1)
        
        # Установка Stable-Baselines3 если не установлен
        if not check_sb3_installed():
            if not install_sb3():
                print("❌ Не удалось установить Stable-Baselines3")
                sys.exit(1)
    else:
        print("⏭️ Пропуск установки зависимостей (--skip-install)")
    
    print("\n📊 Параметры запуска:")
    print(f"  - Дни данных: {args.data_days}")
    print(f"  - Шаги обучения: {args.timesteps:,}")
    print(f"  - Максимум итераций: {args.max_iterations}")
    print(f"  - Мин. доходность: {args.min_return*100:.2f}%")
    print(f"  - Мин. Шарп: {args.min_sharpe:.2f}")
    print(f"  - Макс. просадка: {args.max_drawdown*100:.2f}%")
    
    success = run_pipeline(
        args.data_days,
        args.timesteps,
        args.max_iterations,
        args.min_return,
        args.min_sharpe,
        args.max_drawdown
    )
    
    if success:
        print("\n🎉 Пайплайн успешно завершен!")
        print("Для следующих шагов см. USAGE_GUIDE.md")
    else:
        print("\n❌ Пайплайн завершился с ошибками")
        print("Проверьте сообщения выше для диагностики проблемы")

if __name__ == "__main__":
    main()
