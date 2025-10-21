from hud.rl.config import TrainingConfig, ModelConfig
from hud.rl.logger import console
from hud.rl.train import train


def main() -> None:
    training_config = TrainingConfig()
    training_config.model = ModelConfig(base_model="Qwen/Qwen2.5-VL-3B-Instruct")
    training_config.dp_shard = 2
    training_config.optimizer.use_8bit_optimizer = False
    training_config.loss.kl_beta = 0.0
    training_config.output_dir = "/home/ubuntu/myworkspace/hud-python/hud/rl/tests/outputs"
    training_config.benchmark = True

    console.info("=" * 80)
    console.info("Running trainer...")

    train(training_config, max_steps=5)

if __name__ == "__main__":
    main()
