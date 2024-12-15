# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

from os import path as osp
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from dm_control import suite
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


class HumanoidDataset:
    def __init__(self, num_episodes=1000, episode_length=1000, ratio_split=0.8):
        self.env = suite.load(domain_name="humanoid", task_name="run")
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.ratio_split = ratio_split

    def collect_episode_data(self):
        """Collect single episode data"""
        states, actions, rewards = [], [], []
        time_step = self.env.reset()

        # Get action specification for random sampling
        action_spec = self.env.action_spec()

        for _ in range(self.episode_length):
            action = np.random.uniform(
                action_spec.minimum, action_spec.maximum, size=action_spec.shape
            )

            states.append(self._flatten_observation(time_step.observation))
            actions.append(action)

            time_step = self.env.step(action)
            rewards.append(time_step.reward if time_step.reward is not None else 0.0)

            if time_step.last():
                break

        return np.array(states), np.array(actions), np.array(rewards)

    def _flatten_observation(self, observation):
        """Flatten observation dict to array"""
        return np.concatenate([v.flatten() for v in observation.values()])

    def generate_dataset(self):
        all_states, all_actions, all_rewards = [], [], []

        print("Collecting training data...")
        for i in range(self.num_episodes):
            if i % 10 == 0:
                print(f"Episode {i}/{self.num_episodes}")
            states, actions, rewards = self.collect_episode_data()
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)

        states = np.array(all_states)
        actions = np.array(all_actions)
        rewards = np.array(all_rewards)

        split_idx = int(self.num_episodes * self.ratio_split)

        train_data = {
            "input": {"state": states[:split_idx].reshape(-1, states.shape[-1])},
            "label": {
                "action": actions[:split_idx].reshape(-1, actions.shape[-1]),
                "reward": rewards[:split_idx].reshape(-1, 1),
            },
        }

        val_data = {
            "input": {"state": states[split_idx:].reshape(-1, states.shape[-1])},
            "label": {
                "action": actions[split_idx:].reshape(-1, actions.shape[-1]),
                "reward": rewards[split_idx:].reshape(-1, 1),
            },
        }

        return train_data, val_data


class HumanoidController(paddle.nn.Layer):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.net = paddle.nn.Sequential(
            paddle.nn.Linear(state_size, hidden_size),
            paddle.nn.ReLU(),
            paddle.nn.Linear(hidden_size, hidden_size),
            paddle.nn.ReLU(),
            paddle.nn.Linear(hidden_size, action_size),
            paddle.nn.Tanh(),
        )

    def forward(self, x):
        state = paddle.to_tensor(x["state"], dtype="float32")
        return {"action": self.net(state)}


def train_loss_func(output_dict, label_dict, weight_dict=None):
    """Calculate training loss with properly named components"""
    # Predict next state and maximize reward
    action_loss = paddle.mean(
        paddle.square(output_dict["action"] - label_dict["action"])
    )
    reward_loss = -paddle.mean(
        label_dict["reward"]
    )  # Negative since we want to maximize reward
    total_loss = action_loss + 0.1 * reward_loss
    return {"total_loss": total_loss}


def metric_eval(output_dict, label_dict=None, weight_dict=None):
    """Simple metric function that returns a single scalar value"""
    # Use the same calculation as training loss
    action_loss = float(
        paddle.mean(paddle.square(output_dict["action"] - label_dict["action"]))
    )
    reward_loss = float(-paddle.mean(label_dict["reward"]))
    total_loss = action_loss + 0.1 * reward_loss

    # Return a single scalar metric
    return {"val_loss": total_loss}


def train(cfg: DictConfig):
    # Set random seed
    ppsci.utils.misc.set_random_seed(cfg.seed)
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    # Generate dataset
    dataset = HumanoidDataset(
        num_episodes=cfg.DATA.num_episodes, episode_length=cfg.DATA.episode_length
    )
    train_data, val_data = dataset.generate_dataset()

    # Initialize model
    state_size = train_data["input"]["state"].shape[-1]
    action_size = train_data["label"]["action"].shape[-1]
    model = HumanoidController(state_size, action_size, cfg.MODEL.hidden_size)

    # Convert data to float32
    train_data["input"]["state"] = train_data["input"]["state"].astype("float32")
    train_data["label"]["action"] = train_data["label"]["action"].astype("float32")
    train_data["label"]["reward"] = train_data["label"]["reward"].astype("float32")

    val_data["input"]["state"] = val_data["input"]["state"].astype("float32")
    val_data["label"]["action"] = val_data["label"]["action"].astype("float32")
    val_data["label"]["reward"] = val_data["label"]["reward"].astype("float32")

    # Create training constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": train_data["input"],
                "label": train_data["label"],
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
        },
        ppsci.loss.FunctionalLoss(train_loss_func),
        {"action": lambda out: out["action"]},
        name="sup_train",
    )

    # Create validator
    # In your train function, update the validator creation:
    sup_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": val_data["input"],
                "label": val_data["label"],
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        ppsci.loss.FunctionalLoss(train_loss_func),
        {"action": lambda out: out["action"]},
        metric={"metric": ppsci.metric.FunctionalMetric(metric_eval)},
        name="sup_valid",
    )

    # Initialize optimizer and solver
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)
    solver = ppsci.solver.Solver(
        model,
        {sup_constraint.name: sup_constraint},
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        log_freq=cfg.log_freq,
        validator={sup_validator.name: sup_validator},
    )

    solver.train()
    solver.eval()


class HumanoidEvaluator:
    def __init__(self, model_path, num_episodes=5, episode_length=1000):
        self.env = suite.load(domain_name="humanoid", task_name="run")
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.load_model()

    def load_model(self):
        time_step = self.env.reset()
        state_size = sum(v.size for v in time_step.observation.values())
        action_spec = self.env.action_spec()
        action_size = action_spec.shape[0]

        self.model = HumanoidController(state_size, action_size)
        state_dict = paddle.load(self.model_path)
        self.model.set_state_dict(state_dict)
        self.model.eval()

    def _flatten_observation(self, observation):
        return np.concatenate([v.flatten() for v in observation.values()])

    def evaluate_episode(self, record_video=False):
        """Evaluate single episode and collect detailed data"""
        import mujoco

        time_step = self.env.reset()
        total_reward = 0
        frames = []

        # Data collection lists
        episode_data = {"rewards": [], "actions": [], "com_velocity": []}

        # Setup offscreen renderer if recording video
        if record_video:
            width, height = 640, 480
            renderer = mujoco.Renderer(self.env.physics.model, width, height)

        for t in range(self.episode_length):
            # Get state and predict action
            state = self._flatten_observation(time_step.observation)
            state_tensor = {"state": paddle.to_tensor(state[None, :], dtype="float32")}

            with paddle.no_grad():
                action = self.model(state_tensor)["action"].numpy()[0]

            # Render frame if recording
            if record_video:
                renderer.update_scene(self.env.physics.data)
                pixels = renderer.render()
                frames.append(pixels)

            # Take step
            time_step = self.env.step(action)
            reward = time_step.reward if time_step.reward is not None else 0

            # Collect data
            episode_data["rewards"].append(reward)
            episode_data["actions"].append(action)
            episode_data["com_velocity"].append(time_step.observation["velocity"])

            total_reward += reward

            if time_step.last():
                break

        # Convert lists to numpy arrays
        episode_data["rewards"] = np.array(episode_data["rewards"])
        episode_data["actions"] = np.array(episode_data["actions"])
        episode_data["com_velocity"] = np.array(episode_data["com_velocity"])

        return total_reward, frames, episode_data

    def evaluate(self, save_dir="./evaluation_results"):
        """Run full evaluation with multiple episodes and generate analysis"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        rewards = []
        all_episode_data = []
        logger.info("\nStarting evaluation...")

        for ep in range(self.num_episodes):
            logger.info(f"\nEpisode {ep + 1}/{self.num_episodes}")
            # Record video for first and last episodes
            record_video = ep == 0 or ep == self.num_episodes - 1
            reward, frames, episode_data = self.evaluate_episode(record_video)
            rewards.append(reward)
            all_episode_data.append(episode_data)
            logger.info(f"Episode reward: {reward:.2f}")

            # Save video if frames were recorded
            if record_video and frames:
                import imageio

                video_path = save_dir / f"episode_{ep+1}.mp4"
                imageio.mimsave(video_path, frames, fps=30)
                logger.info(f"Saved video to {video_path}")

        # Generate analysis and save statistics
        self._generate_analysis(rewards, all_episode_data, save_dir)
        self._save_statistics(rewards, all_episode_data, save_dir)

        logger.info("\nEvaluation completed!")
        logger.info(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        return rewards

    def _generate_analysis(self, rewards, all_episode_data, save_dir):
        """Generate comprehensive analysis plots"""
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(rewards, "b-o")
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.hist(rewards, bins=min(len(rewards), 10), color="blue", alpha=0.7)
        plt.axvline(np.mean(rewards), color="r", linestyle="--", label="Mean")
        plt.title("Reward Distribution")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.legend()

        plt.subplot(2, 2, 3)
        for i in range(min(3, len(all_episode_data))):
            plt.plot(all_episode_data[i]["rewards"], label=f"Episode {i+1}", alpha=0.7)
        plt.title("Reward Trajectories")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        for i in range(min(3, len(all_episode_data))):
            vel = all_episode_data[i]["com_velocity"]
            speed = np.linalg.norm(vel, axis=1) if len(vel.shape) > 1 else np.abs(vel)
            plt.plot(speed, label=f"Episode {i+1}", alpha=0.7)
        plt.title("Center of Mass Speed")
        plt.xlabel("Step")
        plt.ylabel("Speed")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_dir / "performance_analysis.png")
        plt.close()

    def _save_statistics(self, rewards, all_episode_data, save_dir):
        """Save detailed statistical analysis"""
        with open(save_dir / "detailed_stats.txt", "w") as f:
            f.write("=== Humanoid Evaluation Statistics ===\n\n")

            # Episode Statistics
            f.write("Episode Statistics:\n")
            f.write(f"Number of episodes: {len(rewards)}\n")
            f.write(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
            f.write(f"Max reward: {np.max(rewards):.2f}\n")
            f.write(f"Min reward: {np.min(rewards):.2f}\n\n")

            # Action Statistics
            all_actions = np.concatenate(
                [ep_data["actions"] for ep_data in all_episode_data]
            )
            f.write("Action Statistics:\n")
            f.write(f"Mean action magnitude: {np.mean(np.abs(all_actions)):.3f}\n")
            f.write(f"Max action magnitude: {np.max(np.abs(all_actions)):.3f}\n")
            f.write(f"Action std: {np.std(all_actions):.3f}\n\n")

            # Movement Statistics
            f.write("Movement Statistics:\n")
            for ep_idx, ep_data in enumerate(all_episode_data[:3]):  # First 3 episodes
                velocities = ep_data["com_velocity"]
                speed = (
                    np.linalg.norm(velocities, axis=1)
                    if len(velocities.shape) > 1
                    else np.abs(velocities)
                )
                f.write(f"Episode {ep_idx + 1}:\n")
                f.write(f"  Mean speed: {np.mean(speed):.3f}\n")
                f.write(f"  Max speed: {np.max(speed):.3f}\n")
                f.write(f"  Distance covered: {np.sum(speed):.3f}\n\n")


def evaluate(cfg: DictConfig):
    """Evaluate trained humanoid controller"""
    # Initialize evaluator with trained model
    evaluator = HumanoidEvaluator(
        model_path=cfg.EVAL.pretrained_model_path,
        num_episodes=cfg.EVAL.num_episodes,
        episode_length=cfg.EVAL.episode_length,
    )

    # Create evaluation output directory
    eval_dir = Path(cfg.output_dir) / "evaluation_results"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    logger.info("Starting evaluation...")
    rewards = evaluator.evaluate(
        save_dir=eval_dir
    )  # Changed from run_evaluation to evaluate

    return rewards


@hydra.main(
    version_base=None, config_path="./conf", config_name="humanoid_control.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(
            f"cfg.mode should be in ['train', 'eval'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
