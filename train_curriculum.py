"""
Training script with Position Curriculum for accelerated learning.

Usage:
  1. Train from scratch:
     RESUME_FROM_STAGE = None
     LOAD_MODEL_PATH = None
     python train_curriculum.py

  2. Resume from checkpoint:
     LOAD_MODEL_PATH = "checkpoints/stage1_dense/microscope_stage1_dense_1000000_steps"
     python train_curriculum.py

     RESUME_FROM_STAGE = "stage2_simple"
     LOAD_MODEL_PATH = "models/stage1_dense_final"
     python train_curriculum.py
"""
from env.env_virt import MicroscopeEnvAggressive
from env.minimax_wrapper import MinimaxOpponentWrapper
from env.random_opponent_wrapper import RandomOpponentWrapper
from env.action_mask_wrapper import ActionMaskWrapper
from env.symmetry_wrapper import SymmetryAugmentationWrapper
from env.position_curriculum_wrapper import PositionCurriculumWrapper

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import torch.nn as nn

from lib.networks import MicroscopeCNN

# Training configuration
RESUME_FROM_STAGE = None
LOAD_MODEL_PATH = None

# Progressive curriculum with aggressive rewards + position curriculum
# All stages use MicroscopeEnvAggressive (70% material, 20% frontier, 10% restriction)
# Depth-1 minimax is skipped - it's pure greedy, same signal as the reward function
STAGE_PROGRESSION = [
    # Stage 1: Learn basic mechanics vs random + position curriculum
    {
        "name": "stage1_vs_random",
        "env_class": MicroscopeEnvAggressive,
        "timesteps": 1_500_000,
        "learning_rate": 3e-4,
        "description": "Foundation: learn conversion mechanics vs random + curriculum",
        "train_opponent": "random",
        "train_depth": None,
        "eval_opponent": "random",
        "eval_depth": None,
        "use_position_curriculum": True,
    },
    # Stage 2: Jump to depth-3 (depth-1 is redundant with reward signal)
    {
        "name": "stage2_vs_minimax3",
        "env_class": MicroscopeEnvAggressive,
        "timesteps": 3_000_000,
        "learning_rate": 3e-4,
        "description": "Tactical: multi-move planning vs minimax-3 + curriculum",
        "train_opponent": "minimax",
        "train_depth": 3,
        "eval_opponent": "minimax",
        "eval_depth": 3,
        "use_position_curriculum": True,
    },
    # Stage 3: Train vs depth-5
    {
        "name": "stage3_vs_minimax5",
        "env_class": MicroscopeEnvAggressive,
        "timesteps": 3_000_000,
        "learning_rate": 1e-4,
        "description": "Mastery: train vs strong minimax-5 + curriculum",
        "train_opponent": "minimax",
        "train_depth": 5,
        "eval_opponent": "minimax",
        "eval_depth": 5,
        "use_position_curriculum": True,
    },
]


class CurriculumTimestepCallback(BaseCallback):
    """
    Updates PositionCurriculumWrapper with current training timestep.

    This drives the position curriculum progression - as training advances,
    the curriculum introduces more complex positions (mid-game, late-game,
    tactical puzzles).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Update each sub-environment's curriculum with current timestep
        timestep = self.num_timesteps
        for env_idx in range(self.training_env.num_envs):
            try:
                # SubprocVecEnv: set attribute on remote envs
                self.training_env.env_method(
                    'set_timestep',
                    timestep,
                    indices=[env_idx]
                )
            except Exception:
                pass  # Not all envs may have the method (e.g. eval envs)
        return True


def make_env_with_timeout(env_class, render_mode=None, opponent_type=None,
                          minimax_depth=3, use_position_curriculum=False):
    """
    Create environment with safety timeout and optional position curriculum.

    Wrapper order (innermost to outermost):
        MicroscopeEnv
          -> PositionCurriculumWrapper (optional, overrides reset positions)
          -> OpponentWrapper (needs direct access to game_grid/action_stage)
          -> SymmetryAugmentationWrapper (transforms agent-facing obs/actions)
          -> TimeLimit (safety timeout)
          -> Monitor (logging)
          -> ActionMaskWrapper (forwards action_masks)

    IMPORTANT: Opponent wrappers must be INSIDE the symmetry wrapper because
    they access game_grid and action_stage directly on the base env.
    Symmetry wrapper only transforms the agent-facing interface.
    """
    def _init():
        env = env_class(render_mode=render_mode)

        # Position curriculum: override reset positions with diverse game states
        # Must be innermost wrapper (directly around base env)
        if use_position_curriculum:
            env = PositionCurriculumWrapper(env)

        # Add opponent BEFORE symmetry wrapper
        # Opponent wrappers access game_grid and action_stage directly
        if opponent_type == 'random':
            env = RandomOpponentWrapper(env, render_mode=render_mode)
        elif opponent_type == 'minimax':
            env = MinimaxOpponentWrapper(env, depth=minimax_depth)
        # else: self-play (no opponent wrapper)

        # Apply symmetry augmentation for 8x data efficiency
        # AFTER opponent so it only transforms the agent's interface
        # Only augment during training (not during visualization)
        if render_mode is None:
            env = SymmetryAugmentationWrapper(env, augment_prob=1.0)

        # High timeout as safety net - games should naturally terminate
        # Each game move = 2 env steps (two-stage actions)
        # Typical games: 80-150 moves = 160-300 steps
        # Safety limit: 2000 steps = 1000 game moves (should never reach this)
        env = TimeLimit(env, max_episode_steps=2000)
        env = Monitor(env)

        # Ensure action_masks is accessible through wrapper stack
        # (TimeLimit and Monitor don't forward this method)
        env = ActionMaskWrapper(env)

        return env
    return _init


if __name__ == "__main__":
    # Determine which stages to run
    start_stage_idx = 0
    if RESUME_FROM_STAGE:
        for idx, stage in enumerate(STAGE_PROGRESSION):
            if stage['name'] == RESUME_FROM_STAGE:
                start_stage_idx = idx
                break
        else:
            print(f"ERROR: Stage '{RESUME_FROM_STAGE}' not found in STAGE_PROGRESSION")
            print(f"Available stages: {[s['name'] for s in STAGE_PROGRESSION]}")
            exit(1)

    print("="*70)
    print("PROGRESSIVE CURRICULUM TRAINING (with Position Curriculum)")
    print("="*70)
    print(f"Total stages: {len(STAGE_PROGRESSION)}")
    print(f"Starting from: {STAGE_PROGRESSION[start_stage_idx]['name']}")
    if LOAD_MODEL_PATH:
        print(f"Loading model from: {LOAD_MODEL_PATH}")
    print("="*70)

    # Model persists across stages
    model = None
    previous_stage_model_path = LOAD_MODEL_PATH

    # Train through each stage sequentially
    for stage_idx in range(start_stage_idx, len(STAGE_PROGRESSION)):
        stage_config = STAGE_PROGRESSION[stage_idx]
        stage_name = stage_config['name']
        use_pos_curriculum = stage_config.get('use_position_curriculum', False)

        print("\n" + "="*70)
        print(f"STAGE {stage_idx + 1}/{len(STAGE_PROGRESSION)}: {stage_name}")
        print("="*70)
        print(f"Description: {stage_config['description']}")
        print(f"Environment: {stage_config['env_class'].__name__}")
        print(f"Timesteps: {stage_config['timesteps']:,}")
        print(f"Learning rate: {stage_config['learning_rate']}")
        print(f"Position curriculum: {'ON' if use_pos_curriculum else 'OFF'}")
        print(f"Train opponent: {stage_config['train_opponent'] or 'self-play'}")
        if stage_config['train_opponent'] == 'minimax':
            print(f"  Train depth: {stage_config['train_depth']}")
        print(f"Eval opponent: {stage_config['eval_opponent']}")
        if stage_config['eval_opponent'] == 'minimax':
            print(f"  Eval depth: {stage_config['eval_depth']}")
        print("="*70)

        # Create training environment with opponent
        print("\nCreating training environments (8 parallel)...")
        env = SubprocVecEnv([
            make_env_with_timeout(
                stage_config['env_class'],
                opponent_type=stage_config['train_opponent'],
                minimax_depth=stage_config.get('train_depth', 3),
                use_position_curriculum=use_pos_curriculum,
            ) for _ in range(8)
        ])

        # Create evaluation environment with DummyVecEnv (serial, prevents deadlocks)
        # Eval envs do NOT use position curriculum (always evaluate from standard starts)
        opponent_type = stage_config['eval_opponent']
        opponent_desc = f"{opponent_type or 'self-play'}"
        if opponent_type == 'minimax':
            opponent_desc += f" depth-{stage_config['eval_depth']}"

        print(f"Creating evaluation environments (4 sequential, vs {opponent_desc})...")
        eval_env = DummyVecEnv([
            make_env_with_timeout(
                stage_config['env_class'],
                render_mode='human',
                opponent_type=opponent_type,
                minimax_depth=stage_config.get('eval_depth', 3),
                use_position_curriculum=False,  # Eval always from standard starts
            ) for _ in range(4)
        ])

        # Load model from previous stage or create new one
        if previous_stage_model_path and model is None:
            print(f"\nLoading model from previous stage: {previous_stage_model_path}")
            try:
                model = MaskablePPO.load(
                    previous_stage_model_path,
                    env=env,
                    verbose=1,
                    tensorboard_log="./tblog/",
                    device="auto"
                )
                print("[SUCCESS] Model loaded successfully")
                print(f"  Updating learning rate to: {stage_config['learning_rate']}")
                model.learning_rate = stage_config['learning_rate']
            except Exception as e:
                print(f"[WARNING] Failed to load model: {e}")
                print("Starting from scratch instead...")
                model = None

        if model is None:
            print("\nCreating new model with CNN architecture...")
            model = MaskablePPO(
                    'CnnPolicy',
                    env,
                    verbose=1,

                    # Training schedule
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,

                    # Learning rate (constant per stage)
                    learning_rate=stage_config['learning_rate'],

                    # Exploration - INCREASED for stronger reward signals
                    # Higher entropy prevents premature convergence with clearer rewards
                    ent_coef=0.10 if stage_idx == 0 else 0.05,

                    # Discount and GAE
                    gamma=0.99,
                    gae_lambda=0.95,

                    # PPO specific
                    clip_range=0.2,
                    vf_coef=0.5,  # Monitor value_loss in tensorboard; reduce to 0.3 if too high
                    max_grad_norm=1.0,  # Increased from 0.5 to allow larger updates with stronger rewards

                    # CNN architecture - optimized for 49-action space with tactical rewards
                    # MicroscopeCNN: 344k params, good balance of capacity and speed
                    policy_kwargs={
                        "features_extractor_class": MicroscopeCNN,
                        "features_extractor_kwargs": {"features_dim": 256},
                        # Smaller heads: 49 actions needs less mapping than 1225
                        # 256 -> 64 -> 49 (policy) and 256 -> 64 -> 1 (value)
                        "net_arch": dict(pi=[64], vf=[64]),
                        "activation_fn": nn.ReLU,
                        "ortho_init": True,
                    },

                    tensorboard_log="./tblog/",
                    device="auto"
                )
        else:
            # Update learning rate for continuing training
            print(f"Continuing with existing model, updating LR to {stage_config['learning_rate']}")
            model.learning_rate = stage_config['learning_rate']
            model.set_env(env)

        # Set up callbacks
        NUM_ENVS = 8
        checkpoint_callback = CheckpointCallback(
            save_freq=50_000 // NUM_ENVS,
            save_path=f"./checkpoints/{stage_name}/",
            name_prefix=f"microscope_{stage_name}",
            verbose=2
        )

        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=f"./best_models/{stage_name}/",
            log_path=f"./eval_logs/{stage_name}/",
            eval_freq=50_000 // NUM_ENVS,
            n_eval_episodes=20,
            deterministic=True,
            verbose=1,
            warn=False
        )

        # Curriculum timestep callback - updates position curriculum progression
        callbacks = [checkpoint_callback, eval_callback]
        if use_pos_curriculum:
            callbacks.append(CurriculumTimestepCallback())

        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Checkpoints: ./checkpoints/{stage_name}/ (every 50k steps)")
        print(f"Best models: ./best_models/{stage_name}/")

        # Train the model
        try:
            model = model.learn(
                total_timesteps=stage_config['timesteps'],
                progress_bar=True,
                callback=callbacks,
                reset_num_timesteps=False  # Keep cumulative timesteps across stages
            )

            # Save stage completion model
            stage_model_path = f"models/{stage_name}_final"
            model.save(stage_model_path)
            print(f"\n[SUCCESS] Stage {stage_name} complete! Saved to: {stage_model_path}")

            # Set up for next stage
            previous_stage_model_path = stage_model_path

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            interrupt_model_path = f"models/{stage_name}_interrupted"
            print(f"Saving current model to: {interrupt_model_path}")
            model.save(interrupt_model_path)
            print("\nTo resume, set:")
            print(f"  RESUME_FROM_STAGE = '{stage_name}'")
            print(f"  LOAD_MODEL_PATH = '{interrupt_model_path}'")
            env.close()
            eval_env.close()
            exit(0)

        except Exception as e:
            print(f"\n\n[ERROR] Training failed at stage {stage_name}: {e}")
            import traceback
            traceback.print_exc()
            env.close()
            eval_env.close()
            raise

        # Clean up environments for this stage
        env.close()
        eval_env.close()
        print(f"Stage {stage_name} environments closed.\n")

    # All stages complete!
    print("\n" + "="*70)
    print("ALL STAGES COMPLETE!")
    print("="*70)
    print(f"\nFinal model: {previous_stage_model_path}")
    print("\nYour agent has completed the full curriculum:")
    print("  [x] Beat random opponents (aggressive rewards + curriculum)")
    print("  [x] Tactical play vs minimax depth-1")
    print("  [x] Advanced play vs minimax depth-3")
    print("  [x] Mastery vs minimax depth-5")
    print("\nNext steps:")
    print("  - Test with: python play.py")
    print("  - Try AlphaZero training with MCTS")
    print("="*70)
