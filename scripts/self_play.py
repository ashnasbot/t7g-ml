from env.env_virt import MicroscopeEnv

from sb3_contrib import MaskablePPO


if __name__ == "__main__":
    seed = 42

    # env = env_virt_AEC.env(render_mode="human")
    env = MicroscopeEnv(render_mode=None)

    print(f"Starting training on {str(env.metadata['name'])}.")

    model = MaskablePPO('MlpPolicy',
                        env,
                        verbose=2,
                        n_steps=1024,
                        policy_kwargs={
                            "net_arch": dict(pi=[2048, 2048], vf=[256, 256]),
                        },
                        tensorboard_log="./tblog/",
                        batch_size=512
                        )
    model.set_random_seed(seed)
    model.learn(total_timesteps=2**20, progress_bar=True, reset_num_timesteps=False)

    # model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()
