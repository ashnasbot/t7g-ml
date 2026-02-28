"""
Wrapper to ensure action_masks method is accessible through wrapper stack.

TimeLimit and Monitor wrappers don't forward the action_masks method,
so we need this wrapper to make it accessible to evaluate_policy.
"""
import gymnasium as gym


class ActionMaskWrapper(gym.Wrapper):
    """
    Ensures action_masks method is accessible through wrapper stack.

    Place this as the outermost wrapper (after Monitor) to ensure
    action_masks is available to evaluate_policy and other external callers.
    """

    def action_masks(self):
        """Forward action_masks call to the wrapped environment"""
        # Walk down the wrapper stack to find an env with action_masks
        env = self.env
        while hasattr(env, 'env'):
            if hasattr(env, 'action_masks') and callable(env.action_masks):
                return env.action_masks()
            env = env.env

        # If we get here, the base env should have action_masks
        if hasattr(env, 'action_masks'):
            return env.action_masks()

        raise AttributeError("No action_masks method found in environment stack")
