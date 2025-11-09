import numpy as np
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
def semi_gradient_n_step_td(env, gamma, pi, n, alpha, V, num_episode):
    """
    n-step semi-gradient TD for estimating v_pi.

    env: environment
    gamma: discount factor
    pi: policy (has action(state) method)
    n: n-step horizon
    alpha: step size
    V: value function with approximation (has __call__ and update)
    num_episode: number of episodes
    """
    for _ in range(num_episode):
        # Reset environment
        s = env.reset()
        # Handle gym/gymnasium API difference
        if isinstance(s, tuple):
            s = s[0]

        states = [s]
        rewards = [0.0]  # r_0 dummy so that rewards[t] = r_t
        T = float("inf")
        t = 0

        while True:
            if t < T:
                # Select action from target policy
                a = pi.action(states[t])

                step_out = env.step(a)
                # Support both (s', r, done, info) and (s', r, terminated, truncated, info)
                if len(step_out) == 5:
                    s_next, r, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    s_next, r, done, _ = step_out

                rewards.append(r)
                states.append(s_next)

                if done:
                    T = t + 1  # episode ends

            tau = t - n + 1  # time whose estimate we update
            if tau >= 0:
                # Compute n-step return G_t^n
                G = 0.0
                # Sum of rewards from tau+1 to min(tau+n, T)
                upper = min(tau + n, T)
                for i in range(tau + 1, upper + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]

                # Bootstrap if n-step target does not reach terminal
                if tau + n < T:
                    G += (gamma ** n) * V(states[tau + n])

                # Semi-gradient update at state s_tau
                V.update(alpha, G, states[tau])

            t += 1
            if tau == T - 1:
                break


