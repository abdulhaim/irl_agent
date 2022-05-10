import random
import torch
import numpy as np
from misc.utils import make_env, generate_parameters

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialization
    config = generate_parameters(mode="phaseII")

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = make_env(config)

    if not config.phaseII:
        from agents.phase_I import PhaseIAgent
        agent = PhaseIAgent(config, env, device)
    else:
        from agents.phase_II import PhaseIIAgent
        agent = PhaseIIAgent(config, env, device)

    # Train model
    if not config.phaseII:
        agent.train(env)
    else:
        agent.inverse_train()
        agent.test_agent(env)

