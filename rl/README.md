# HUD environments for Reinforcement Learning

## Options

- **ART**: [ART](https://github.com/OpenPipe/ART) is a library for training multi-step agents. You can use ART to build, train, and benchmark RL agents, and it can be integrated with custom environments such as those provided in this repository. See [art/README.md](art/README.md) for details.

- **Verifiers**: The primary supported option here. Use the [hud-vf-gym](https://github.com/hud-evals/hud_vf_gym) module to access CUA environments built with HUD MCP for RL agent training and evaluation. See [verifiers/README.md](verifiers/README.md) for installation and usage instructions.

For both options we support running evaluations for a baseline assessment, and then RL training on this dataset!

