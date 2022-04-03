# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import isaacgym

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from utils.reformat import omegaconf_to_dict, print_dict
from utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator

from utils.utils import set_np_formatting, set_seed

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import yaml
import torch


## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    # ensure checkpoints can be specified as relative paths
    assert cfg.checkpoint
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # CamelCase to snake_case
    import re
    task_file = re.sub(r'(?<!^)(?=[A-Z])', '_', cfg.task_name).lower()
    from pydoc import locate
    task_class = locate(f'isaacgymenvs.tasks.{task_file}.{cfg.task_name}')

    # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
    # We use the helper function here to specify the environment config.
    create_rlgpu_env = get_rlgames_env_creator(
        omegaconf_to_dict(cfg.task),
        task_class,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        multi_gpu=cfg.multi_gpu,
    )

    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    })

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    experiment_dir = os.path.join('runs', cfg.train.params.config.name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Begin exporting ONNX for inference
    print("Exporting!")

    # Load model from checkpoint
    player = runner.create_player()
    player.restore(runner.load_path)

    # Create dummy observations tensor for tracing torch model
    obs_shape = player.obs_shape
    actions_num = player.actions_num
    obs_num = obs_shape[0]
    dummy_input = torch.zeros(obs_shape, device='cuda:0')

    # Simplified network for actor inference
    # Tested for continuous_a2c_logstd
    class ActorModel(torch.nn.Module): 
        def __init__(self, a2c_network):
            super().__init__()
            self.a2c_network = a2c_network
        def forward(self,x):
            x = self.a2c_network.actor_mlp(x)
            x = self.a2c_network.mu(x)
            return x
    model = ActorModel(player.model.a2c_network)

    # Since rl_games uses dicts, we can flatten the inputs and outputs of the model: see https://github.com/Denys88/rl_games/issues/92
    # Not necessary with the custom ActorModel defined above, but code is included here if needed
    import flatten as flatten
    with torch.no_grad():
        adapter = flatten.TracingAdapter(model, dummy_input, allow_non_tensor=True)
        torch.onnx.export(adapter, adapter.flattened_inputs, f"{cfg.checkpoint}.onnx", verbose=True, 
            input_names = ['observations'],
            output_names = ['actions']) # outputs are mu (actions), sigma, value
        traced = torch.jit.trace(adapter, dummy_input,check_trace=True)
        flattened_outputs = traced(dummy_input)
    print(f"Exported to {cfg.checkpoint}.onnx!")

    # Print dummy output and model output (make sure these have the same values)
    print("Flattened outputs: ", flattened_outputs)
    print(model.forward(dummy_input))

    print("# Observations: ", obs_num)
    print("# Actions: ", actions_num)


if __name__ == "__main__":
    launch_rlg_hydra()
