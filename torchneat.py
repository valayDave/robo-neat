# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import os

import click
import gym
from gym.wrappers import FlattenObservation, FilterObservation
import neat
from py_neat.pytorch_neat.multi_env_eval import MultiEnvEvaluator
from py_neat.pytorch_neat.neat_reporter import LogReporter
from py_neat.pytorch_neat.recurrent_net import RecurrentNet
import visualize
import matplotlib.pyplot as plt
import pickle
import json
max_env_steps = 200

import numpy as np


class EnvEvaluator:
    def __init__(self, make_net, activate_net, batch_size=1, max_env_steps=None, make_env=None, envs=None):
        if envs is None:
            self.envs = [make_env() for _ in range(batch_size)]
        else:
            self.envs = envs
        self.make_net = make_net
        self.activate_net = activate_net
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps

    def eval_genome(self, genome, config, debug=False):
        net = self.make_net(genome, config, self.batch_size)

        fitnesses = np.zeros(self.batch_size)
        states = [env.reset() for env in self.envs]
        dones = [False] * self.batch_size

        step_num = 0
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break
            if debug:
                actions = self.activate_net(
                    net, states, debug=True, step_num=step_num)
            else:
                actions = self.activate_net(net, states)
            assert len(actions) == len(self.envs)
            for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
                if not done:
                    state, reward, done, _ = env.step(action)
                    fitnesses[i] += reward
                    if not done:
                        states[i] = state
                    dones[i] = done
            if all(dones):
                break

        return sum(fitnesses) / len(fitnesses)


def make_env(with_monitor=False,folder_name='results'):
    env = gym.make("FetchReach-v1")
    env.env.reward_type = 'dense'
    env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
    if with_monitor:
        env = gym.wrappers.Monitor(env, folder_name, force=True)
    return env 


def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)


def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return outputs


@click.command()
@click.option("--n_generations", type=int, default=1000)
@click.option("--simulation_steps", type=int, default=200)
@click.option("--simulation_runs", type=int, default=200)
def run(n_generations,simulation_steps,simulation_runs):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "torchneat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = EnvEvaluator(
        make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps
    )

    def eval_genomes(genomes, config):
        for _, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("torchneat.log", evaluator.eval_genome)
    pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)
    visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")
    best_genomes = stats.best_unique_genomes(10)

    best_networks = []
    num_dones = {}
    network_scores = {}

    
    for g in best_genomes:
        best_networks.append((make_net(g, config,evaluator.batch_size),g.key))
        num_dones[g.key] = 0
        network_scores[g.key] = [] 
        
    for sim_num in range(simulation_runs):

        for network,key in best_networks:
            env = make_env(with_monitor=True,folder_name='results/'+str(key)+'/'+str(sim_num))
            print("Testing Network ",key)
            observation = env.reset()
            score = 0
            while True:
                action = activate_net(network,[observation])
                observation, reward, done, info = env.step(action[0])
                score+=reward
                if done:
                    num_dones[key]+=1
                    break
            env.close()
            network_scores[key].append(score)
    print("Final Dones :",num_dones)               
    print("Final Score :",network_scores)
    final_object = {
        'num_dones':num_dones,
        'network_scores':network_scores
    }
    with open('results/final_data.json', 'w') as outfile:
        json.dump(final_object,outfile)

    for n, g in enumerate(best_genomes):
        name = 'results/winner-{0}'.format(n)
        with open(name+'.pickle', 'wb') as f:
            pickle.dump(g, f)

        visualize.draw_net(config, g, view=False, filename=name+"-net.gv")
        visualize.draw_net(config, g, view=False, filename=name+"-net-enabled.gv",
                            show_disabled=False)
        visualize.draw_net(config, g, view=False, filename=name+"-net-enabled-pruned.gv",
                            show_disabled=False, prune_unused=True)
    visualize.plot_species(stats)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter