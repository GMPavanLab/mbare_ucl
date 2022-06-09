#!/usr/bin/env python3

import os, time, copy, sys
from fstpso import FuzzyPSO
import yaml
from shared.fitness_function import parallel_eval_function
from argparse import Namespace
import numpy as np
from scipy.interpolate import interp1d
from shared.fs import import_experimental_data, check_slot_consistency
import pickle

ns = Namespace()

with open("input.yaml", 'r') as stream:
    ns.args = yaml.safe_load(stream)
check_slot_consistency(ns)

ns.process_alive_time_sleep = 20  # nb of seconds between process alive check cycles
ns.process_alive_nb_cycles_dead = int(ns.args['sim_kill_delay'] / ns.process_alive_time_sleep)

# OUTPUT FOLDER - checkpoints
#by default we assume that there are no checkpoints
if ns.args['exec_folder'] == '':
    # ns.args['exec_folder'] = time.strftime('EAU_OPTI_STARTED_%d-%m-%Y_%Hh%Mm%Ss')
    ns.args['exec_folder'] = sys.argv[1]
if ns.args['continue_from_checkpoint']:
    ns.fstpso_checkpoint = f"{ns.args['exec_folder']}/fstpso_checkpoint.obj"
    # look for last swarm iteration
    g = open(ns.fstpso_checkpoint, "rb")
    obj = pickle.load(g)
    ns.current_swarm_iteration = obj._Iteration + 2
else:
    ns.fstpso_checkpoint = None
    ns.current_swarm_iteration = 0
os.makedirs(ns.args['exec_folder'], exist_ok=True)

ns.tuned_parameters = ['eps_hata',
'eps_haha',
'eps_haoa',
'eps_haca',
'eps_tata',
'eps_taoa',
'eps_taca',
'eps_caca',
'eps_caoa',
'eps_oaoa']


ns.search_space = np.zeros((len(ns.tuned_parameters), 2))
# each row of the search space is a variable
for index, row in enumerate(ns.search_space):
    #start from opc3 +/- 5sigma
    row[0] = 0.2
    row[1] = 10
    if index == 0:
        row[0] = 0.2
        row[1] = 15


# number of particles of the swarm
ns.particles_in_swarm = ns.args['particles_in_swarm']

#######
#read AA_data
ns.hists = ['hata','ha1ha2']
ns.AA_hists = {}
ns.bins = {}
ns.matrices = {}
for hist in ns.hists:
    #read bins
    ns.bins[hist] = np.loadtxt(f"AA_data/H_{hist}.dat")[:,0]
    #read and normalize aa histograms
    ns.AA_hists[hist] = np.loadtxt(f"AA_data/H_{hist}.dat")[:, 1]
    ns.AA_hists[hist] = ns.AA_hists[hist]/ np.sum(ns.AA_hists[hist])
    # define wasserstein distance matrices #
    m_euclidean = np.empty([len(ns.bins[hist]), len(ns.bins[hist])], dtype=float)
    for i in range(len(ns.bins[hist])):
        for j in range(len(ns.bins[hist])):
            if i == j:
                m_euclidean[i, j] = 0
            else:
                if j > i:
                    m_euclidean[i, j] = (ns.bins[hist][j] - ns.bins[hist][i])
                else:
                    m_euclidean[i, j] = (ns.bins[hist][i] - ns.bins[hist][j])
    ns.matrices[hist] = m_euclidean


ns.args['best_solution'] = {
    'score': np.inf,
    'parameters': [],
    'swarm_iter': 0,
    'particle_iter': 0
}

FP = FuzzyPSO()
FP.set_search_space(ns.search_space)
FP.set_swarm_size(ns.particles_in_swarm)  # if not set, then the choice of nb of particles is automatic
FP.set_parallel_fitness(fitness=parallel_eval_function, arguments=ns, skip_test=True)
result = FP.solve_with_fstpso(max_iter=ns.args['max_swarm_iterations'],
                              initial_guess_list=None,
                              max_iter_without_new_global_best=ns.args['max_swarm_iterations_without_improvement'],
                              restart_from_checkpoint=ns.fstpso_checkpoint,
                              save_checkpoint=f"{ns.args['exec_folder']}/fstpso_checkpoint.obj",
                              verbose=True
                              )
# print(f"Best solution found in swarm iter {ns.args['best_solution']['swarm_iter']} at particle "
#        f"{ns.args['best_solution']['particle_iter']} with score {ns.args['best_solution']['score']}")
#  print(f"Best solution: {ns.args['best_solution']['parameters']}")
