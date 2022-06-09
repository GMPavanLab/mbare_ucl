import os
import shutil
import sys
from random import randint
import tempfile
import re
import multiprocessing
from itertools import repeat
from shared.simulations import run_parallel, init_process
import numpy as np
from pyemd import emd
from scipy.interpolate import interp1d

def sed_inplace(filename, pattern, repl):
    '''
    Perform the pure-Python equivalent of in-place `sed` substitution: e.g.,
    `sed -i -e 's/'${pattern}'/'${repl}' "${filename}"`.
    '''
    # For efficiency, precompile the passed regular expression.
    pattern_compiled = re.compile(pattern)

    # For portability, NamedTemporaryFile() defaults to mode "w+b" (i.e., binary
    # writing with updating). This is usually a good thing. In this case,
    # however, binary writing imposes non-trivial encoding constraints trivially
    # resolved by switching to text writing. Let's do that.
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        with open(filename) as src_file:
            for line in src_file:
                tmp_file.write(pattern_compiled.sub(repl, line))

    # Overwrite the original file with the munged temporary file in a
    # manner preserving file attributes (e.g., permissions).
    shutil.copystat(filename, tmp_file.name)
    shutil.move(tmp_file.name, filename)

def calculate_score_for_particle(args):
    """
    Calculate the score attributed to a given particle of the swarm, according to the results of potentially multiple
    parallel simulations.

        Parameters:
            args (dict): Contains the arguments passed via the config file and everything else required for processing.

        Returns:
            score (float): Score for this given particle of the swarm.

    """

    # TODO: Calculate a real score

    score = randint(0, 10)
    return score


def parallel_eval_function(parameters_for_swarm, ns):
    """
    This function receives a list of lists, which corresponds, in order, to the parameters suggested by FST-PSO
    for each particle of the swarm. It has to return a list of floats which correspond to the score for each particle
    in this swarm iteration.

        Parameters:
            parameters_for_swarm (list of lists of floats): Inner lists are the parameters per particle of the swarm.
            args (dict): Contains the arguments passed via the config file and everything else required for processing.

        Returns:
            scores_for_swarm (list of floats): Score for each particle of the swarm.

    """
    p_evaluation_number = []
    p_job_exec_dir = []

    ns.current_swarm_iteration += 1
    print(f"Starting with swarm iteration {ns.current_swarm_iteration}")

    ############################################
    # prepare files to run the swarm iteration #
    ############################################
    for particle_number, particle_parameters in enumerate(parameters_for_swarm):
        evaluation_number = (ns.current_swarm_iteration -1) * ns.particles_in_swarm + particle_number +1
        # create directories and copy input files
        os.makedirs(f"{ns.args['exec_folder']}/eval_{evaluation_number}", exist_ok=True)
        #make dirs
        for file in ['ff.itp','cg.top','equi.gro','MLA_CG.itp','index.ndx','md.mdp','plumed.dat','reweigh_cg.dat']:
            shutil.copy(f"simulation_inputs/{file}", f"{ns.args['exec_folder']}/eval_{evaluation_number}")



        # store folder name for parallel simulation executions
        p_job_exec_dir.append(f"{ns.args['exec_folder']}/eval_{evaluation_number}")
        p_evaluation_number.append(evaluation_number)

        # change forsifildi
        for particle_parameter, tuned_parameter in zip(particle_parameters, ns.tuned_parameters):
            pattern = tuned_parameter + '_ph'
            sed_inplace(f"{ns.args['exec_folder']}/eval_{evaluation_number}/ff.itp", pattern, "{:.3f}".format(particle_parameter))

    ###########################################
    # run fking simulations XDDDDDDDDDDDDDDDDD #
    ###########################################
    slots_states = multiprocessing.Array('i', ns.nb_slots, lock=True)
    for j in range(ns.nb_slots):  # multiprocessing.Array does NOT like list comprehension
        slots_states[j] = 1  # mark slot as available

    with multiprocessing.Pool(processes=ns.nb_slots, initializer=init_process, initargs=(slots_states,)) as pool:
        p_args = zip(repeat(ns), p_job_exec_dir, p_evaluation_number)
        p_res = pool.starmap(run_parallel, p_args)
        p_time_start_str, p_time_end_str, p_time_elapsed_str, esplosa = list(map(list, zip(*p_res)))


    #esplosa = [False]*ns.args['particles_in_swarm']
    ########################
    # SCORE EACH PARTICLE  # <---for the current swarm iteration
    ########################
    scores = [1000]*ns.args['particles_in_swarm']
    detailed_scores = np.empty((ns.args['particles_in_swarm'],len(ns.hists)),dtype=float)

    for i, (dir, particle_parameters) in enumerate(zip(p_job_exec_dir,parameters_for_swarm)):

        if not esplosa[i]:
            #scoro
            hists_cg = {}
            #read hist and normalize it
            score = 0
            for j,hist in enumerate(ns.hists):
                hists_cg[hist] = np.loadtxt(f"{dir}/H_{hist}.dat")[:,1]
                #normalizzo
                hists_cg[hist] = hists_cg[hist] / np.sum(hists_cg[hist])
                #scoro
                # emd(hist_aa, hist_cg, matrices[hist])
                detailed_scores[i,j] =  (100*emd(ns.AA_hists[hist], hists_cg[hist], ns.matrices[hist]))
            scores[i] = np.sum(detailed_scores[i])
        else:
            #exploded, put maximum score
            scores[i] = 1000

    #scrivere i log
    #log scores
    header = '\t'.join(ns.hists) + '\taggregate'
    np.savetxt(f"{ns.args['exec_folder']}/swarm_{ns.current_swarm_iteration}_scores.log",
               np.column_stack([detailed_scores, scores]), fmt='%2.2f', delimiter='\t', header=header)
    #log parametri
    header = '\t'.join(ns.tuned_parameters) + '\tscore'
    np.savetxt(f"{ns.args['exec_folder']}/swarm_{ns.current_swarm_iteration}_parameters.log",
               np.column_stack([parameters_for_swarm, scores]), fmt='%2.2f', delimiter='\t', header=header)

    print('swarm iteration ended')

    for particle_number, score in enumerate(scores):
        if score < ns.args['best_solution']['score']:
            ns.args['best_solution']['score'] = score
            ns.args['best_solution']['parameters'] = parameters_for_swarm[particle_number]
            ns.args['best_solution']['swarm_iter'] = ns.current_swarm_iteration
            ns.args['best_solution']['particle_iter'] = particle_number+1
    print(f"best solution at particle {ns.args['best_solution']['particle_iter']} of swarm {ns.args['best_solution']['swarm_iter']}, score = {ns.args['best_solution']['score']}")
    return scores  # this goes to FST-PSO for choosing the next sets of parameters supposed to improve the score

