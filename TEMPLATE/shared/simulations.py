import os
import sys
import signal
import subprocess
import time
from datetime import datetime
from shared.context_managers import working_dir
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from scipy.interpolate import interp1d
import numpy as np


# build gromacs command with arguments
def gmx_args(gmx_cmd, gpu_id):
    if gpu_id != '':
        gmx_cmd += f" -gpu_id {gpu_id}"
    return gmx_cmd


def run_sims(ns, gpu_id):
    gmx_start = datetime.now().timestamp()
    esplosa = False

    # grompp -- PROD
    gmx_cmd = f"{ns.args['gmx_path']} grompp -f md.mdp  -c equi.gro  -p cg.top -o metad.tpr"
    gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    gmx_out = gmx_process.communicate()[1].decode()

    if os.path.isfile('metad.tpr'):
        # mdrun -- PROD
        run_tstart = datetime.now().timestamp()
        gmx_cmd = f"{ns.args['gmx_path']} mdrun -deffnm metad -nt {ns.args['nt']} -plumed plumed.dat"
        gmx_cmd = gmx_args(gmx_cmd, gpu_id)
        gmx_process = subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       start_new_session=True)  # create a process group for the MD run

        # check if PROD run is stuck because of instabilities
        last_log_file_size = 0
        while gmx_process.poll() is None:  # while process is alive
            time.sleep(ns.process_alive_time_sleep)
            if ((datetime.now().timestamp() - run_tstart)/3600) > ns.args['walltime_run']:
                esplosa = True
                os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)
            if os.path.isfile('metad.log'):
                log_file_size = os.path.getsize('metad.log')
                if last_log_file_size == log_file_size:
                    os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)
                else:
                    last_log_file_size = log_file_size
    else:
        esplosa = True
        # sim_status = 'MD run failed (simulation crashed)'
        print(
            'Gmx grompp failed at the PRODUCTION step, see gmx error message above\nPlease check the parameters of the provided MDP file\n')

    if os.path.isfile('metad.gro'):
        #FARE REWEIGHT
        #plumed driver --mf_xtc metad.xtc --plumed reweigh_cg.dat --kt 2.477
        cmd = f"plumed driver --mf_xtc metad.xtc --plumed reweigh_cg.dat --kt 2.477"
        gmx_process = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gmx_out = gmx_process.communicate()[1].decode()

    else:
        esplosa = True

    gmx_time = datetime.now().timestamp() - gmx_start

    return gmx_time, esplosa


# for making a shared multiprocessing.Array() to handle slots states when running simulations in LOCAL (= NOT HPC)
def init_process(arg):
    global g_slots_states
    g_slots_states = arg


def run_parallel(ns, job_exec_dir, evaluation_number):
    while True:
        time.sleep(1)
        for i in range(len(g_slots_states)):
            if g_slots_states[i] == 1:  # if slot is available
                # print(f'Starting simulation for evaluation {evaluation_number}')
                g_slots_states[i] = 0  # mark slot as busy
                gpu_id = ns.gpu_ids[i]
                with working_dir(job_exec_dir):
                    gmx_time, esplosa = run_sims(ns, gpu_id)
                g_slots_states[i] = 1  # mark slot as available
                # print(f'Finished simulation for particle {nb_eval_particle} with {lipid_code} {temp} on slot {i + 1}')

                time_start_str, time_end_str = '', ''  # NOTE: this is NOT displayed anywhere atm & we don't care much
                time_elapsed_str = time.strftime('%H:%M:%S', time.gmtime(round(gmx_time)))

                return time_start_str, time_end_str, time_elapsed_str, esplosa


