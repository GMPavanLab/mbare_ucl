import sys
from argparse import Namespace
import numpy as np
from scipy.interpolate import interp1d

def import_experimental_data(ns):
    ns.experimental_rdfs = {}
    ns.experimental_density = {}
    ns.experimental_epsilon = {}
    rdfs = ['goo','goh','ghh']
    for temperature in ns.args['temperatures']:
        #1 import rdfs
        ns.experimental_rdfs[temperature] = {}
        if temperature == 298:
            columns_idx = [1,3,5]
            relative_path = "ref_exp/298rdfs"
            ff = np.loadtxt(f'{relative_path}/298K_parsed.txt')
            for rdf, column_idx in zip(rdfs,columns_idx):
                rfd_interpolated = interp1d(ff[:, 0], ff[:, column_idx], kind='cubic')
                ns.experimental_rdfs[temperature][rdf] = rfd_interpolated(ns.rdf_distance)
        else:
            relative_path = f'ref_exp/india/{temperature}K'
            for rdf in rdfs:
                simulated_rdf = np.loadtxt(f"{relative_path}/{rdf}.dat")
                ns.experimental_rdfs[temperature][rdf] = np.ascontiguousarray(simulated_rdf[:, 1])
        # else:
        #     relative_path = f'ref_exp/india/resampling_precise'
        #     for rdf in rdfs:
        #         simulated_rdf = np.loadtxt(f"{relative_path}/{rdf}/{rdf}{temperature}K.dat")
        #         ns.experimental_rdfs[temperature][rdf] = np.ascontiguousarray(simulated_rdf[:,1])

        #2 import density
        relative_path = "ref_exp/parsed"
        ff_density = np.loadtxt(f'{relative_path}/density.txt')
        dens_iterp = interp1d(ff_density[:, 0], ff_density[:, 1], kind='cubic')
        ns.experimental_density[temperature] = dens_iterp(temperature)

        # 3 import epsilon
        relative_path = "ref_exp/parsed"
        ff_epsilon = np.loadtxt(f'{relative_path}/epsilon.txt')
        eps_iterp = interp1d(ff_epsilon[:, 0], ff_epsilon[:, 1], kind='cubic')
        ns.experimental_epsilon[temperature] = eps_iterp(temperature)

    return

def check_slot_consistency(ns):
    ns.nb_slots = ns.args['nb_slots']
    ns.nt = ns.args['nt']
    if ns.args['gpu_ids'] == '':
        ns.gpu_ids = ['']*ns.nb_slots
    else:
        ns.gpu_ids = ns.args['gpu_ids'].split(' ')
        if len(ns.gpu_ids) != ns.nb_slots:
            sys.exit('now the code must assign each slot a gpu. nb_slots must be equal to len(gpu_ids)')
    return ns


