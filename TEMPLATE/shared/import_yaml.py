import yaml

def read_input_file():
    with open(ns.cfg_file, 'r') as stream:
        args = yaml.safe_load(stream)
    ns.args = args


    # READ number of slots and checks (TO BE PUT IN check_all_arguments function) + handling
    if len(ns.args['nb_threads'].split()) != len(ns.args['gpu_ids'].split()):
        sys.exit('error: nb_threads and gpu_ids must have same length')
    ns.nb_slots = len(ns.args['nb_threads'].split())
    ns.slots_nts = ns.args['nb_threads'].split()
    ns.slots_gpu_ids = ns.args['gpu_ids'].split()

    return ns