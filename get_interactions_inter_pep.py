import MDAnalysis as mda
import MDAnalysis
import MDAnalysis.transformations as trans

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


from utils.classes import Peptide

step_size = 500


peptide = Peptide(
    "WF2_8_1", 
    "/Volumes/miru_backup/pleu_conc/pleu8/pleu8/md_0_1.xtc",
    "/Volumes/miru_backup/pleu_conc/pleu8/pleu8/md_0_1.tpr",
    peptide_number=8, amino_acid_count=25, step_size=step_size)

def get_interactions(obj):
    u = obj.u
    obj_name = obj.pep_name
    pep_num= obj.peptide_number

    frames, n_frames = obj.load_traj()

    all_list = []
    for _, ts in tqdm(enumerate(u.trajectory[frames]), total=n_frames):
        pair_done = []
        for pep in range(1, pep_num):
            res_range = obj.pep_dict[pep]
            for pep2 in range(pep+1, pep_num+1):
                res_range2 = obj.pep_dict[pep2]
                for res in range(res_range[0], res_range[1]+1):
                    for res2 in range(res_range2[0], res_range2[1]+1):
                        selection1 = u.select_atoms('resid %s'%res).center_of_mass()
                        selection2 = u.select_atoms('resid %s'%res2).center_of_mass()
                        dist = MDAnalysis.lib.distances.distance_array(
                            selection1, selection2, box=u.dimensions, result=None, backend='serial')
                        min_dist = dist
                        all_list.append((res, res2, pep, pep2, 
                                         min_dist[0].astype(float)[0], int(u.trajectory.time/1000)))
            pair_done.append((pep, pep2))

    df = pd.DataFrame(all_list, columns=['Res1','Res2','Peptide1', 'Peptide2', 'mindist', 'Time(ns)'])
    df.to_csv(f"interactions_{obj_name}.csv")
    return df


if __name__ == "__main__":
    print("Starting calculations")
    get_interactions(peptide)