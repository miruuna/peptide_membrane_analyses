import pandas as pd
from tqdm import tqdm

import MDAnalysis as mda 
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA

import warnings
warnings.filterwarnings('ignore') 

from utils.classes import Peptide

step_size = 500 

peptide = Peptide(
    "WF2_8_1", 
    "/Volumes/miru_backup/pleu_conc/pleu8/pleu8/md_0_1.xtc",
    "/Volumes/miru_backup/pleu_conc/pleu8/pleu8/md_0_1.tpr",
    peptide_number=8, amino_acid_count=25, step_size=step_size)


def hbonds_per_res(obj):
    """
    Calculate inter-peptide hydrogen bonds
    """
    hbonds = []
    u =  obj.u
    peptide_name = obj.pep_name
    pep_num = obj.peptide_number

    for pep in tqdm(range(1, pep_num)):
        res_range = obj.pep_dict[pep]
        for pep2 in range(pep+1, pep_num+1):
            res_range2 = obj.pep_dict[pep2]
            for res in range(res_range[0], res_range[1]+1):
                for res2 in range(res_range2[0], res_range2[1]+1):

                    h_bonds = HBA(u, update_selections=True, between=[f"resid {res}", f"resid {res2}"])
                    protein_hydrogens_sel = h_bonds.guess_hydrogens(f"resid {res}")
                    protein_acceptors_sel = h_bonds.guess_acceptors(f"resid {res}")

                    membrane_hydrogens_sel = h_bonds.guess_hydrogens(f"resid {res2}")
                    membrane_acceptors_sel = h_bonds.guess_acceptors(f"resid {res2}")

                
                    h_bonds.acceptors_sel = f"({protein_acceptors_sel}) or ({membrane_acceptors_sel})"

                    h_bonds.hydrogens_sel = f"({protein_hydrogens_sel}) or ({membrane_hydrogens_sel})"

                    h_bonds.run(start=-10, step=10)
                    
                    hydrogen_count = h_bonds.count_by_time()
                    time_count = h_bonds.times
                    for i in range(len(time_count)):
                        hbonds.append((pep, res, pep2, res2, hydrogen_count[i], time_count[i]))

    df = pd.DataFrame(hbonds) 
    df.columns = ["Peptide1", "Resid1", "Peptide2", "Resid2", "Hbonds", "Time"]
    df['Time'] = df['Time'].astype(float)/1000
    df['Time'] = df['Time'].astype(int)
    df = df.rename(columns={"Time": "Time (ns)"})
    df.to_csv(f"hbonds_pep_{peptide_name}.csv")
    return df

if __name__ == "__main__":
    hbonds_per_res(peptide)
