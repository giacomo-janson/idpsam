import os
import numpy as np
import mdtraj
from sam.data.sequences import aa_three_letters, aa_one_to_three_dict


def get_ca_topology(sequence):
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    for res in sequence:
        res_obj = topology.add_residue(aa_one_to_three_dict[res], chain)
        topology.add_atom("CA", mdtraj.core.topology.elem.carbon, res_obj)
    return topology


def slice_traj_to_com(traj, get_xyz=True):
    ha_ids = [a.index for a in traj.topology.atoms if \
              a.residue.name in aa_three_letters and \
              a.element.symbol != "H"]
    ha_traj = traj.atom_slice(ha_ids)
    residues = list(ha_traj.topology.residues)
    com_xyz = np.zeros((ha_traj.xyz.shape[0], len(residues), 3))
    for i, residue_i in enumerate(residues):
        ha_ids_i = [a.index for a in residue_i.atoms]
        masses_i = np.array([a.element.mass for a in residue_i.atoms])
        masses_i = masses_i[None,:,None]
        tot_mass_i = masses_i.sum()
        com_xyz_i = np.sum(ha_traj.xyz[:,ha_ids_i,:]*masses_i, axis=1)/tot_mass_i
        com_xyz[:,i,:] = com_xyz_i
    if get_xyz:
        return com_xyz
    else:
        return mdtraj.Trajectory(
            xyz=com_xyz,
            topology=get_ca_topology(
                sequence="".join([r.code for r in ha_traj.topology.residues])
            ))