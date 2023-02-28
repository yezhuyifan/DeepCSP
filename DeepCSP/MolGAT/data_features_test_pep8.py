#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import itertools as it
import autograd.numpy as np

from rdkit.Chem import MolFromSmiles
from rdkit import Chem
import rdkit.Chem.rdPartialCharges as rdPartialCharges

import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_v2_behavior()



degrees = [1, 2, 3, 4, 5]

class memoize(object):
    """Memoize class to store and retrieve parameters"""
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            result = self.func(*args)
            self.cache[args] = result
            return result

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, add_Gasteiger):
    """Returns an array of atom features.
    Args:
        atom (rdchem.Atom): atom object
        add_Gasteiger (bool): whether to add Gasteiger charge

    Returns:
        np.array: array of atom features
    """
    symbol = one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H', 'Unknown']) # H?
    degree = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    n_h = one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    valence = one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
    aromatic = [atom.GetIsAromatic()]
    charge = [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    hybrid = one_of_k_encoding(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, 
                                                         Chem.rdchem.HybridizationType.SP2,
                                                         Chem.rdchem.HybridizationType.SP3, 
                                                         Chem.rdchem.HybridizationType.SP3D, 
                                                         Chem.rdchem.HybridizationType.SP3D2])
    gasteiger = [add_Gasteiger]
    return np.array(symbol + degree + n_h + valence + aromatic + charge + hybrid + gasteiger)

def bond_features(bond):
    """Returns bond features in an array"""
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])

def num_atom_features():
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a, 0.0))

def num_bond_features():
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


class MolGraph(object):
    """A class to store and manipulate molecules"""
    def __init__(self):
        self.nodes = {}

    def new_node(self, ntype, features=None, rdkit_ix=None):
        """Create a new node"""
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        """Add a subgraph to the molecule"""
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def sort_nodes_by_degree(self, ntype):
        """Sort the nodes by degree"""
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        """Return an array containing the features of the nodes"""
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        """Return an array containing the RDKit index of the nodes"""
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        """Return a list of the neighbor indices for each node"""
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]


class Node(object):
    # Declare the attributes that can be assigned to a Node object
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']

    # Initialize a Node object
    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    # Add a list of neighbors to the Node
    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    # Return a list of neighbors with the given node type
    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]


def graph_from_smiles_tuple(smiles_tuple):
    #Convert SMILES tuple to graph
    graph_list = [graph_from_smiles(s) for s in smiles_tuple]
    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)

    big_graph.sort_nodes_by_degree('atom')
    return big_graph

def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = MolFromSmiles(smiles)
    # print(smiles)
    # Check if parsing of SMILES string was successful
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    
    # Compute Gasteiger Charges
    rdPartialCharges.ComputeGasteigerCharges(mol)

    # Add atoms to graph
    for atom in mol.GetAtoms():
        add_Gasteiger = float(atom.GetProp('_GasteigerCharge'))
        if np.isnan(add_Gasteiger) or np.isinf(add_Gasteiger):
            add_Gasteiger = 0.0
        new_atom_node = graph.new_node('atom', features=atom_features(atom, add_Gasteiger), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node

    # Add bonds to graph
    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        # Add neighbors to new bond node
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    # Add molecule node to graph
    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])

    return graph


@memoize

def array_rep_from_smiles(smiles):
    # Take SMILES notation and return array representation
    molgraph = graph_from_smiles_tuple(smiles)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'),
                'rdkit_ix'      : molgraph.rdkit_ix_array()} 
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep

def trans(substances):
    # Transforms substances into the format for the graph neural network
    # Make array of substance and atoms indices
    substance_atoms = []
    for substance_i, atoms_i in enumerate(substances['atom_list']):
         substance_atoms += [ [substance_i, atoms_ij] for atoms_ij in atoms_i]
    substance_atoms = np.array(substance_atoms)

    # Make sparse tensor for substance and atoms
    n_atoms = substance_atoms.shape[0]
    substance_atom_indices = substance_atoms
    substance_atom_values = tf.fill(tf.expand_dims(tf.to_int32(n_atoms), 0), 1.0)
    substance_atom_shape = [substance_i+1, n_atoms]
    substance_atoms_tensor = tf.SparseTensor(substance_atom_indices, 
                                             substance_atom_values, substance_atom_shape)
    substances['substance_atoms'] = substance_atoms_tensor
    
    # Convert atom and bond features to float32
    substances["atom_features"] = substances["atom_features"].astype(np.float32)
    substances["bond_features"] = substances["bond_features"].astype(np.float32)
    
    # Make array of compounds and atoms RDKit indices
    compounds_rdkit_ix = []
    for substance_i, atoms_i in enumerate(substances["atom_list"]):
        atom_rdkit_ix = substances["rdkit_ix"][atoms_i]
        compounds_rdkit_ix += [[substance_i, atom_rdkit_ix_i] for atom_rdkit_ix_i in atom_rdkit_ix]
    compounds_rdkit_ix = np.array(compounds_rdkit_ix)
    substances["compounds_rdkit_ix"] = compounds_rdkit_ix
            
    # Make arrays of atom and bond neighbors
    for degree in degrees:
        atom_neighbors = substances[('atom_neighbors', degree)]
        substances['atom_neighbors_{}'.format(degree)] = atom_neighbors
        substances.pop(('atom_neighbors', degree)) 
    for degree in degrees:
        bond_neighbors = substances[('bond_neighbors', degree)]
        substances['bond_neighbors_{}'.format(degree)] = bond_neighbors
        substances.pop(('bond_neighbors', degree))

    # Make array for RNN input
    N_compounds = max(substances["compounds_rdkit_ix"][:, 0])+1
    N_max_seqlen = max(substances["compounds_rdkit_ix"][:, 1]) + 1
    rnn_raw_input = np.zeros((N_compounds, N_max_seqlen), dtype=np.int64) + n_atoms
    for i, atoms_i in enumerate(substances['atom_list']):
        for j, a_j in enumerate(atoms_i):
            rnn_raw_input[i, j] = a_j
    substances["rnn_raw_input"] = rnn_raw_input
    
    substances['atom_list'] = np.array(substances['atom_list'])
    return substances