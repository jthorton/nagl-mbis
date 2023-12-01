# models for the nagl run

import abc
from typing import Dict, List, Literal, Optional

import torch
from nagl.training import DGLMoleculeLightningModel
from nagl.molecules import DGLMolecule
from rdkit import Chem


class MBISGraphModel(DGLMoleculeLightningModel):
    "A wrapper to make it easy to load and evaluate models"

    def compute_properties(self, molecule: Chem.Mol) -> Dict[str, torch.Tensor]:
        dgl_molecule = DGLMolecule.from_rdkit(molecule, self.config.model.atom_features, self.config.model.bond_features)

        return self.forward(dgl_molecule)
