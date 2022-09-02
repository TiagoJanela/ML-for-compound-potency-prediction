import abc
from typing import *
from collections import defaultdict

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import scipy.sparse as sparse
from rdkit import DataStructs


def construct_check_mol_list(smiles_list: List[str]) -> List[Chem.Mol]:
    mol_obj_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    if None in mol_obj_list:
        invalid_smiles = []
        for smiles, mol_obj in zip(smiles_list, mol_obj_list):
            if not mol_obj:
                invalid_smiles.append(smiles)
        invalid_smiles = "\n".join(invalid_smiles)
        raise ValueError(f"Following smiles are not valid:\n {invalid_smiles}")
    return mol_obj_list


def construct_check_mol(smiles: str) -> Chem.Mol:
    mol_obj = Chem.MolFromSmiles(smiles)
    if not mol_obj:
        raise ValueError(f"Following smiles are not valid: {smiles}")
    return mol_obj


class AtomEnvironment(NamedTuple):
    """"A Class to store environment-information for morgan-fingerprint features"""
    central_atom: int  # Atom index of central atom
    radius: int  # bond-radius of environment
    environment_atoms: Set[int]  # set of all atoms within radius


class Fingerprint(metaclass=abc.ABCMeta):
    """A metaclass representing all fingerprint subclasses."""

    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def n_bits(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, mol_obj_list: List[Chem.Mol]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def fit_transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        raise NotImplementedError

    def fit_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        self.fit(mol_obj_list)

    def fit_transform_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.fit_transform(mol_obj_list)

    def transform_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.transform(mol_obj_list)


class _MorganFingerprint(Fingerprint):
    def __init__(self, radius: int = 2, use_features=False):
        super().__init__()
        self._n_bits = None
        self._use_features = use_features
        if isinstance(radius, int) and radius >= 0:
            self._radius = radius
        else:
            raise ValueError(f"Number of bits has to be a positive integer! (Received: {radius})")

    def __len__(self):
        return self.n_bits

    @property
    def n_bits(self) -> int:
        if self._n_bits is None:
            raise ValueError("Number of bits is undetermined!")
        return self._n_bits

    @property
    def radius(self):
        return self._radius

    @property
    def use_features(self) -> bool:
        return self._use_features

    @abc.abstractmethod
    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict:
        raise NotImplementedError

    def bit2atom_mapping(self, mol_obj: Chem.Mol) -> Dict[int, List[AtomEnvironment]]:  # use
        bit2atom_dict = self.explain_rdmol(mol_obj)
        result_dict = defaultdict(list)

        # Iterating over all present bits and respective matches
        for bit, matches in bit2atom_dict.items():  # type: int, tuple
            for central_atom, radius in matches:  # type: int, int
                if radius == 0:
                    result_dict[bit].append(AtomEnvironment(central_atom, radius, {central_atom}))
                    continue
                env = Chem.FindAtomEnvironmentOfRadiusN(mol_obj, radius, central_atom)
                amap = {}
                _ = Chem.PathToSubmol(mol_obj, env, atomMap=amap)
                env_atoms = amap.keys()
                assert central_atom in env_atoms
                result_dict[bit].append(AtomEnvironment(central_atom, radius, set(env_atoms)))

        # Transforming defaultdict to dict
        return {k: v for k, v in result_dict.items()}


class FoldedMorganFingerprint(_MorganFingerprint):
    def __init__(self, n_bits=2048, radius: int = 2, use_features=False):
        super().__init__(radius=radius, use_features=use_features)
        if isinstance(n_bits, int) and n_bits >= 0:
            self._n_bits = n_bits
        else:
            raise ValueError(f"Number of bits has to be a positive integer! (Received: {n_bits})")

    def fit(self, mol_obj_list: List[Chem.Mol]) -> None:
        pass

    def transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        fingerprints = []
        for mol in mol_obj_list:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, useFeatures=self._use_features,
                                                       nBits=self._n_bits)
            fingerprints.append(sparse.csr_matrix(fp))
        return sparse.vstack(fingerprints)

    def transform_to_bv(self, mol_obj_list: List[Chem.Mol]):
        fingerprints = []
        for mol in mol_obj_list:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, useFeatures=self._use_features,
                                                       nBits=self._n_bits)
            fingerprints.append(fp)
        return fingerprints

    def fit_transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        return self.transform(mol_obj_list)

    def explain_rdmol(self, mol_obj: Chem.Mol) -> dict:
        bi = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(mol_obj, self.radius, useFeatures=self._use_features, bitInfo=bi,
                                                  nBits=self._n_bits)
        return bi


def ECFP4(smiles_list: List[str]) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
    """
    Converts array of SMILES to ECFP bitvectors.
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
        mol: RDKit molecules
        radius: ECFP fingerprint radius
        length: number of bits

    Returns: a list of fingerprints
    """
    mols = construct_check_mol_list(smiles_list)
    return [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
