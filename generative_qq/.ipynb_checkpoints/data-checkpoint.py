"""
Data generation methods + data instances
"""
from .vqe import *
import jax.numpy as jnp
import scipy as sc
import pennylane as qml
from tqdm.notebook import tqdm
import autohf as hf

from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import get_ground_state, get_sparse_operator
import numpy
import scipy
import scipy.linalg
from openfermionpyscf import run_pyscf
import numpy as np
from matplotlib import pyplot as plt

from openfermion.chem import MolecularData
from tqdm.notebook import tqdm


def hamming_weight(n):
    """Computes the Hamming weight of the binary representation of a decimal number

    Args
        n (int): An integer
    Returns
        int
    """
    return bin(n).count("1")


def spin_z(n):
    state = list(bin(n)[2:])
    
    odd, even = 0, 0
    for c, s in enumerate(state):
        if s == "1" and c % 2 == 0:
            even += 1
        if s == "1" and c % 2 == 0:
            odd += 1
    return np.abs(even - odd)


def vec_hamming_weight(vec, weight, tol=1e-4):
    """Checks if, when expanded in the computational basis, the non-zero components of a vector
    correspond to basis state with a fixed Hamming weight.

    Args
        vec (Iterable): The vector being checked
        weight (int): The fixed Hamming weight
        tol (float): The tolerance for a vector entry being equal to 0
    Returns
        bool
    """
    out = True
    for count, num in enumerate(vec):
        if abs(num) > tol and hamming_weight(count) != weight:
            out = False
    return out


def vec_spin(vec, spin, tol=1e-4):
    out = True
    for count, num in enumerate(vec):
        if abs(num) > tol and spin_z(count) != spin:
            out = False
    return out
    

def h_to_matrix(h):
    """Converts a PennyLane Hamiltonian object into a matrix

    Args
        h (qml.Hamiltonian): The input Hamiltonian
    Returns
        numpy.array
    """
    matrix = np.zeros((2 ** len(h.wires), 2 ** len(h.wires)), dtype='complex128')

    for coeff, op in tqdm(zip(h.coeffs, h.ops)):
        tensor = []
        wires = [o.wires.tolist()[0] for o in qml.operation.Tensor(op).non_identity_obs]
        for w in h.wires:
            if w in wires:
                tensor.append(qml.operation.Tensor(op).non_identity_obs[wires.index(w)])
            else:
                tensor.append(qml.Identity(w))
        matrix += coeff * qml.operation.Tensor(*tensor).matrix

    return matrix


def allowed_vec_val(matrix, weight):
    """Returns the eigenvectors v and corresponding eigenvalues w such that
    vec_hamming_weight(v, weight) = True, where weight is some given Hamming weight.

    Args
        matrix (numpy.array): Input matrix
        weight (int): Fixed Hamming weight
    Returns
        (numpy.array, numpy.array)
    """

    w, v = np.linalg.eigh(matrix)
    v = np.transpose(v)

    allowed_vals = []
    allowed_vecs = []

    for c, vec in enumerate(v):
        if vec_hamming_weight(vec, weight):
            allowed_vals.append(w[c])
            allowed_vecs.append(vec)

    return jnp.array(allowed_vecs).T, jnp.array(allowed_vals)


def allowed_vec_val_sparse(matrix, weight, spin, k=6):
    """Returns the eigenvectors v and corresponding eigenvalues w such that
    vec_hamming_weight(v, weight) = True, where weight is some given Hamming weight.

    Args
        matrix (numpy.array): Input matrix
        weight (int): Fixed Hamming weight
    Returns
        (numpy.array, numpy.array)
    """

    w, v = sc.sparse.linalg.eigsh(matrix, k=k, which="SA")
    v = np.transpose(v)

    allowed_vals = []
    allowed_vecs = []

    for c, vec in enumerate(v):
        if vec_hamming_weight(vec, weight) and vec_spin(vec, spin):
            allowed_vals.append(w[c])
            allowed_vecs.append(vec)

    return jnp.array(allowed_vecs).T, jnp.array(allowed_vals)


########### Data generation methods ###########

def pyscf_hamiltonian(mol):
    h = lambda coordinates: qchem.molecular_hamiltonian(
        mol.symbols,
        coordinates,
        charge=mol.charge,
        mult=mol.multiplicity,
        basis='sto-3g',
        active_electrons=mol.active_electrons,
        active_orbitals=mol.active_orbitals,
        mapping='jordan_wigner',
    )[0]
    return lambda r : qml.utils.sparse_hamiltonian(h(r))


def generate_random_geometry(mol, n, sampler, pl, perturbation, symmetry=None):
    """Generates a random geometry, close to Hartree-Fock geometry"""
    if symmetry is None:
        x = sampler(size=(n, 3 * len(mol.symbols)), low=pl, high=perturbation)
        return x
    x = sampler(size=(n, symmetry), low=pl, high=perturbation)
    return x


def exact_diag_data(H, samples, weight):
    training = []
    for r in tqdm(samples):
        mat = H(r).matrix()
        v, w = allowed_vec_val(mat, weight)
        min_index = np.where(w == min(w))
        training.append([jnp.array(r), jnp.array(v.T[min_index][0])])
    return training


def exact_diag_data_sparse(H, samples, weight, spin, k=None):
    """Generates data wirh exact diagonalization of the Hamiltonian (assumes the generated Hamiltonian is sparse)"""
    training_g, training_s = [], []
    for r in tqdm(samples):
        h = H(r).sparse_matrix()
        k = 6 if k is None else k
        v, w = allowed_vec_val_sparse(h, weight, spin, k=k)
        min_index = np.where(w == min(w))
        training_g.append(jnp.array(r))
        training_s.append(jnp.array(v.T[min_index][0]))
    return jnp.array(training_g), jnp.array(training_s)

def vqe_data(H, samples, circuit, dev, optimizer, steps, initial_params, transform=None, sparse=False, bar=True, diff_method="adjoint"):
    """Generates data with VQE. The training data is bundled as the parameters passed into the Hamiltonian and the angles passed 
    into the VQE circuit that yield the desired ground state"""
    diff_method = "parameter-shift" if sparse else diff_method
    training = []
    for r in samples:
        energy, params = vqe(circuit, H(r), dev, optimizer, steps, initial_params, sparse=sparse, bar=bar, diff_method=diff_method)
        training.append([jnp.array(r), jnp.array(params)])
    return training


########### Molecule data ##############

def generate_hf_state(molecule):
    """Returns the HF state for a molecule"""
    return qchem.hf_state(molecule.active_electrons, 2 * molecule.active_orbitals)

def hf_state(mol, dev):
    """Returns the Hartree-Fock state"""
    @qml.qnode(dev)
    def hf_circ():
        qml.BasisState(generate_hf_state(mol), wires=dev.wires)
        return qml.state()
    return hf_circ()

def sparse_hamiltonian(molecule, wires, center=0.0, symmetry=None):
    """Generates a sparse molecular Hamiltonian, with respect to a certain symmetry constraint"""
    if symmetry is None:
        sym_fun = lambda r : r
    else:
        sym_fun = molecule.coordinate_tr[symmetry]
    return lambda r : hf.sparse_H(molecule, wires)(center + sym_fun(r))

########### A collection of molecules ############


class H2:
    symbols = ["H", "H"]
    basis_name = "sto-3g"
    active_electrons = 2
    n_electrons = 2
    n_orbitals = 2
    active_orbitals = 2
    charge = 0
    multiplicity = 1
    
    hf_geometry = angs_bohr * np.array([0.00000, 0.00000, -0.3561433440, 0.00000, 0.00000, 0.3561433440])
    fci_geometry = angs_bohr * np.array([0.00000, 0.00000, -0.3674329136, 0.00000, 0.00000, 0.3674329136])

    coordinate_tr = {
        1 : (lambda r : np.array([0.0, 0.0, -r[0], 0.0, 0.0, r[0]]))
    }

class H3_Plus:
    symbols = ["H", "H", "H"]
    basis_name = "sto-3g"
    
    n_electrons = 2
    n_orbitals = 3
    active_electrons = 2
    active_orbitals = 3
    
    charge = 1
    multiplicity = 1
    
    """
    hf_geometry = np.array([ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -9.13622790e-01, -1.58251261e+00,
        0.00000000e+00, -1.82729998e+00, -5.31182316e-05,  0.00000000e+00])
    """
    
    hf_geometry = np.array([0.00000000e+00,  0.00000000e+00,  0.00000000e+00, (np.sqrt(3)/2) * 1.82729998e+00, (1/2) * 1.82729998e+00, 0.0, 
                            (np.sqrt(3)/2) * 1.82729998e+00, (-1/2) * 1.82729998e+00, 0.0])
    
    fci_geometry = np.array([ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -9.31547231e-01, -1.61291929e+00,
        0.00000000e+00, -1.86313084e+00, -4.41650688e-05,  0.00000000e+00])

    coordinate_tr = {
        4 : (lambda r : np.array([0.0, 0.0, 0.0, r[0], r[1], 0.0, r[2], r[3], 0.0])),
        2 : (lambda r : np.array([0.0, 0.0, 0.0, r[0], r[1], 0.0, r[0], -r[1], 0.0]))
    }

# Molecule we wish to simulate
class H4:
    symbols = ["H", "H", "H", "H"]
    basis_name = "sto-3g"
    active_electrons = 4
    active_orbitals = 4
    n_electrons = 4
    n_orbitals = 4
    charge = 0
    multiplicity = 1
    
    # Hartee-Fock geometry of the molecule
    hf_geometry = angs_bohr * np.array([
        0.0000000000,   0.0000000000,   0.0000000000,
        0.0000000000,   3.6426639964,   0.0000000000,
        0.7122551091,   0.0030138567,   0.0000000000,
        0.7126635035,   3.6421209279,   0.0000000000
    ])

    coordinate_tr = {
        1 : (lambda r : np.array([0.0, 0.0, 0.0, 0.0, r[0], 0.0, r[0], 0.0, 0.0, r[0], r[0], 0.0])),
        7 : (lambda r : np.array([0.0, 0.0, 0.0, r[0], r[1], 0.0, r[2], r[3], 0.0, r[4], r[5], r[6]]))
    }

class BeH2:
    symbols = ["Be", "H", "H"]
    basis_name = "sto-3g"
    active_electrons = 4
    active_orbitals = 6
    n_electrons = 6
    n_orbitals = 7
    charge = 0
    multiplicity = 1

    hf_geometry = angs_bohr * np.array([
        0.0000000000, 0.0000000000, 0.0000000000, 
        0.0000000000, 1.2905503952, -0.0000000000, 
        0.0000000000, -1.2905503952, -0.0000000000
        ])

    fci_geometry = angs_bohr * np.array([0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -1.3164290143, 0.0000000000, 0.0000000000, 1.3164290143, 0.0000000000])
    
    coordinate_tr = {
        1 : (lambda r : np.array([0.0, 0.0, 0.0, 0.0, r[0], 0.0, 0.0, -r[0], 0.0])),
        2 : (lambda r : np.array([0.0, 0.0, 0.0, 0.0, r[0], 0.0, 0.0, r[1], 0.0]))
    }

class H2O:
    symbols = ['O', 'H', 'H']
    basis_name = "sto-3g"
    active_electrons = 8
    active_orbitals = 6
    n_electrons = 10
    n_orbitals = 11
    charge = 0
    multiplicity = 1

    hf_geometry = angs_bohr * np.array([
        0.000000000, 0.000000000, 0.0000000000, 
        0.7580811324 + 0.0000089206, 0.6883823042 - 0.0526064822, 0.0000000000, 
        -0.7581449681 + 0.0000089206, 0.6883210739 - 0.0526064822,   0.0000000000
        ])
    
    coordinate_tr = {
        2 : (lambda r : np.array([0.0, 0.0, 0.0, r[0], r[1], 0.0, -r[0], r[1], 0.0]))
    }
    
    
############## Hamiltonian buidling ###############

def build_of_ham(mol, R, center=None):
    """
    Builds a sparse Hamiltonian using OpenFermion
    """
    # Makes the geometry
    center = mol.hf_geometry if center is None else center
    geo_1 = 0.529177 * (center + R).reshape((3,len(mol.symbols)))
    geometry = [(symbol, l) for symbol, l in zip(mol.symbols, geo_1)]
    
    # Make molecule and print out a few interesting facts about it.
    molecule = MolecularData(geometry, mol.basis_name, mol.multiplicity,
                             mol.charge, description=str(geometry))

    molecule = run_pyscf(molecule, run_scf=True)
    core, active = qml.qchem.active_space(mol.electrons, mol.orbitals, mult=mol.multiplicity, active_electrons=mol.active_electrons, active_orbitals=mol.active_orbitals)
    molecular_hamiltonian = molecule.get_molecular_hamiltonian(occupied_indices=core, active_indices=active)
    
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    qubit_hamiltonian.compress()

    # Get sparse operator and ground state energy.
    sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian)
    return sparse_hamiltonian

def vec_conversion(c_vec, weight):
    """
    Converts PySCF vector to qubit state
    """
    pass

def data_with_pyscf(mol, R, center=None):
    """
    Computes ground state and ground state energy with PySCF FCI
    """
    center = mol.hf_geometry if center is None else center
    geo_1 = (center + R).reshape((3,len(mol.symbols)))
    
    mol = pyscf.M(
        atom = 'H 0 0 0; H 0 0 {}; H 0 0 {}'.format(R, -R),
        basis = 'sto-3g',
        charge=1,
        unit = "B")

    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    e, fci_vec = cisolver.kernel()
    return e, vec_conversion(fci_vec.flatten())
    