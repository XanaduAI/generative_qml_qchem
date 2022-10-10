"""Executes the entire generative model for a set of parameters"""

import autohf as hf
import pennylane.numpy as np
import pennylane as qml
import jax
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import jax.numpy as jnp
import tailgating as tg
import scipy as sc
import optax
from pennylane import qchem
import sys
from .data import *
from .nn import *
from .utils import *
from .vqe import *

np.set_printoptions(linewidth=100000)

"""
class ModelObject:
    
    # Molecule class and quantum device
    molecule = qq.H4()
    dev = qml.device('default.qubit', wires=(2 * molecule.n_orbitals))
    
    # Geometric parametrization
    in_length = 1
    fun_r = lambda r : mol.hf_geometry + np.array([0.0, 0.0, 0.0, 0.0, r[0], 0.0, r[0], 0.0, 0.0, r[0], r[0], 0.0])
    parametrization = (fun_r, in_length)
    
    # Data samples
    N_samples = 5
    samples = [[x] for x in np.linspace(-0.2, 0.5, N_samples)]
    
    nn_layers = [40, 40]
    optimizer = optax.adam(0.01)
    n_steps = 2000
    geo_range = np.linspace(-0.4, 4.0, 50)
"""

def run_model(model_obj):
    
    # Basic stuff
    mol = model_obj.mol
    dev = model_obj.device
   
    # Builds the Hamiltonian with a particular parametrization
    n_params = model_obj.n_params
    samples = model_obj.samples
    
    core, active = qml.qchem.active_space(
            mol.n_electrons, mol.n_orbitals, mol.multiplicity, mol.active_electrons, mol.active_orbitals
        )
    
    guess = hf.w_coeffs(mol)(mol.hf_geometry)
    pl_mol = lambda R : qml.qchem.Molecule(
            mol.symbols,
            np.array(R, requires_grad=False).reshape(len(mol.symbols), 3),
            charge=mol.charge,
            mult=mol.multiplicity,
            basis_name="sto-3g"
        )
    
        
    # Constructs the sparse Hamiltonian
    print("-----------------------------------------------------")
    print("CONSTRUCTING HARTREE-FOCK HAMILTONIAN")
    print("-----------------------------------------------------")
    H_sparse = lambda r : qml.SparseHamiltonian(qml.qchem.sparse_hamiltonian(pl_mol(model_obj.parametrization(r)), core=core, active=active, guess=guess), wires=dev.wires)
    H_hf = H_sparse([0.0 for _ in range(n_params)])
    
    # Gets data
    print("GENERATING TRAINING DATA")
    print("-----------------------------------------------------")
    g_data, s_data = exact_diag_data_sparse(H_sparse, samples, mol.active_electrons, 0)
    print("-----------------------------------------------------")
    
    # Constructs circuit ansatz
    print("CONSTRUCTING CIRCUIT ANSATZ")
    print("-----------------------------------------------------")
    # Initializes an optimizer for the adaptive procedure
    optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

    # Runs the adaptive procedure, with a pool and single and double excitations
    pool = gate_pool(mol)
    gates = []
    
    for geo in model_obj.geo_samples:
        H_hf = H_sparse(geo)
        gates_t, params = tg.batch_adapt_vqe(H_hf, dev, pool, generate_hf_state(mol), optimizer, model_obj.circ_steps, bar=True, sparse=True, tol=1e-5)
        gates.extend(gates_t)
    gates = list(set(gates))
    
    print("Number of gates = {}".format(len(gates)))
    print("-----------------------------------------------------")
    
    circ = circuit(gates, dev.wires, generate_hf_state(mol)) # Circuit ansatz
    state_circ = state_circuit(circ, dev) # State-generating circuit
    
    key = jax.random.PRNGKey(100)
    layer_sizes = [n_params] + model_obj.nn_layers + [len(gates)] # Specifies the sizes of the layers of the feed-forward NN
    initial_NN_params = network_params(layer_sizes, key, zero=True) # Specifies the initial NN params (all set to 0.0)
    
    # Quantum model
    def model(geometry, NN_theta):
        angles = neural_network(NN_theta, geometry)
        return state_circ(angles)
    
    optimizer = model_obj.optimizer
    params = {'w': initial_NN_params}
    opt_state = optimizer.init(params)

    # Loss function
    loss = lambda NN : exact_fidelity_loss(model, g_data, s_data)(NN['w'])

    # Gradient of loss function
    gradient_fn = jax.jit(jax.value_and_grad(loss))
    
    steps = model_obj.n_steps
    print("TRAINING MODEL")
    print("-----------------------------------------------------")
    
    bar = tqdm(range(steps))

    # Performs optimization of the model
    for s in bar:
        v, gr = gradient_fn(params)
        bar.set_description(str(v))

        # Computes the gradient, updates the parameters
        updates, opt_state = optimizer.update(gr, opt_state)
        params = optax.apply_updates(params, updates)
    print("Final loss = {}".format(loss(params)))

    # Optimized model
    optimized_model = lambda g : model(g, params['w'])
    print("-----------------------------------------------------")
    
    def extract_data(R):
        # Extracts data from the model

        model_energy_list, real_energy_grnd, real_energy_excited = [], [], []
        ground_overlap, hf_overlap = [], []

        for c, x in tqdm(list(enumerate(R))):

            # Returns the true and model energies, wavefunctions
            fn = model_energy(optimized_model, lambda r : H_sparse(r).sparse_matrix(), mol.active_electrons)
            model_e, real_e, model_v, real_v = fn(jnp.array([x]))

            # Computes the true ground state energy and wavefunction
            real_e_sorted = sorted(real_e)
            e_grnd = real_e_sorted[0]
            var = True if model_obj.cut_up is None else c < model_obj.cut_up
            if c >= model_obj.cut_down and var:
                e_excited = real_e_sorted[1]
                real_energy_excited.append(e_excited)
            v_grnd = real_v.T[list(real_e).index(e_grnd)]

            # Records all relevant energies
            model_energy_list.append(model_e)
            real_energy_grnd.append(e_grnd)
            #real_energy_excited.append(e_excited)

            # Records all relevant state overlaps
            ground_overlap.append(np.abs(np.inner(np.conj(v_grnd), model_v)) ** 2)
            hf_overlap.append(np.abs(np.inner(np.conj(hf_v), model_v)) ** 2)

        return model_energy_list, real_energy_grnd, real_energy_excited, ground_overlap, hf_overlap

    geo_range = model_obj.geo_range
    
    hf_v = hf_state(mol, dev) # Computes the Hartree-Fock state
    
    print("EXTRACTING DATA FOR PLOTS")
    print("-----------------------------------------------------")

    model_e, real_e, real_e_ex, grnd_overlap, hf_overlap = extract_data(geo_range) # Gets data points over range of geometries
    model_pt, real_pt, real_pt_excited, ground_pt, hf_pt = extract_data(g_data.flatten()) # Gets data points for training geometries
    
    print("-----------------------------------------------------")
    
    # Shows the plots!
    
    fig, ax = plt.subplots(figsize=(34, 14))

    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })

    f = 35

    ax1 = plt.subplot(1, 2, 1)
    plt.xlabel("$\delta_H$ (Bohr)", fontsize=f)
    plt.ylabel("Energy (Hartrees)", fontsize=f)
    l1 = ax1.plot(geo_range, model_e, label="Model", linewidth=3, color="#ff8080ff")
    l2 = ax1.plot(geo_range, real_e, label="Ground State", linewidth=3, color="#00ccb3ff")
    if model_obj.excited:
        l3 = ax1.plot(geo_range[model_obj.cut_down:model_obj.cut_up], real_e_ex, '--', label="First Excited State", linewidth=3, color="black")
    l1_scatter = ax1.scatter(g_data.flatten(), real_pt, zorder=10, c="#00ccb3ff", s=150)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, prop={'size': f})

    plt.xticks(fontsize=f)
    plt.yticks(fontsize=f)

    ax2 = plt.subplot(1, 2, 2)
    l4 = ax2.plot(geo_range, grnd_overlap, linewidth=3, color="#00ccb3ff")
    l5 = ax2.plot(geo_range, hf_overlap, linewidth=3, color="#ff8080ff")
    l4_scatter = ax2.scatter(g_data.flatten(), ground_pt, s=150, color="#00ccb3ff")
    l5_scatter = ax2.scatter(g_data.flatten(), hf_pt, s=150, color="#ff8080ff")
    plt.xlabel("$\delta_H$ (Bohr)", fontsize=f)
    plt.ylabel("Fidelity", fontsize=f)

    plt.xticks(fontsize=f)
    plt.yticks(fontsize=f)

    plt.savefig(model_obj.filename)
    plt.show()