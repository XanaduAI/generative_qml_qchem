from jax._src.numpy.lax_numpy import diff
import pennylane as qml
import pennylane.numpy as np
from pennylane import qchem
from tqdm.notebook import tqdm
from functools import partial
import scipy

angs_bohr = 1.88973


################# Ground State Electronic Structure Calculations #################

def vqe(circuit, H, dev, optimizer, steps, params, sparse=False, bar=True, diff_method="adjoint"):
    """
    Performs the VQE (Variational Quantum Eigensolver) process for a given circuit and Hamiltonian.
    Optimizes a function of the form C(theta) = < psi(theta) | H | psi(theta) >.
    Args
        circuit (function): A quantum function, implementing a series of parametrized gates
        H (qml.Hamiltonian, qml.SparseHamiltonian): A Hamiltonian to be optimized
        dev (qml.Device): The device on which to perform VQE
        optimizer (qml.GradientDescentOptimizer): The optimizer used during VQE
        steps (int): The number of steps taken by the VQE procedure
        params (Iterable): Initial parameters for VQE optimization
    Kwargs
        sparse (bool): Indicated whether to simulate using sparse methods
        bar (bool): Indicates whether to display a progress bar during optimization
        diff_method (str): The differentiation method to use for VQE (Note: Only works for non-sparse VQE)
    Returns
        (Optimized energy, optimized parameters): (float, Iterable)
    """
    diff_method = "parameter-shift" if sparse else diff_method

    @qml.qnode(dev, diff_method=diff_method)
    def cost_fn(params):
        circuit(params)
        return qml.expval(H)

    nums = tqdm(range(steps)) if bar else range(steps)

    for s in nums:
        #params, energy, grad = optimizer.step_and_cost_and_grad(cost_fn, params)
        p = optimizer.compute_grad(cost_fn, (params,), {})
        grad, energy = p
        grad = grad[0]
        params = params - (optimizer.stepsize) * grad
        
        if np.allclose(grad, 0.0, atol=1e-5):
            break
        if bar:
            nums.set_description("Energy = {}".format(energy))

    return energy, params


def adapt_vqe(H, dev, operator_pool, hf_state, optimizer, max_steps, vqe_steps, bar=False):
    """Performs the original ADAPT-VQE procedure using the sparse VQE method.
    See [arXiv:1812.11173v2] for more details.
    Args
        H (qml.Hamiltonian): A Hamiltonian used to perform VQE
        dev (qml.Device): A device on which to perform the simulations
        operator_pool (Iterable[function]): A collection of parametrized quantum gates which will make up the operator pool
        Each element is of type (float or array) -> (qml.Operation)
        hf_state (array): The Hartree-Fock state
        optimizer (qml.GradientDescentOptimizer): The optimizer used for VQE
        steps (float): The number of times the adaptive loop should be executed
        vqe_steps (float): The number of steps that VQE should take, for each adaptive loop
    Kwargs
        bar (bool): Specifies whether to show a progress bar
    Returns
        (Iterable[function]): The sequence of quantum operations yielded from ADAPT-VQE
        (Iterable[float]): The optimized parameters of the circuit consisting of the outputted quantum operations
    """
    optimal_params = []
    seq = []
    termination = False
    counter = 0

    while not termination and counter < max_steps:
        grads = []
        for op in operator_pool:

            # Constructs the new circuit
            @qml.qnode(dev, diff_method='parameter-shift')
            def cost_fn(param):
                qml.BasisState(hf_state, wires=dev.wires)
                for operation, p in zip(seq, optimal_params):
                    operation(p)
                op(param)
                return qml.expval(H)

            # Computes the gradient of the circuit
            grad_fn = qml.grad(cost_fn)(0.0)
            grads.append(grad_fn)

        abs_ops = [abs(x) for x in grads]
        if np.allclose(abs_ops, 0.0):
            termination = True
            break
        chosen_op = operator_pool[abs_ops.index(max(abs_ops))]

        def vqe_circuit(params):
            qml.BasisState(hf_state, wires=dev.wires)
            for operation, p in zip(seq, params[:len(params) - 1]):
                operation(p)
            chosen_op(params[len(params) - 1])

        energy, optimal_params = vqe(vqe_circuit, H, dev, optimizer, vqe_steps, np.array(list(optimal_params) + [0.0]), sparse=True, bar=bar)
        seq.append(chosen_op)
        counter += 1
    return seq, optimal_params


def batch_adapt_vqe(H, dev, batch_pool, init_state, optimizer, vqe_steps, bar=False, tol=1e-4, sparse=False, diff_method="best"):
    """
    ADAPT-VQE, but computing derivatives of collections of gates instead of single gates
    """
    diff_method = "parameter-shift" if sparse else diff_method
    optimal_params = []
    seq = []
    for c, batch in enumerate(batch_pool):

        #Constructs the circuit
        @qml.qnode(dev, diff_method=diff_method)
        def cost_fn(param):
            qml.BasisState(init_state, wires=dev.wires)
            for operation, p in zip(seq, optimal_params):
                operation(p)
            for p, op in zip(param, batch):
                op(p)
            return qml.expval(H) 
        
        # Computes the gradient of the circuit
        grad_fn = qml.grad(cost_fn)(np.array([0.0 for _ in range(len(batch))]))

        counter = 0
        for b in range(len(batch)):
            if abs(grad_fn[b]) >= tol:
                seq.append(batch[b])
                counter += 1

        # Performs VQE
        if c < len(batch_pool) - 1:
            def vqe_circuit(params):
                qml.BasisState(init_state, wires=dev.wires)
                for operation, p in zip(seq, params):
                    operation(p)

            energy, optimal_params = vqe(vqe_circuit, H, dev, optimizer, vqe_steps, np.array(list(optimal_params) + [0.0 for _ in range(counter)]), sparse=True, bar=bar, diff_method=diff_method)
        else:
            optimal_params = np.array(list(optimal_params) + [0.0 for _ in range(counter)])
    return seq, optimal_params


def gate_pool(active_electrons, active_orbitals):
    """
    Generates a gate pool and single and double excitations
    """
    singles, doubles = qml.qchem.excitations(electrons=active_electrons, orbitals=2 * active_orbitals)
    pool = []

    for s in singles:
        pool.append(lambda p, w=s: qml.SingleExcitation(p, wires=w))
    for d in doubles:
        pool.append(lambda p, w=d: qml.DoubleExcitation(p, wires=w))
    return pool


def batch_gate_pool(mol):
    """Generates a gate pool and single and double excitations"""
    singles, doubles = qml.qchem.excitations(electrons=mol.active_electrons, orbitals=2 * mol.active_orbitals)
    pool1, pool2 = [], []

    for s in singles:
        pool1.append(lambda p, w=s: qml.SingleExcitation(p, wires=w))
    for d in doubles:
        pool2.append(lambda p, w=d: qml.DoubleExcitation(p, wires=w))
    return pool2, pool1


def compute_state(circuit, dev, optimal_params):
    """Returns the statevector yielded from a parametrized circuit
    Args
        circuit (func): A quantum function representing a circuit
        dev (qml.device): The device on which to execute the circuit
        optimal_params (Iterable): The parameters to be fed into the circuit
    Returns
        numpy.array
    """

    @qml.qnode(dev)
    def ansatz(params):
        circuit(params)
        return qml.state()

    return ansatz(optimal_params)


################# Chemical ansatz #################


def gate_pool(mol, sz=0):
    """Generates a gate pool and single and double excitations"""
    singles, doubles = qml.qchem.excitations(electrons=mol.active_electrons, orbitals=2 * mol.active_orbitals, delta_sz=sz)
    pool1, pool2 = [], []

    for s in singles:
        pool1.append(lambda p, w=s: qml.SingleExcitation(p, wires=w))
    for d in doubles:
        pool2.append(lambda p, w=d: qml.DoubleExcitation(p, wires=w))
    return pool2, pool1

################ Circuits ###################

def circuit(gates, wires, initial):
    """Circuit for generating data"""
    def data_generating_circuit(params):
        qml.BasisState(initial, wires=wires)
        for p, g in zip(params, gates):
            g(p)
    return data_generating_circuit

def state_circuit(circ, dev):
    """State generating circuit"""
    @qml.qnode(dev, interface="jax")
    def state_circ(params):
        circ(params)
        return qml.state()
    return state_circ