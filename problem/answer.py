import sys
from typing import Any, Callable, Optional, Literal

import numpy as np
from dataclasses import dataclass
import traceback
import time

from openfermion.transforms import jordan_wigner

from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit)
from quri_parts.core.estimator import QuantumEstimator, ParametricQuantumEstimator
from quri_parts.core.estimator.gradient import parameter_shift_gradient_estimates
from quri_parts.core.operator import Operator, truncate, SinglePauli
from quri_parts.core.measurement import (
    bitwise_commuting_pauli_measurement,
    CachedMeasurementFactory,
)
from quri_parts.core.sampling.shots_allocator import (
    create_proportional_shots_allocator,
)
from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState
from quri_parts.core.utils.array import readonly_array
from quri_parts.algo.optimizer import (
    CostFunction,
    GradientFunction,
    Optimizer,
    OptimizerState,
    OptimizerStatus,
    Params,
    AdaBelief,
)
from quri_parts.algo.optimizer.tolerance import ftol as create_ftol
from quri_parts.openfermion.operator import operator_from_openfermion_op

sys.path.append("../")
from utils.challenge_2024 import ChallengeSampling, problem_hamiltonian, ExceededError

challenge_sampling = ChallengeSampling()
"""
####################################
add codes here
####################################
"""

########################################################################
""" VQE function module
"""


def vqe(
    hamiltonian: Operator,
    parametric_state: ParametricCircuitQuantumState,
    estimator: ParametricQuantumEstimator[ParametricCircuitQuantumState],
    init_params,
    optimizer: Optimizer,
    max_iter=200,
    gradient_estimator=None,
    challenge_sampling=None,
):
    """ Perform VQE 
    
    Args:
        hamiltonian (Operator): Hamiltonian Operator
        parametric_state (ParametricCircuitQuantumState): Parametric Circuit State of the ansatz
        estimator: Estimator
        init_params: Initial parameters
        optimizer (Optimizer): Optimizer
        max_iter (int): Maximum iteration
        challenge_sampling: ChallengeSampling

    Returns:
        opt_state: Optimizer state
        energy_history: Energy history
        shots_history: Shots history
    """

    opt_state = optimizer.get_init_state(init_params)
    energy_history = []
    shots_history = []

    def c_fn(param_values):
        estimate = estimator(hamiltonian, parametric_state, param_values)
        return estimate.value.real

    def g_fn(param_values):
        grad = parameter_shift_gradient_estimates(hamiltonian,
                                                  parametric_state,
                                                  param_values,
                                                  gradient_estimator)
        return np.asarray([i.real for i in grad.values])

    while True:
        try:
            opt_state = optimizer.step(opt_state, c_fn, g_fn)
            energy_history.append(opt_state.cost)

            if challenge_sampling is not None:
                shots_history.append(challenge_sampling.total_shots)

            print(f"Energy: {opt_state.cost}")
            print(f"Parameters: {opt_state.params}")
        except ExceededError as e:
            print(str(e))
            print(opt_state.cost)
            return opt_state, energy_history, shots_history

        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
        if opt_state.niter >= max_iter:
            print("Reached max iteration")
            break

    return opt_state, energy_history, shots_history


########################################################################
""" Hamiltonian Variational Ansatz
"""


class HamiltonianVariational(
        ImmutableLinearMappedUnboundParametricQuantumCircuit):
    """Hamiltonian variational ansatz.
   
    Args:
        qubit_count: Number of qubits.
        hamiltonian: Hamiltonian operator.
    """

    def __init__(self, n_sites: int):
        circuit = LinearMappedUnboundParametricQuantumCircuit(n_sites * 2)

        self._add_g_gates(n_sites, circuit)

        self._add_o_gates(n_sites, circuit)

        self._add_h_gates(n_sites, circuit)

        super().__init__(circuit)

    def _add_g_gates(self, n_sites, circuit):
        idx = 0
        while 2 * idx + 2 < n_sites:
            self._add_one_g_gate(circuit, 2 * idx + 1, 2 * idx + 2)
            self._add_one_g_gate(circuit, 2 * idx + 1 + n_sites,
                                 2 * idx + 2 + n_sites)
            idx += 1

        idx = 0
        while 2 * idx + 1 < n_sites:
            self._add_one_g_gate(circuit, 2 * idx, 2 * idx + 1)
            self._add_one_g_gate(circuit, 2 * idx + n_sites,
                                 2 * idx + 1 + n_sites)
            idx += 1

    def _add_o_gates(self, n_sites, circuit):
        for idx in range(n_sites):
            self._add_one_o_gate(circuit, idx, idx + n_sites)

    def _add_h_gates(self, n_sites, circuit):
        idx = 0
        while 2 * idx + 2 < n_sites:
            self._add_one_h_gate(circuit, 2 * idx + 1, 2 * idx + 2)
            self._add_one_h_gate(circuit, 2 * idx + 1 + n_sites,
                                 2 * idx + 2 + n_sites)
            idx += 1

        idx = 0
        while 2 * idx + 1 < n_sites:
            self._add_one_h_gate(circuit, 2 * idx, 2 * idx + 1)
            self._add_one_h_gate(circuit, 2 * idx + n_sites,
                                 2 * idx + 1 + n_sites)
            idx += 1

    def _add_one_g_gate(self, circuit, target_index1, target_index2):
        theta = circuit.add_parameter("theta")

        circuit.add_U2_gate(target_index2, 0, 0)

        circuit.add_CNOT_gate(target_index2, target_index1)

        circuit.add_ParametricRY_gate(target_index1, {theta: -1 / 2})
        circuit.add_ParametricRY_gate(target_index2, {theta: -1 / 2})

        circuit.add_CNOT_gate(target_index2, target_index1)

        circuit.add_U2_gate(target_index2, -np.pi, -np.pi)

    def _add_one_h_gate(self, circuit, target_index1, target_index2):
        phi = circuit.add_parameter("phi")

        circuit.add_U1_gate(target_index1, np.pi / 2)

        circuit.add_U2_gate(target_index2, 0, 0)

        circuit.add_CNOT_gate(target_index2, target_index1)

        circuit.add_ParametricRY_gate(target_index1, {phi: -1 / 2})
        circuit.add_ParametricRY_gate(target_index2, {phi: -1 / 2})

        circuit.add_CNOT_gate(target_index2, target_index1)

        circuit.add_U1_gate(target_index1, -np.pi / 2)

        circuit.add_U2_gate(target_index2, -np.pi, -np.pi)

    def _add_one_o_gate(self, circuit, controll_index, target_index):
        psi = circuit.add_parameter("psi")

        circuit.add_ParametricRZ_gate(controll_index, {psi: 1 / 2})

        circuit.add_CNOT_gate(controll_index, target_index)

        circuit.add_ParametricRZ_gate(target_index, {psi: -1 / 2})

        circuit.add_CNOT_gate(controll_index, target_index)

        circuit.add_ParametricRZ_gate(target_index, {psi: 1 / 2})


########################################################################
""" Rotosolve optimizer module
"""


@dataclass(frozen=True)
class OptimizerStateRotosolve(OptimizerState):
    """Optimizer state for Rotosolve."""
    pass


class Rotosolve(Optimizer):
    """Rotosolve optimization algorithm.
    
    Args:
        num_repeatation: Number of repeatation of parameter updating in one step. 
            Each parameter is updated once in each repeatation.
        phai: Base phase value for the analytic minimization.  
        ftol: If not None, judge convergence by cost function tolerance.
            See :func:`~.tolerance.ftol` for details.
    """

    def __init__(self,
                 num_repeatation=20,
                 phai=0.0,
                 ftol: Optional[float] = 1e-5,
                 reversed_update=False):
        self.num_repeatation = num_repeatation
        self.phai = phai
        self._ftol: Optional[Callable[[float, float], bool]] = None
        if ftol is not None:
            if not 0.0 < ftol:
                raise ValueError("ftol must be a positive float.")
            self._ftol = create_ftol(ftol)
        self.reversed_update = reversed_update

    def get_init_state(self, init_params: Params) -> OptimizerStateRotosolve:
        params = readonly_array(np.array(init_params, dtype=float))
        return OptimizerStateRotosolve(params=params)

    def step(
        self,
        state: OptimizerStateRotosolve,
        cost_function: CostFunction,
        grad_function: Optional[GradientFunction] = None,
    ) -> OptimizerStateRotosolve:
        funcalls = state.funcalls
        niter = state.niter + 1

        if niter == 1:
            cost_prev = cost_function(state.params)
            funcalls += 1
        else:
            cost_prev = state.cost

        params = state.params.copy()

        indecies = range(len(params))
        if self.reversed_update:
            indecies = reversed(indecies)

        for _ in range(self.num_repeatation):
            for par_idx in indecies:

                def univariate(x):
                    params[par_idx] = x
                    return cost_function(params)

                param_min = self.min_analytic(univariate, base_value=self.phai)
                funcalls += 3
                params[par_idx] = param_min

        cost = cost_function(params)
        funcalls += 1

        if self._ftol and self._ftol(cost, cost_prev):
            status = OptimizerStatus.CONVERGED
        else:
            status = OptimizerStatus.SUCCESS
        return OptimizerStateRotosolve(
            params=params,
            cost=cost,
            status=status,
            niter=niter,
            funcalls=funcalls,
            gradcalls=state.gradcalls,
        )

    def min_analytic(self,
                     univariate_cost_function,
                     base_value=0.0,
                     base_cost=None):
        """Analytically minimize a function that depends on a single parameter 
        and has a single frequency. Uses two or three function evaluations.

        Args:
            univariate_cost_function: The function to minimize.
            base_value: The base value of the parameter at which the cost function is evaluated.
            base_cost: The cost of the function at `base_value`. If not provided, it is calculated.
        """
        if base_cost is None:
            base_cost = univariate_cost_function(base_value)

        shift = 0.5 * np.pi
        fp = univariate_cost_function(base_value + shift)
        fm = univariate_cost_function(base_value - shift)
        B = np.arctan2(2 * base_cost - fp - fm, fp - fm)
        param_min = base_value - shift - B

        if param_min > np.pi or param_min < -np.pi:
            param_min = np.arctan2(np.sin(param_min), np.cos(param_min))

        return param_min


########################################################################
""" Rotoselect module
"""


class Rotoselect:
    """ Rotoselect """

    def __init__(self,
                 qubit_count: int,
                 hamiltonian: Operator,
                 estimator: ParametricQuantumEstimator[
                     ParametricCircuitQuantumState],
                 init_params,
                 init_pauli_matrices,
                 base_circuit=None):
        self.qubit_count = qubit_count
        self.hamiltonian = hamiltonian
        self.estimator = estimator
        assert len(init_params) == qubit_count
        self.parms = init_params
        assert len(init_pauli_matrices) == qubit_count
        self.pauli_matrices = init_pauli_matrices
        self.base_circuit = base_circuit

    def set_base_circuit(self, base_circuit):
        self.base_circuit = base_circuit

    def _get_parametric_circuit_state(self, pauli_matrices):
        assert len(pauli_matrices) == self.qubit_count

        rotoselect_ansatz = LinearMappedUnboundParametricQuantumCircuit(
            self.qubit_count)
        params_names = [f"theta_{i}" for i in range(self.qubit_count)]
        params_list = rotoselect_ansatz.add_parameters(*params_names)

        for qubit_index, pauli_matrix in enumerate(pauli_matrices):
            if pauli_matrix == SinglePauli.X:
                rotoselect_ansatz.add_ParametricRX_gate(
                    qubit_index, params_list[qubit_index])
            if pauli_matrix == SinglePauli.Y:
                rotoselect_ansatz.add_ParametricRY_gate(
                    qubit_index, params_list[qubit_index])
            if pauli_matrix == SinglePauli.Z:
                rotoselect_ansatz.add_ParametricRZ_gate(
                    qubit_index, params_list[qubit_index])

        idx = 0
        while 2 * idx + 2 < self.qubit_count:
            rotoselect_ansatz.add_CZ_gate(2 * idx + 1, 2 * idx + 2)
            idx += 1

        idx = 0
        while 2 * idx + 1 < self.qubit_count:
            rotoselect_ansatz.add_CZ_gate(2 * idx, 2 * idx + 1)
            idx += 1

        circuit = None
        if self.base_circuit is not None:
            circuit = self.base_circuit.get_mutable_copy()
            circuit += rotoselect_ansatz
        else:
            circuit = rotoselect_ansatz

        parametric_state = ParametricCircuitQuantumState(
            self.qubit_count, circuit)

        return parametric_state

    def cost_function(self, parametric_state, params):
        estimate = self.estimator(self.hamiltonian, parametric_state, params)
        return estimate.value.real

    def step(self):
        params = self.parms.copy()
        pauli_matrices = self.pauli_matrices.copy()

        min_energy = 0
        for param_idx in range(len(params)):
            base_cost = None
            best_energy_value = 1 << 32
            best_param = None
            best_pauli_matrix = None
            for pauli_matrix in range(1, 4):
                pauli_matrices[param_idx] = pauli_matrix
                parametric_state = self._get_parametric_circuit_state(
                    pauli_matrices)

                def univariate(x):
                    params[param_idx] = x
                    return self.cost_function(parametric_state, params)

                if base_cost is None:
                    base_cost = univariate(0.0)

                param_min, value = self.min_analytic(univariate, base_cost)

                if value < best_energy_value:
                    best_energy_value = value
                    best_param = param_min
                    best_pauli_matrix = pauli_matrix

            min_energy = min(min_energy, best_energy_value)
            params[param_idx] = best_param
            pauli_matrices[param_idx] = best_pauli_matrix

        self.parms = params
        self.pauli_matrices = pauli_matrices

        return min_energy

    def min_analytic(self, univarate_cost_function, base_cost):
        shift = 0.5 * np.pi
        fp = univarate_cost_function(shift)
        fm = univarate_cost_function(-shift)
        C = 0.5 * (fp + fm)
        B = np.arctan2(2 * base_cost - fp - fm, fp - fm)
        param_min = -shift - B
        A = np.sqrt((base_cost - C)**2 + 0.25 * (fp - fm)**2)
        value = -A + C

        if param_min <= -2 * shift:
            param_min = param_min + 4 * shift

        return param_min, value


########################################################################


class RunAlgorithm:

    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self, seed: int,
                              hamiltonian_directory: str) -> tuple[Any, float]:
        energy_final = self.get_result(seed, hamiltonian_directory)
        total_shots = challenge_sampling.total_shots
        return energy_final, total_shots

    def get_result(self, seed: int, hamiltonian_directory: str) -> float:
        """
            param seed: the last letter in the Hamiltonian data file, taking one of the values 0,1,2,3,4
            param hamiltonian_directory: directory where hamiltonian data file exists
            return: calculated energy.
        """
        start_time = time.time()

        n_qubits = 28
        ham = problem_hamiltonian(n_qubits, seed, hamiltonian_directory)
        """
        ####################################
        add codes here
        ####################################
        """
        n_site = n_qubits // 2
        total_shots = 2e3
        jw_hamiltonian = jordan_wigner(ham)
        hamiltonian = truncate(operator_from_openfermion_op(jw_hamiltonian),
                               atol=3.5e-2)
        
        shots_allocator = create_proportional_shots_allocator()
        cached_measurement_factory = CachedMeasurementFactory(
            bitwise_commuting_pauli_measurement)
        sampling_estimator = (
            challenge_sampling.create_parametric_sampling_estimator(
                total_shots, cached_measurement_factory, shots_allocator))

        ####################################################################
        ### HVA and Rotosolve ###
           
        # make hf + HV ansatz
        hf_gates = ComputationalBasisState(n_qubits,
                                           bits=2**n_site - 1).circuit.gates
        hf_circuit = LinearMappedUnboundParametricQuantumCircuit(
            n_qubits).combine(hf_gates)
        ansatz = HamiltonianVariational(n_site)
        hf_circuit.extend(ansatz)
        parametric_state = ParametricCircuitQuantumState(n_qubits, hf_circuit)

        optimizer = Rotosolve(num_repeatation=1, ftol=1e-7)

        init_param = [0] * ansatz.parameter_count

        result, energy_history, shots_history = vqe(
            hamiltonian,
            parametric_state,
            sampling_estimator,
            init_param,
            optimizer,
            max_iter=12,
        )

        print(f"shots used in Rotosolve: {challenge_sampling.total_shots}")
        #######################################################################       
        ### Rotoselect ###
        base_circuit = hf_circuit.bind_parameters(result.params)
        init_params_rotoselect = [0] * n_qubits
        init_pauli_matrices = [SinglePauli.Y] * n_qubits

        rotoselcet = Rotoselect(n_qubits, hamiltonian, sampling_estimator,
                                init_params_rotoselect, init_pauli_matrices,
                                base_circuit)
        
        n_iters_rotoselect = 15
        for _ in range(n_iters_rotoselect):
            # break before time limitation
            current_time = time.time()
            if current_time - start_time > 5.5e6:
                break
            try:
                energy = rotoselcet.step()
                print(f"Energy: {energy}")
                energy_history.append(energy)
                shots_history.append(challenge_sampling.total_shots)
            except :
                traceback.print_exc()
                break

        print(f"shots used: {challenge_sampling.total_shots}")
        min_energy = min(energy_history)
        print(f"min energy: {min_energy}")
        return min_energy


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(
        run_algorithm.get_result(seed=0,
                                 hamiltonian_directory="../hamiltonian"))
