from atomate2.jobs.lammps.core import CustomLammpsMaker
from jobflow import job, flow, Maker
from pymatgen.io.lammps.inputs import LammpsInputFile
from atomate2.lammps.schemas.task import LammpsTaskDocument
from dataclasses import dataclass, field
from emmet.core.vasp.calculation import StoreTrajectoryOption
from scipy.integrate import cumulative_trapezoid as cumtrapz
import numpy as np
import scipy.constants as sc
from typing import List

kB = sc.value('Boltzmann constant in eV/K')
eV = sc.value('electron volt')
hbar = sc.value('Planck constant over 2 pi in eV s')
mu = sc.value('atomic mass constant')

@dataclass
class ConstrainedRelaxMaker(CustomLammpsMaker):
    """LAMMPS job maker for constrained relaxations about the center of mass of the structure."""

    name: str = "constrained_relaxation"
    temperature: float = 300.0  # Default temperature in Kelvin
    pressure : float = 1.0  # Default pressure in bar

    def __post_init__(self):
        self.input_file: LammpsInputFile | str = "in.constrained_relaxation"
        
        self.settings.update({
            "temperature": self.temperature,
            "pressure": self.pressure,
        })
        
        self.task_document_kwargs = {
            "store_trajectory": StoreTrajectoryOption.PARTIAL,
            "parse_additional_outputs": ["avg.dat"],
        }
        super().__post_init__()
        self.input_set_generator.calc_type = "Constrained Relaxation"
        
        
@dataclass
class FrenkelLaddMaker(CustomLammpsMaker):
    """LAMMPS job maker for Frenkel Ladd simulations."""

    name: str = "frenkel_ladd"
    temperature: float = 300.0  # Default temperature in Kelvin
    pressure: float = 1.0  # Default pressure in bar
    spring_constants: list[float] = None  # Default spring constants in eV/A^2

    def __post_init__(self):
        self.input_file: LammpsInputFile | str = "in.frenkel_ladd"
        
        self.settings.update({
            "temperature": self.temperature,
            "pressure": self.pressure,
            "spring_constants": self.spring_constants
        })
        
        self.task_document_kwargs = {
            "store_trajectory": StoreTrajectoryOption.PARTIAL,
            "parse_additional_outputs": ["forward.dat", "backward.dat"],
        }
        super().__post_init__()
        self.input_set_generator.calc_type = "Frenkel Ladd"
        

@dataclass
class ReversibleScalingMaker(CustomLammpsMaker):
    """LAMMPS job maker for Reversible Scaling simulations."""

    name: str = "reversible_scaling"
    final_temperature: float = 300.0
    pressure: float = 1.0
    reference_free_energy: dict = None  # Reference free energy dictionary
    def __post_init__(self):
        self.input_file: LammpsInputFile | str = "in.reversible_scaling"
        
        self.settings.update({
            "final_temperature": self.final_temperature,
            "temperature": self.reference_free_energy.keys()[0],  # Assuming the first key is the temperature.
            "pressure": self.pressure,
            "reference_free_energy": self.reference_free_energy
        })
        
        self.task_document_kwargs = {
            "store_trajectory": StoreTrajectoryOption.PARTIAL,
            "parse_additional_outputs": ["forward.dat", "backward.dat"],
        }
        super().__post_init__()
        self.input_set_generator.calc_type = "Reversible Scaling"

@job
def collect_msd_and_equiliberated_structures(relaxation_outputs : List[LammpsTaskDocument] = None) -> dict:
    """
    Collects the MSD and equilibrated structure from a list of relaxation outputs.
    
    Args:
        relaxation_outputs: List of relaxation outputs from LAMMPS jobs.
        initial_structure: Initial structure to use if no equilibration is skipped.
        spring_constants: List of spring constants used if relaxation is skipped. 
        
    Returns:
        Equilibrated structure or initial structure if none found.
    """
    
    output_dict = {}
    temperatures = []
    pressures = []
    structures = []
    spring_constants = {}
    
    for relaxation_output in relaxation_outputs:
        if relaxation_output is None:
            raise ValueError("No relaxation output provided.")

        if relaxation_output.dump_files is None:
            raise ValueError("No dump files found in the relaxation output.")
        
        if relaxation_output.trajectories is None:
            raise ValueError("No trajectories found in the relaxation output.")
        
        structure = relaxation_output.trajectories[0].get_structure(-1)
        temperature = relaxation_output.input_set_generator.settings.get("temperature", 300.0)
        pressure = relaxation_output.input_set_generator.settings.get("pressure", 1.0)
        avg_data = relaxation_output.additional_outputs.get("avg.dat")
        spring_constant = {}
        for i, el in enumerate(structure.composition.elements):
            msd = np.mean(avg_data[6+i][100:])
            spring_constant[el.symbol] = 3*kB*temperature/msd
        temperatures.append(temperature)
        pressures.append(pressure)
        structures.append(structure)
        spring_constants.update(spring_constant)
            
    output_dict.update({'temperatures': temperatures, 
                        'pressures': pressures, 
                        'structures': structures, 
                        'spring_constants': spring_constants})
    return output_dict
        

@job
def process_FL_path(FL_path_taskdocs : List[LammpsTaskDocument], spring_constants : dict = None) -> dict:
    """
    Processes the Frenkel Ladd path to compute the free energy.
    Args:
        FL_path: List of Frenkel Ladd task documents.
    Returns:
        Dictionary with temperatures as keys and free energies as values.
    """
    free_energy_dict = {}
    for task_doc in FL_path_taskdocs:
        if task_doc is None or task_doc.dump_files is None:
            raise ValueError("Invalid Frenkel Ladd task document.")
        
        temperature = task_doc.inputs['in.lammps'].get_args('temperature', default=300.0)
        if spring_constants is None:
            spring_constants = task_doc.inputs['in.lammps'].get_args('spring_constants', default=spring_constants)
        natoms = task_doc.structure.num_sites
        # Load the forward and backward integration data.
        dE, lamb = np.loadtxt(task_doc.additional_outputs.get('forward.dat'), unpack=True)
        I_forw = np.trapz(dE, lamb)
        dE, lamb = np.loadtxt(task_doc.additional_outputs.get('backward.dat'), unpack=True)
        I_back = np.trapz(dE, lamb)
        # Compute the reversible work.
        W = (I_forw - I_back) / 2
        
        F_harm = 0
        for el in task_doc.structure.composition.elements:
            k = spring_constants[el.symbol]
            omega = np.sqrt(k * eV / (el.weight * mu)) * 1.0e+10  # [1/s].    
            F_harm += 3*natoms*kB*temperature * np.log(hbar*omega/(kB*temperature)) # [eV].
        V = (task_doc.structure.volume**1/2) * natoms # Total volume.
        F_CM = (kB*temperature)*np.log((natoms/V) * (2*np.pi*kB*temperature / (natoms*k))**(3/2)) # [eV].
        F = (F_harm + W + F_CM) / natoms
        free_energy_dict[temperature] = {'free_energy': F}
    
    return free_energy_dict


@job
def process_RS_path(RS_path_taskdoc : LammpsTaskDocument, reference_free_energy_dict : dict) -> dict:
    """
    Processes the Reversible Scaling path to compute the free energy.
    Args:
        RS_path_taskdoc: Reversible Scaling path task document.
        reference_free_energy_dict: Dictionary with temperatures as keys and reference free energies as values.
    Returns:
        Dictionary with temperatures as keys and free energies as values.
    """    
    
    if RS_path_taskdoc is None or RS_path_taskdoc.dump_files is None:
        raise ValueError("Invalid RS path task document.")
    temperature = reference_free_energy_dict.keys()[0]  # Assuming the first key is the temperature.
    F0 = reference_free_energy_dict[temperature] # Reference free energy at the given temperature.
    if RS_path_taskdoc.additional_outputs is None:
        raise ValueError("No additional outputs found in the RS path task document.")
    # Load the forward and backward integration data.
    U_f, lamb_f = np.loadtxt(RS_path_taskdoc.additional_outputs.get('forward.dat'), unpack=True)
    U_b, lamb_b = np.loadtxt(RS_path_taskdoc.additional_outputs.get('backward.dat'), unpack=True)
    U_f /= lamb_f
    U_b /= lamb_b

    # Compute work done using cummulative integrals [Eq.(21) in the paper].
    I_f = cumtrapz(U_f,lamb_f,initial=0)
    I_b = cumtrapz(U_b[::-1],lamb_b[::-1],initial=0)
    W = (I_f+I_b) / (2*lamb_f)
    T = temperature / lamb_f
    F = F0/lamb_f + 1.5*kB*T*np.log(lamb_f) + W
    free_energy_dict = {t:f for t, f in zip(T, F)}
    
    return free_energy_dict