from ..jobs.base import ConstrainedRelaxMaker, FrenkelLaddMaker, \
    ReversibleScalingMaker, collect_msd_and_equiliberated_structures, \
        process_FL_path, process_RS_path
from jobflow import Maker, Flow
from dataclasses import dataclass, field
from ..utils.objects import ReferenceState
from atomate2.lammps.jobs.core import CustomLammpsMaker

class BaseFreeEnergyMaker(Maker):
    
    equilibration_maker: ConstrainedRelaxMaker = None
    free_energy_maker: CustomLammpsMaker = None
    reference_state: ReferenceState = ReferenceState.IDEAL
    
    def make():
        pass

class FLFreeEnergyMaker(BaseFreeEnergyMaker):
    """
    Free energy maker for Frenkel Ladd simulations.
    
    This class is responsible for setting up the flow for Frenkel Ladd free energy calculations.
    It uses the ConstrainedRelaxMaker for equilibration and the FrenkelLaddMaker for the free energy calculation.
    """
    
    equilibrate : bool = True  # Whether to perform equilibration before free energy calculation
    free_energy_maker: CustomLammpsMaker = field(default_factory=FrenkelLaddMaker)
    equilibration_maker : ConstrainedRelaxMaker = field(default_factory=ConstrainedRelaxMaker)
    temperatures = [300.0]  # Default temperature in Kelvin
    pressures = [1.0]  # Default pressure in bar
    spring_constants : dict = None # Default spring constants in eV/A^2 if not equiliberating
    
    def make(self, structure, prev_dir=None):
        
        jobs = []   
        
        for T, P in zip(self.temperatures, self.pressures):
            if self.equilibrate:
                self.spring_constants[T] = {}
                equilibration_flow = self.equilibration_maker.make(
                    structure=structure,
                    prev_dir=prev_dir,
                    temperature=T,
                    pressure=P
                )
                jobs.append(equilibration_flow)
                jobs.append(collect_msd_and_equiliberated_structures(jobs))
                structure = jobs[-1].output.get('structures', [structure])[0]
                self.spring_constants[T].update({P: jobs[-1].output['spring_constants']})
            
            free_energy_flow = self.free_energy_maker.make(
                structure=structure,
                prev_dir=prev_dir,
                temperatures=[T],
                pressures=[P],
                spring_constants=self.spring_constants
            )        

            jobs.append(free_energy_flow)
        
        return Flow(
            [equilibration_flow, free_energy_flow],
            name="Frenkel Ladd Free Energy Calculation"
        )
    

class ReversibleScalingFlow(BaseFreeEnergyMaker):
    """
    Free energy maker for Reversible Scaling simulations.
    
    This class is responsible for setting up the flow for Reversible Scaling free energy calculations.
    It uses the ConstrainedRelaxMaker for equilibration and the ReversibleScalingMaker for the free energy calculation.
    """
    
    equilibrate : bool = True  # Whether to perform equilibration before free energy calculation
    free_energy_maker: CustomLammpsMaker = field(default_factory=ReversibleScalingMaker)
    equilibration_maker : ConstrainedRelaxMaker = field(default_factory=ConstrainedRelaxMaker)
    
    def make(self, structure, prev_dir=None):
        
        jobs = []
        
        if self.equilibrate:
            equilibration_flow = self.equilibration_maker.make(structure, prev_dir=prev_dir)
            jobs.append(equilibration_flow)
            jobs.append(collect_msd_and_equiliberated_structures(jobs))
            structure = jobs[-1].output.get('structures', [structure])[0]
        
        free_energy_flow = self.free_energy_maker.make(
            structure=structure,
            prev_dir=prev_dir
        )
        jobs.append(free_energy_flow)
        
        return Flow(
            [equilibration_flow, free_energy_flow],
            name="Reversible Scaling Free Energy Calculation"
        )