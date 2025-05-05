from .base import BaseFreeEnergyMaker
from ..jobs.base import ConstrainedRelaxMaker, FrenkelLaddMaker, \
    ReversibleScalingMaker, collect_msd_and_equiliberated_structures, \
    process_FL_path, process_RS_path, CustomLammpsMaker
from freeflow.lammps.jobs.base import LammpsInputFile
from dataclasses import dataclass, field
from freeflow.utils.objects import ReferenceState
import numpy as np
from jobflow import Flow


class TemperatureDependentFreeEnergyMaker(BaseFreeEnergyMaker):
    equilibration_maker : CustomLammpsMaker = field(default_factory=ConstrainedRelaxMaker)
    fl_free_energy_maker: CustomLammpsMaker = field(default_factory=FrenkelLaddMaker)
    rs_free_energy_maker: CustomLammpsMaker = field(default_factory=ReversibleScalingMaker)
    temperatures: list[float] = field(default_factory=lambda: [300.0, 1000.0])  # Default temperature in Kelvin
    pressures: list[float] = field(default_factory=lambda: [1.0])  # Default pressure in bar
    spring_constants: dict = field(default_factory=dict)  # Default spring constants in eV/A^2 if not equilibrating
    
    
    def make(self, structure, prev_dir=None):
        
        eq_jobs = []
        fe_jobs = []
        process_jobs = []

        if len(self.temperatures) != len(self.pressures):
            if len(self.pressures) == 1:
                self.pressures = self.pressures * len(self.temperatures)
            else:
                raise ValueError("Temperatures and pressures must have the same length or pressures must be a single value.")
        
        if len(self.temperatures) == 2:
            T_diff = self.temperatures[1] - self.temperatures[0]   
            if T_diff // 1000 != 0:
                self.temperatures.append(np.ceil(T_diff/2) + self.temperatures[0])
            self.temperatures.sort()
        
        for T, P in zip(self.temperatures, self.pressures):
            if self.equilibration_maker:
                equilibration_flow = self.equilibration_maker.make(
                    structure=structure,
                    prev_dir=prev_dir,
                    temperature=T,
                    pressure=P
                )
                eq_jobs.append(equilibration_flow)
                eq_jobs.append(collect_msd_and_equiliberated_structures(eq_jobs))
                structure = eq_jobs[-1].output.get('structures', [structure])[0]
                self.spring_constants[T].update({P: eq_jobs[-1].output['spring_constants']})

            fl_free_energy_flow = self.fl_free_energy_maker.make(
                structure=structure,
                prev_dir=prev_dir,
                temperature=T,
                pressure=P
            )
            fe_jobs.append(fl_free_energy_flow)
            
        process_job = process_FL_path(fe_jobs, self.spring_constants)
        process_jobs.append(process_job)
        free_energy_dict = process_job.output.get('free_energy', {})

        rs_free_energy_flow = self.rs_free_energy_maker.make(
            structure=structure,
            prev_dir=prev_dir,
            final_temperature=self.temperatures[-1],
            pressure=self.pressures[0],
        )
        fe_jobs.append(rs_free_energy_flow)
        
        process_rs = process_RS_path(rs_free_energy_flow, free_energy_dict)
        process_jobs.append(process_rs)
        
        jobs = eq_jobs + fe_jobs + process_jobs

        return Flow(
            jobs,
            name="Temperature Dependent Free Energy Calculation",
            output=process_rs.output.get('free_energy', {})
            )