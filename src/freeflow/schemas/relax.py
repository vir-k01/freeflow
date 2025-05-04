from typing import Any, Optional, List
from pydantic import BaseModel, Field

class ConstrainedRelaxationTask(BaseModel):
    '''
    Task schema for constrained relaxation in LAMMPS.
    '''
    task_label: Optional[str] = Field(None, description='task_label')
    temperature: float = Field(..., description='Temperature in Kelvin')
    pressure: float = Field(..., description='Pressure in bar')
    constraints: List[Any] = Field(..., description='List of constraints for the relaxation')
    
