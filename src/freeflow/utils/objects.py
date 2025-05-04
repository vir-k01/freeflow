from monty.json import MSONable

class ReferenceState(MSONable):
    """
    Enum for reference states.
    """
    EINSTEIN = "einstein_solid"
    LIQUID = "liquid"
    HARMONIC = "harmonic_solid"
    IDEAL = "ideal_gas"

    @classmethod
    def values(cls):
        return [cls.SOLID, cls.LIQUID]

    @classmethod
    def is_valid(cls, value):
        return value in cls.values()