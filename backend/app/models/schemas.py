from pydantic import BaseModel, Field
from typing import List

class SimRequest(BaseModel):
    f_min: float = Field(20.0, ge=1)
    f_max: float = Field(20000.0, gt=1)
    n_points: int = Field(400, ge=64, le=5000)

    bass_gain_db: float = Field(0.0, ge=-24, le=24)
    treble_gain_db: float = Field(0.0, ge=-24, le=24)
    resonance_hz: float = Field(3000.0, ge=50, le=18000)
    resonance_q: float = Field(1.0, ge=0.2, le=20.0)
    resonance_gain_db: float = Field(0.0, ge=-24, le=24)

class SimResponse(BaseModel):
    freqs: List[float]
    spl_db: List[float]
