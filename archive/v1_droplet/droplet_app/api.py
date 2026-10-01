"""REST API for beta_max predictions.

Run:   uvicorn api:app --port 8000        (from inside droplet_app/)
Docs:  http://localhost:8000/docs
"""
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from core import Predictor, fluids, surfaces

app = FastAPI(title="Droplet β_max API", version="1.0",
              description="Maximum spreading ratio of a droplet impacting a laser-textured or smooth surface.")
P = Predictor()
FL, SF = fluids(), surfaces()


class Impact(BaseModel):
    D_mm: float = Field(..., gt=0, description="Droplet diameter in mm", examples=[2.5])
    V: float = Field(..., gt=0, description="Impact velocity in m/s", examples=[1.5])
    fluid: Optional[str] = Field(None, description="One of GET /fluids, e.g. '0 wt.% glycerol'")
    rho: Optional[float] = Field(None, gt=0, description="Density kg/m³ (if no fluid name)")
    sigma: Optional[float] = Field(None, gt=0, description="Surface tension N/m (if no fluid name)")
    mu: Optional[float] = Field(None, gt=0, description="Viscosity Pa·s (if no fluid name)")
    surface: Optional[str] = Field(None, description="One of GET /surfaces, e.g. 'D200' or 'REF-H'")
    spacing_um: Optional[float] = Field(None, ge=0, description="Laser track pitch in µm (if no surface name)")
    depth_um: Optional[float] = Field(None, ge=0, description="Channel depth in µm (if no surface name)")


@app.get("/health")
def health():
    return {"status": "ok", "model": P.meta["model_name"], "model_key": P.meta["model_key"]}


@app.get("/fluids")
def get_fluids(): return FL


@app.get("/surfaces")
def get_surfaces(): return SF


@app.post("/predict")
def predict(x: Impact):
    if x.fluid:
        if x.fluid not in FL: raise HTTPException(422, f"Unknown fluid. Choose one of: {list(FL)}")
        rho, sigma, mu = FL[x.fluid]["rho"], FL[x.fluid]["sigma"], FL[x.fluid]["mu"]
    elif None not in (x.rho, x.sigma, x.mu):
        rho, sigma, mu = x.rho, x.sigma, x.mu
    else:
        raise HTTPException(422, "Give either a fluid name or all of rho, sigma and mu.")
    if x.surface:
        if x.surface not in SF: raise HTTPException(422, f"Unknown surface. Choose one of: {list(SF)}")
        sp, dp = SF[x.surface]["spacing_um"], SF[x.surface]["depth_um"]
    elif (x.spacing_um is None) != (x.depth_um is None):
        raise HTTPException(422, "Give both spacing_um and depth_um (or a surface name). Omit both for a smooth plate.")
    else:
        sp, dp = x.spacing_um or 0.0, x.depth_um or 0.0
    return P.predict(x.D_mm, x.V, rho, sigma, mu, sp, dp)
