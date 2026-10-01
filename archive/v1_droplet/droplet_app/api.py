"""REST API for beta_max predictions.

Run:   uvicorn api:app --port 8000        (from inside droplet_app/)
Docs:  http://localhost:8000/v1/api/docs

Routes live under /v1/api (the public path on Vercel) and are also answered at the bare paths
(/predict, /health, ...) so local scripts and a prefix-stripping proxy both work.
"""
from typing import Optional
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field
from core import Predictor, fluids, surfaces

PREFIX = "/v1/api"
app = FastAPI(title="Droplet β_max API", version="1.0",
              description="Maximum spreading ratio of a droplet impacting a laser-textured or smooth surface.",
              docs_url=f"{PREFIX}/docs", redoc_url=None, openapi_url=f"{PREFIX}/openapi.json")
router = APIRouter()
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


@router.get("/health")
def health():
    return {"status": "ok", "model": P.meta["model_name"], "model_key": P.meta["model_key"]}


@router.get("/fluids")
def get_fluids(): return FL


@router.get("/surfaces")
def get_surfaces(): return SF


@router.post("/predict")
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


app.include_router(router, prefix=PREFIX)
app.include_router(router, include_in_schema=False)
