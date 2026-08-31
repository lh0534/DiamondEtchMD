"""
STEP_EDGE_surfaces.py — generate the 4 canonical step-edge starting surfaces.

  C(100) 2×1   step_angle=0°   edge ∥ y, lower terrace on −x half
  C(100) 2×1   step_angle=90°  edge ∥ x, lower terrace on −y half
  C(111) 1×1   step_angle=0°   0°-step in crystal coordinates
  C(111) 1×1   step_angle=90°  30°-step in crystal coordinates

Run from the DiamondEtchMD directory:
    python examples/STEP_EDGE_surfaces.py

Submit each surface job individually, e.g.:
    cd step_edges/100_step0; sbatch submit

Open  impact_snaps/0.data  or  ML_impacts.dump  in OVITO after the job finishes.
The step edge is visible as the zmax surface; the lower terrace is one bilayer lower.
"""

from pathlib import Path
from diamond_etch_md import SimSpec, compute_ml, make_sim, validate

OUTDIR = Path("step_edges")

CASES = [
    # (orientation, surface, position, step_angle, depth, invert, label)
    ("100", "2x1",   [0.25, 0.4],   0.0,    2.0,  False,   "100_step0"),
    ("100", "2x1",   [0.1, 0.3],    90.0,   2.0,  False,  "100_step90"),
    ("111", "1x1",   [0.23, 0.35],  0.0,    2.0,  False,  "111_step0"),
    ("111", "1x1",   [0.25, 0.35],  90.0,   2.0,  False,  "111_step90"),
]

BOX = {
    "100": dict(box_x=8, box_y=8, box_depth=2),   # even dims required for 2×1
    "111": dict(box_x=5, box_y=9, box_depth=1),
}

for orient, surface, position, angle, depth, invert, name in CASES:
    box = BOX[orient]
    ml  = compute_ml(orient, box["box_x"], box["box_y"])

    spec = SimSpec(
        orientation         = orient,
        surface             = surface,
        surface_temperature = 300.0,

        # dummy ion — surface-only run (fluence=0)
        species             = "Ar",
        energy              = 50.0,
        fluence             = 0.01,

        ml                  = ml,
        **box,

        # step-edge parameters
        step_edge           = True,
        step_position       = position,
        step_invert         = invert,
        step_angle          = angle,
        step_depth          = depth,

        wall_hours          = 1,
        nice                = 1,
        name                = name,
    )

    validate(spec)
    make_sim(spec, OUTDIR / name)
    print(f"  {name:20s}  C({orient}) {surface}  angle={angle:.4g}°")

print(f"\nGenerated {len(CASES)} step-edge directories under {OUTDIR}/")
print("Submit each with:  cd step_edges/<name>; sbatch submit")
print("Open impact_snaps/0.data in OVITO to inspect the step edge.")
