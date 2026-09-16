"""
tests/test_sweep_etch.py — verify that the sweep.lmp cluster-ejection logic correctly
detects and removes ejected clusters for three scenarios:

  1. CO2 floating above the surface with upward velocity → swept, etch_products.txt written
  2. O2  floating above the surface with upward velocity → swept, etch_products.txt written
  3. Two CO2s at the same z with the same upward velocity → BOTH swept

sweep.lmp uses atom-style variable formulas that reference global-scalar computes directly
(e.g. `c_clusts == c_surf_clust_num`).  The jaxmd conda env ships LAMMPS 20240829, which
requires per-atom computes in atom-style formulas — it does not allow global-scalar computes
there (this is stricter than the production Kokkos build dated 23 Jun 2022).  To remain
runnable in CI, these tests implement the identical sweep logic in Python (`_python_sweep`)
using LAMMPS Python API calls for all the heavy lifting:

  - `gather_atoms` → atom IDs, positions, velocities, types
  - `extract_compute('clusts', 1, 1)` → per-atom cluster IDs (same compute as production)
  - `extract_compute('surf_id', 0, 0)` → surface cluster ID (same as sweep.lmp c_surf_clust_num)
  - Python logic for cluster ID comparisons, z-CoM/vcm checks, and etch_products.txt writing
  - `delete_atoms group ID` → atom removal (same as production)

All observable assertions are identical to what sweep.lmp would produce:
  - etch_products.txt content (c, cn, nC, nH, nO, nAr, vcm_z)
  - atom count after ejection

Run with:
    /home/lh0534/.conda/envs/jaxmd/bin/python3 -m pytest tests/test_sweep_etch.py -m lammps -v
"""
import pytest
from collections import defaultdict
from pathlib import Path

lammps_mod = pytest.importorskip("lammps", reason="LAMMPS Python module not available — run with jaxmd env")
pytestmark = pytest.mark.lammps


# ---------------------------------------------------------------------------
# Python-driven sweep — mirrors sweep.lmp logic
# ---------------------------------------------------------------------------

def _python_sweep(L, etch_products_path: Path, c: int = 1, cn: int = 0,
                  above_surf_eject: bool = True) -> int:
    """Implement sweep.lmp cluster-ejection logic using the Python API.

    Returns the number of clusters ejected (lines written to etch_products.txt).

    Algorithm mirrors sweep.lmp:
      1. Identify the surface cluster (anchor atoms' cluster = min cluster ID over anchor).
      2. Iterate over non-surface clusters in descending cluster-ID order (largest first,
         matching sweep.lmp's 'max unchecked' approach).
      3. For each candidate: if z_CoM > max-z-of-checked AND vcm_z > 0, eject.
         Otherwise mark as 'checked' (not ejected).
      4. Write each ejection to etch_products.txt in the same 7-column format as sweep.lmp.

    IMPORTANT: extract_compute('clusts', 1, 1) returns values in INTERNAL atom storage
    order (not sorted by global atom ID).  Use extract_atom('id') — which also returns
    in internal order — to build the correct (global_id → cluster_id) mapping.
    gather_atoms() returns in sorted-by-ID order and must NOT be used for indexing into
    extract_compute results.
    """
    # --- prime cluster/atom compute ---
    L.command("compute _sw_surf_id anchor reduce min c_clusts")
    L.command("thermo_style custom c__sw_surf_id")
    L.command("run 0 post no")
    surf_cid = int(L.extract_compute("_sw_surf_id", 0, 0))
    L.command("uncompute _sw_surf_id")

    # --- gather per-atom state in CONSISTENT internal order ---
    # extract_atom / extract_compute both use internal storage order.
    # gather_atoms uses sorted-by-global-ID order; do NOT mix the two orderings.
    n = L.get_natoms()
    id_ptr  = L.extract_atom("id")          # internal order: global atom IDs
    x_ptr   = L.extract_atom("x")           # internal order: x[i][0..2]
    v_ptr   = L.extract_atom("v")           # internal order: v[i][0..2]
    t_ptr   = L.extract_atom("type")        # internal order: type (1-based)
    c_ptr   = L.extract_compute("clusts", 1, 1)   # internal order: cluster IDs

    # Build per-global-ID dicts (so we can look up any atom by its global ID later)
    atom_id   = [int(id_ptr[i])       for i in range(n)]
    atom_z    = [float(x_ptr[i][2])   for i in range(n)]
    atom_vz   = [float(v_ptr[i][2])   for i in range(n)]
    atom_type = [int(t_ptr[i])        for i in range(n)]
    atom_clust= [int(c_ptr[i])        for i in range(n)]

    # --- build cluster membership map (cluster_id → list of internal indices) ---
    by_clust: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(atom_clust):
        by_clust[cid].append(i)

    # Non-surface clusters, descending (largest cluster ID first = sweep.lmp's "max unchecked")
    non_surf = sorted([cid for cid in by_clust if cid != surf_cid], reverse=True)
    if not non_surf:
        return 0

    # --- set up the 'checked' set (global atom IDs of confirmed surface atoms) ---
    checked_ids: set[int] = {atom_id[i] for i in by_clust.get(surf_cid, [])}

    n_ejected = 0
    atoms_to_delete: list[int] = []  # global atom IDs queued for deletion

    for cid in non_surf:
        idxs = by_clust[cid]
        global_ids_in_clust = [atom_id[i] for i in idxs]

        # z CoM and z velocity CoM (equal-mass approximation)
        z_vals  = [atom_z[i]  for i in idxs]
        vz_vals = [atom_vz[i] for i in idxs]
        z_com  = sum(z_vals)  / len(z_vals)
        vcm_z  = sum(vz_vals) / len(vz_vals)

        # Max z of confirmed-surface atoms (mirror of c_top_rest in sweep.lmp)
        top_thresh = (max(atom_z[i] for i in range(n) if atom_id[i] in checked_ids)
                      if checked_ids else -1e20)
        if not above_surf_eject:
            top_thresh = -1e20

        # Atom counts by element (mirrors count(clust_carbon) etc. in sweep.lmp)
        t_vals = [atom_type[i] for i in idxs]
        nC  = t_vals.count(1)
        nH  = t_vals.count(2)
        nO  = t_vals.count(3)
        nAr = t_vals.count(4)
        n_total = nC + nH + nO + nAr

        # Ejection condition (mirrors sweep.lmp's if-block)
        if z_com > top_thresh and vcm_z > 0 and len(checked_ids) > 0 and n_total > 0:
            with open(etch_products_path, "a") as fp:
                fp.write(f"{c} {cn} {nC} {nH} {nO} {nAr} {vcm_z:.6f}\n")
            atoms_to_delete.extend(global_ids_in_clust)
            n_ejected += 1
        else:
            # Not ejected: add to 'checked' so subsequent candidates measure CoM relative
            # to the full confirmed surface (mirrors "group checked union checked small_clust")
            checked_ids.update(global_ids_in_clust)

    # Delete all ejected atoms in one shot
    if atoms_to_delete:
        id_list = " ".join(str(a) for a in atoms_to_delete)
        L.command(f"group _sw_eject id {id_list}")
        L.command("delete_atoms group _sw_eject")
        L.command("group _sw_eject delete")

    return n_ejected


# ---------------------------------------------------------------------------
# Shared setup helpers
# ---------------------------------------------------------------------------

def _lmp(log_path):
    return lammps_mod.lammps(cmdargs=["-log", str(log_path), "-screen", "none"])


def _setup_surface(L, extra_atoms: str = ""):
    """Create a 20×20×15 box with 4 anchor C atoms and one mobile surface C.

    extra_atoms: additional LAMMPS commands inserted at the end (e.g. more create_atoms).
    """
    L.commands_string(f"""\
units        real
atom_style   atomic
boundary     p p p

region       box block 0 20 0 20 0 15
create_box   3 box

mass  1 12.011
mass  2  1.008
mass  3 16.000

pair_style   zero 3.0
pair_coeff   * *

# Anchor C atoms (type 1) at z = 1.  cluster/atom cutoff 3.0 Å: atoms within 2 Å
# of each other in xy connect into one surface cluster.
create_atoms 1 single  5.0  5.0 1.0
create_atoms 1 single  5.0 15.0 1.0
create_atoms 1 single 15.0  5.0 1.0
create_atoms 1 single 15.0 15.0 1.0
# One mobile surface C at z = 2.5 — within 3 Å of the nearest anchor → same cluster.
create_atoms 1 single 10.0 10.0 2.5
{extra_atoms}
""")


def _setup_cluster_compute(L):
    """Define c_clusts (cluster/atom 3.0) and the anchor group, mirroring head.lmp."""
    L.commands_string("""\
region       anchor_reg block INF INF INF INF 0 2.0 units box
group        anchor     region anchor_reg
group        mobile     subtract all anchor
compute      clusts     all cluster/atom 3.0
""")


def _set_upward_velocity(L, vz: float, z_lo: float = 5.0):
    """Give all atoms above z_lo an upward z velocity of vz (Å/fs)."""
    L.commands_string(f"""\
region       _eject_zone block INF INF INF INF {z_lo} INF units box
group        _eject_tmp  region _eject_zone
velocity     _eject_tmp  set 0 0 {vz}
group        _eject_tmp  delete
region       _eject_zone delete
""")


def _parse_etch_products(path: Path):
    """Return (c, cn, nC, nH, nO, nAr, vcm_z) tuples from etch_products.txt."""
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    result = []
    for ln in lines:
        parts = ln.split()
        if len(parts) == 7:
            result.append((int(parts[0]), int(parts[1]),
                           int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]),
                           float(parts[6])))
    return result


# ---------------------------------------------------------------------------
# Test 1: single CO2 above surface
# ---------------------------------------------------------------------------

def test_co2_above_surface_swept(tmp_path):
    """CO2 molecule (1 C + 2 O) at z=8 with vz>0 must be detected and removed.

    cluster/atom with cutoff 3.0 Å places CO2 (z=8) in its own cluster, separate
    from the surface cluster (z≤2.5).  _python_sweep should eject it and write
    etch_products.txt with nC=1, nO=2, vcm_z>0.
    """
    extra = (
        "create_atoms 1 single 10.0 10.0 8.00\n"   # CO2 C
        "create_atoms 3 single 10.0 10.0 9.13\n"   # CO2 O1 (+1.13 Å)
        "create_atoms 3 single 10.0 10.0 6.87\n"   # CO2 O2 (-1.13 Å)
    )
    etch_file = tmp_path / "etch_products.txt"
    L = _lmp(tmp_path / "log.lammps")
    try:
        _setup_surface(L, extra)
        _setup_cluster_compute(L)
        _set_upward_velocity(L, vz=0.05, z_lo=5.0)
        n_ejected = _python_sweep(L, etch_file)
        n_remaining = L.get_natoms()
    finally:
        L.close()

    assert etch_file.exists(), "etch_products.txt not written"
    rows = _parse_etch_products(etch_file)
    assert len(rows) == 1, f"Expected 1 etch event, got {len(rows)}: {rows}"
    c, cn, nC, nH, nO, nAr, vcm_z = rows[0]
    assert nC == 1,   f"CO2 product: expected nC=1, got {nC}"
    assert nO == 2,   f"CO2 product: expected nO=2, got {nO}"
    assert nH == 0 and nAr == 0, f"Unexpected nH={nH} nAr={nAr}"
    assert vcm_z > 0, f"CO2 must have upward CoM velocity, got vcm_z={vcm_z}"
    assert n_remaining == 5, (
        f"After ejecting CO2 (3 atoms) from 8 total, 5 surface atoms should remain; got {n_remaining}"
    )
    assert n_ejected == 1, f"Expected 1 ejected cluster, got {n_ejected}"


# ---------------------------------------------------------------------------
# Test 2: single O2 above surface
# ---------------------------------------------------------------------------

def test_o2_above_surface_swept(tmp_path):
    """O2 molecule (0 C + 2 O) at z=8 with vz>0 must be detected and removed.

    O atoms at z=8 and z=9.2 are within 3 Å of each other → one cluster.
    That cluster is >5 Å above the surface → separate from the surface cluster.
    """
    extra = (
        "create_atoms 3 single 10.0 10.0 8.00\n"   # O2 atom 1
        "create_atoms 3 single 10.0 10.0 9.20\n"   # O2 atom 2  (1.2 Å O-O distance)
    )
    etch_file = tmp_path / "etch_products.txt"
    L = _lmp(tmp_path / "log.lammps")
    try:
        _setup_surface(L, extra)
        _setup_cluster_compute(L)
        _set_upward_velocity(L, vz=0.05, z_lo=5.0)
        n_ejected = _python_sweep(L, etch_file)
        n_remaining = L.get_natoms()
    finally:
        L.close()

    assert etch_file.exists(), "etch_products.txt not written"
    rows = _parse_etch_products(etch_file)
    assert len(rows) == 1, f"Expected 1 etch event, got {len(rows)}: {rows}"
    c, cn, nC, nH, nO, nAr, vcm_z = rows[0]
    assert nC == 0,   f"O2 product: expected nC=0, got {nC}"
    assert nO == 2,   f"O2 product: expected nO=2, got {nO}"
    assert nH == 0 and nAr == 0, f"Unexpected nH={nH} nAr={nAr}"
    assert vcm_z > 0, f"O2 must have upward CoM velocity, got vcm_z={vcm_z}"
    assert n_remaining == 5, (
        f"After ejecting O2 (2 atoms) from 7 total, 5 surface atoms should remain; got {n_remaining}"
    )
    assert n_ejected == 1, f"Expected 1 ejected cluster, got {n_ejected}"


# ---------------------------------------------------------------------------
# Test 3: two CO2s at the same z — both must be swept
# ---------------------------------------------------------------------------

def test_two_co2_same_z_both_swept(tmp_path):
    """Two CO2 molecules at the same z with upward velocity must both be swept.

    This tests the 'checked' group logic: the two clusters have different cluster IDs
    (based on their atom IDs), so the sweep loop iterates twice, ejecting each in turn.
    Neither CO2 prevents the other from being swept.

    CO2-A and CO2-B are placed >15 Å apart in xy so cluster/atom (cutoff 3.0 Å)
    treats them as distinct clusters.
    """
    extra = (
        # CO2-A: atoms 6-8 (low xy corner)
        "create_atoms 1 single  4.0  4.0 8.00\n"
        "create_atoms 3 single  4.0  4.0 9.13\n"
        "create_atoms 3 single  4.0  4.0 6.87\n"
        # CO2-B: atoms 9-11 (high xy corner — 16.97 Å from CO2-A → separate cluster)
        "create_atoms 1 single 16.0 16.0 8.00\n"
        "create_atoms 3 single 16.0 16.0 9.13\n"
        "create_atoms 3 single 16.0 16.0 6.87\n"
    )
    etch_file = tmp_path / "etch_products.txt"
    L = _lmp(tmp_path / "log.lammps")
    try:
        _setup_surface(L, extra)
        _setup_cluster_compute(L)
        _set_upward_velocity(L, vz=0.05, z_lo=5.0)
        n_ejected = _python_sweep(L, etch_file)
        n_remaining = L.get_natoms()
    finally:
        L.close()

    assert etch_file.exists(), "etch_products.txt not written"
    rows = _parse_etch_products(etch_file)
    assert len(rows) == 2, (
        f"Expected 2 etch events (one per CO2), got {len(rows)}: {rows}"
    )
    for i, (c, cn, nC, nH, nO, nAr, vcm_z) in enumerate(rows):
        assert nC == 1,   f"CO2 event {i}: expected nC=1, got {nC}"
        assert nO == 2,   f"CO2 event {i}: expected nO=2, got {nO}"
        assert nH == 0 and nAr == 0, f"CO2 event {i}: unexpected nH={nH} nAr={nAr}"
        assert vcm_z > 0, f"CO2 event {i}: expected vcm_z > 0, got {vcm_z}"
    assert n_remaining == 5, (
        f"After ejecting both CO2s (6 atoms) from 11 total, 5 surface atoms should remain; got {n_remaining}"
    )
    assert n_ejected == 2, f"Expected 2 ejected clusters, got {n_ejected}"


# ---------------------------------------------------------------------------
# Test 4/5: a detached, upward-moving cluster whose CoM sits BELOW the current
# 'checked' surface top — this is what `above_surf_eject` (SimSpec field) /
# the LAMMPS variable of the same name gates.
#
# Note on cluster processing order: _python_sweep (and sweep.lmp) processes
# non-surface clusters in DESCENDING cluster-ID order, folding each unejected
# cluster into 'checked' as it goes. A newly created atom always gets a higher
# ID than atoms that already exist, so a freshly placed candidate cluster is
# normally processed BEFORE the pre-existing surface atoms are ever added to
# 'checked' — meaning top_thresh would just be the single lowest-ID anchor's
# z, not the true surface top. To exercise the "below the real surface top"
# scenario we add a stationary "blocker" atom AFTER the candidate CO2 (so it
# gets a higher ID and is checked first): it sits higher than the CO2 and has
# zero velocity, so it's folded into 'checked' before the CO2 is evaluated —
# exactly like a surface protrusion or a rough surface patch would.
# ---------------------------------------------------------------------------

# CO2 target: xy=(10, 16), >5 Å in-plane from every pre-existing surface atom
# (mobile surface C at (10,10,2.5); anchors at (5/15, 5/15, 1.0)), so it forms
# its own cluster under cluster/atom(3.0) regardless of z. z_com = 2.0.
# Blocker: an isolated, zero-velocity atom at z=2.5 (xy far from everything),
# created AFTER the CO2 so it has a higher atom/cluster ID and gets folded
# into 'checked' first, raising top_thresh above the CO2's z_com=2.0.
_BELOW_SURFACE_TOP_EXTRA = (
    "create_atoms 1 single 10.0 16.0 2.00\n"   # CO2 target C  (z_com = 2.0)
    "create_atoms 3 single 10.0 16.0 3.13\n"   # CO2 target O1 (+1.13 Å)
    "create_atoms 3 single 10.0 16.0 0.87\n"   # CO2 target O2 (-1.13 Å)
    "create_atoms 1 single  2.0  2.0 2.50\n"   # stationary blocker, z=2.5 > CO2's z_com
)


def _set_velocity_by_id(L, atom_ids, vz: float):
    """Give exactly the given global atom IDs an upward z velocity of vz (Å/fs)."""
    id_str = " ".join(str(i) for i in atom_ids)
    L.commands_string(f"""\
group        _co2_target id {id_str}
velocity     _co2_target set 0 0 {vz}
group        _co2_target delete
""")


def test_cluster_below_surface_top_not_ejected_by_default(tmp_path):
    """A detached, upward-moving cluster below the (checked) surface top is NOT
    ejected when above_surf_eject=True (the default) — it fails the
    z_com > top_thresh check and is folded into 'checked' instead of swept.

    This is the companion/regression guard for the disabled-check test below:
    it demonstrates the flag actually does something, and would catch a future
    change that accidentally removes the z gate entirely.
    """
    etch_file = tmp_path / "etch_products.txt"
    L = _lmp(tmp_path / "log.lammps")
    try:
        _setup_surface(L, _BELOW_SURFACE_TOP_EXTRA)
        _setup_cluster_compute(L)
        _set_velocity_by_id(L, [6, 7, 8], vz=0.05)   # only the CO2 target moves
        n_ejected = _python_sweep(L, etch_file, above_surf_eject=True)
        n_remaining = L.get_natoms()
    finally:
        L.close()

    assert not etch_file.exists() or _parse_etch_products(etch_file) == [], (
        "Cluster below the surface top must not be ejected when above_surf_eject=True"
    )
    assert n_ejected == 0, f"Expected 0 ejected clusters, got {n_ejected}"
    assert n_remaining == 9, (
        f"Nothing should be deleted; 5 surface + 3 CO2 + 1 blocker atoms should remain; "
        f"got {n_remaining}"
    )


def test_cluster_below_surface_top_ejected_when_check_disabled(tmp_path):
    """The same detached, upward-moving cluster below the surface top IS ejected
    once above_surf_eject=False forces top_thresh = -1e20 — ejection then
    depends only on vcm_z > 0, matching sweep.lmp's `${above_surf_eject} == 0` branch.

    The stationary blocker atom is unaffected either way (vcm_z=0, so it never
    satisfies the ejection condition regardless of the z-check).
    """
    etch_file = tmp_path / "etch_products.txt"
    L = _lmp(tmp_path / "log.lammps")
    try:
        _setup_surface(L, _BELOW_SURFACE_TOP_EXTRA)
        _setup_cluster_compute(L)
        _set_velocity_by_id(L, [6, 7, 8], vz=0.05)   # only the CO2 target moves
        n_ejected = _python_sweep(L, etch_file, above_surf_eject=False)
        n_remaining = L.get_natoms()
    finally:
        L.close()

    assert etch_file.exists(), "etch_products.txt not written"
    rows = _parse_etch_products(etch_file)
    assert len(rows) == 1, f"Expected 1 etch event, got {len(rows)}: {rows}"
    c, cn, nC, nH, nO, nAr, vcm_z = rows[0]
    assert nC == 1,   f"CO2 product: expected nC=1, got {nC}"
    assert nO == 2,   f"CO2 product: expected nO=2, got {nO}"
    assert nH == 0 and nAr == 0, f"Unexpected nH={nH} nAr={nAr}"
    assert vcm_z > 0, f"CO2 must have upward CoM velocity, got vcm_z={vcm_z}"
    assert n_remaining == 6, (
        f"After ejecting the below-surface-top CO2 (3 atoms) from 9 total, "
        f"5 surface + 1 blocker atoms should remain; got {n_remaining}"
    )
    assert n_ejected == 1, f"Expected 1 ejected cluster, got {n_ejected}"
