# `fast_wind` problem generator and the `capture_cool` setups

This document describes how the fast-wind runaway-BH problem is implemented in AthenaK.
It covers the `fast_wind` problem generator, the source-term changes it relies on, and
the three input files in this directory.

| File | Purpose |
|---|---|
| [`src/pgen/fast_wind.cpp`](../../../src/pgen/fast_wind.cpp) | Problem generator |
| [`src/srcterms/srcterms.{hpp,cpp}`](../../../src/srcterms/srcterms.cpp), [`srcterms_newdt.cpp`](../../../src/srcterms/srcterms_newdt.cpp), [`ismcooling.hpp`](../../../src/srcterms/ismcooling.hpp) | Density-matched heating and the shared `CoolLambda()` lookup |
| [`fast_wind.athinput`](fast_wind.athinput) | Adiabatic 0.1c wind, BH at 100 km/s (no wind cooling) |
| [`capture_cool.athinput`](capture_cool.athinput) | Cooled (momentum-driven) 0.03c wind, BH at 1000 km/s, n_cgm = 1 |
| [`capture_cool_n10.athinput`](capture_cool_n10.athinput) | Same as `capture_cool` with n_cgm = 10 and 10x the wind momentum |

Build: `./make_build_file.sh fast_wind` (double) or `./make_build_single.sh fast_wind`
(single). The 0.1c `fast_wind` run must be double precision: in the free wind
E_kin/E_th ~ 1e6, so float loses the pressure. The `capture_cool` runs work in single.

---

## 1. Geometry and frame

- 3D Cartesian, **BH frame**: the BH sits at the origin and the CGM flows in through
  the outer-x1 face with velocity -v_bh x̂. The wake leaves through inner-x1.
- Boundaries: `ox1_bc = user` (uniform CGM inflow), all other faces `outflow`.
- Static mesh refinement concentrates resolution around the BH and the stand-off region
  (see the mesh tables in each input file).

## 2. Physics in `fast_wind.cpp`

All input parameters are converted to code units on the host in double and cast to
`Real` once, so the same source works in single and double precision.

### 2.1 Ambient CGM

- Uniform `n_cgm` [cm⁻³] and `T_cgm` [K], with ρ = n μ m_u (μ from `<units>`).
- Used as the initial condition everywhere (velocity -v_bh x̂, tracer 0) and imposed
  every step in the outer-x1 ghost cells by `InflowX1BC()`. Both `w0` and `u0` are set
  in the ghosts.

### 2.2 Wind injection: hard reset (`ResetWind()`)

Every RK stage, every cell with r < `r_inj` is overwritten with the free-wind solution:

```
rho = Mdot / (4 pi r_eff^2 v_w),   r_eff = max(r, r_inj/4)
v   = v_w r_hat
T   = T_w            (cold, unshocked wind; default 1e4 K)
s_00 = rho           (wind mass fraction = 1)
s_01 = 0             (impact-parameter tracer, if present)
```

- r_eff = max(r, r_inj/4) avoids the 1/r² divergence at the centre. Cells outside
  r_inj/4 have the exact free-wind density, so the mass flux through r_inj is exactly
  Mdot.
- The reset is called last in `AddUserSrcs()`, after gravity and wind cooling. User
  sources run last in the hydro source-term step, so the reset has the final say each
  stage.
- It is also applied once at the end of the initial conditions.

### 2.3 BH gravity

Softened point mass, applied outside r_inj from the primitives `w0`:

```
g = -G M_bh r_vec / (r^2 + eps^2)^{3/2}
```

The momentum and energy (ρ g·v) are both updated. (r²+ε²)^{3/2} is computed as
s2*sqrt(s2) rather than with `pow`, to avoid `pow(Real,double)` on the device.

### 2.4 Wind cooling: momentum-driven limit (`CoolWind()`, `wind_cool = true`)

This option models a shocked wind that loses its thermal pressure immediately, so the
wind acts on the CGM through its **momentum only**.

Every stage, before the wind reset, for each cell with wind fraction
f_w = s_00/ρ ≥ `wind_cool_fmin` (default 0.5):

```
e_kin = |M|^2 / (2 rho)
e_th  = E - e_kin
if e_th > rho T_wind_cool / (gamma-1):
    E = e_kin + rho T_wind_cool / (gamma-1)
```

- Density, momentum and tracers are unchanged. Only the excess internal energy is
  removed, so the reset never heats gas.
- It acts on `u0` after this stage's update, so exactly the excess energy is removed.
- CGM-dominated cells (f_w < `wind_cool_fmin`) are never touched. Cooling of shocked
  or compressed CGM is left to the normal `ism_cooling` source term (§3).
- This is an instantaneous reset (t_cool → 0), not a physical cooling rate. See §6 for
  how far it can be justified physically.

### 2.5 Passive scalars

| Scalar | Meaning | Set where |
|---|---|---|
| `s_00` | wind mass fraction f_w (1 in wind, 0 in CGM) | reset region = 1; IC and inflow = 0 |
| `s_01` (only if `nscalars >= 2`) | impact parameter b = √(y²+z²) of the inflowing CGM, in code length | IC and inflow ghosts = local b; wind = 0 |

`s_01` is advected with the gas, so a captured parcel near the BH still carries the
impact parameter at which it entered the box. This shows where captured gas came from.

### 2.6 Startup diagnostics

Rank 0 prints the code-unit densities and pressures, the Mach number of v_bh, ρ_w(r_inj),
the stand-off radius

```
R_0 = sqrt( Mdot v_w / (4 pi (rho v_bh^2 + P_cgm)) )      (ram + thermal)
R_0,ram = sqrt( Mdot v_w / (4 pi rho v_bh^2) )            (ram only)
```

as well as R_0/r_inj, ṗ_w = Mdot v_w in dyn, R_cap = 2 R_0,ram, and the wind-cooling
and heating settings. Check these against the comment block of the input file.

### 2.7 History output (`user_hist = true`) → `<basename>.user.hst`

Computed by `CaptureHistory()`. "CGM gas" means cells with f_w < 0.5, weighted by the CGM
mass density ρ(1 - f_w).

| Column | Definition | Units |
|---|---|---|
| `M_halt` | CGM mass within r < `r_diag` with v_x > -v_bh/2 (halted, i.e. dragged along with the BH) | Msun |
| `Mdot_up` | Outward CGM mass flux (v_r > 0) through the upstream hemisphere (x > 0) of the shell r = `r_diag` (cells within dx/2 of the shell) | Msun/yr |
| `Mh_b0.0` … `Mh_b3.0` | `M_halt` split by entry impact parameter b/R_0 into bins [0,.5), [.5,1), …, [3,∞) | Msun |

`Mdot_up` is zero for undisturbed inflow. `r_diag` defaults to 2 R_0. The `Mh_b*`
columns are 0 unless `nscalars >= 2`. Sums are rank-local reductions in double, and
the history output sums over ranks.

## 3. Cooling and heating of the CGM (`<hydro_srcterms>`)

The setups use AthenaK's `ism_cooling` with the tabulated Λ(T) (`cool_table = true`: a
float table, uniform in log₁₀T from 10 K to 10⁹ K, evaluated through `CoolLambda()` in
`ismcooling.hpp`). There are two ways to keep the ambient 10⁶ K CGM from cooling away.

### 3.1 Density-matched heating (`heat_balance = true`), used by `capture_cool*`

A constant heating rate Γ = n_cgm Λ(T_cgm) leaves the 10⁶ K ambient medium thermally
unstable: it runs away on ~t_cool. Instead the heating is tied to the *local* Λ(T):

```
heating per particle = n_cgm Lambda(T)     for T < T_heat_max (default 3e6 K)
net cooling per volume = n (n - n_cgm) Lambda(T)
```

- Ambient gas (n = n_cgm) has zero net cooling at **any** T, so it cannot run away.
- Compressed gas (n > n_cgm) cools and rarefied gas below T_heat_max is heated.
- Gas above T_heat_max (e.g. shocked CGM) cools at the full n²Λ(T).

Implementation:
- `SourceTerms` has two new members, `heat_ref_rho` (off when ≤ 0) and `heat_tmax`.
  The problem generator sets `heat_ref_rho = rho_cgm` and `heat_tmax = T_heat_max`, and
  forces `hrate = 0`. It aborts if `ism_cooling` is off.
- The extra heating term `heat_ref_rho * Lambda(T)` (for T < `heat_tmax`) is added in
  every cooling path, so they stay consistent:
  - the explicit `ISMCooling()`;
  - the sub-cycled `SubcycleISMCooling()` (both the update and the sub-step size
    estimate);
  - `MinCoolingTime()`;
  - the cooling time-step limit in `srcterms_newdt.cpp`.

### 3.2 Cooling mask (`cool_mask_scalar = 0`), used by `fast_wind.athinput`

Cells with s_00 below `cool_mask_threshold` (pure CGM) do not cool at all. Only wind or
mixed gas cools through Λ(T).

## 4. Input files

### 4.1 `fast_wind.athinput`: adiabatic fast wind, subsonic BH

| | |
|---|---|
| BH | M = 2×10⁷ Msun, v_bh = 100 km/s, ε = 0.5 pc |
| CGM | n = 1 cm⁻³, T = 10⁶ K (c_s = 151 km/s, Mach 0.66: no CGM bow shock) |
| Wind | Mdot = 10⁻⁴ Msun/yr, v_w = 0.1c, T_w = 10⁴ K, r_inj = 2.5 pc |
| Scales | R_0 ≈ 26 pc (40 pc ram only), R_0/r_inj ≈ 10 |
| Mesh | 320×160×160 pc at 1 pc root, level 1 (0.5 pc) in the inner 96 pc cube; 439 blocks of 32³ |
| Cooling | Explicit `ism_cooling`, CGM masked (§3.2); `wind_cool` off |
| Run | tlim = 1000 (≈ 1 Myr), RK2, CFL 0.3, PLM + HLLC, FOFC on, T floor 10⁴ K |
| Precision | Double required |

The shocked wind is at ~10¹⁰ K and n ~ 2×10⁻⁵ cm⁻³, so it is effectively adiabatic. This
is the energy-driven reference.

### 4.2 `capture_cool.athinput`: momentum-driven wind, supersonic BH

This run tests whether the AGN wind momentum can halt and capture ambient gas out to
R_cap = 2 R_0.

| | |
|---|---|
| BH | M = 2×10⁷ Msun, v_bh = 1000 km/s (Mach 6.6), ε = 1 pc |
| CGM | n = 1 cm⁻³, T = 10⁶ K |
| Wind | ṗ = 3.5×10³³ dyn → v_w = 0.03c (8994 km/s), Mdot = 0.0614 Msun/yr, T_w = 10⁴ K, r_inj = 10 pc |
| Scales | R_0 = 54 pc, R_cap = 109 pc, R_0/r_inj = 5.4; shocked CGM ~1.4×10⁷ K |
| Wind cooling | `wind_cool = true`, `T_wind_cool = 1e5`, `wind_cool_fmin = 0.5` |
| CGM thermodynamics | `heat_balance = true`, `T_heat_max = 3e6`, sub-cycled cooling (`cool_cfl = 0.1`, `cool_nsub_max = 1000`) so a dense cooled shell does not crush the hydro dt. No cooling mask, so wind gas also goes through `ism_cooling` |
| Tracers | `nscalars = 2` (wind fraction + impact parameter) |
| Diagnostics | `user_hist = true`, `r_diag = 109` pc (= R_cap), history every ~1 kyr |
| Mesh | 2000×1000×1000 pc; level 0 at 7.81 pc, level 1 at 3.91 pc, level 2 at 1.95 pc around the head (R_0 ≈ 27 cells, r_inj ≈ 5 cells); 380 blocks, 12.5M cells |
| Run | tlim = 2000 (1.96 Myr, one box crossing), ~3.2×10⁴ cycles, ~2 h in single precision on an RTX 4000 Ada |

Predicted captured-gas rate (the capture model): Mdot_capt = ṗ/(v_bh + v_w,slow) ≈
0.46 Msun/yr for v_w,slow = 200 km/s. Compare this against `M_halt` and `Mdot_up`.

### 4.3 `capture_cool_n10.athinput`: higher-density variant

This is identical to `capture_cool` except for:
- `n_cgm = 10`;
- `Mdot = 0.614` Msun/yr, so ṗ = 3.5×10³⁴ dyn, which keeps R_0 = 54 pc and the same mesh.
  This corresponds to L ~ 10⁴⁵ erg/s, ~40% of Eddington for 2×10⁷ Msun.
- `basename = capture_cool_n10`.

At n = 10 the shocked CGM (~1.7×10⁷ K, n ~ 35) has t_cool ~ 0.17 Myr, comparable to the
flow time, compared with 1.7 Myr at n = 1. This run tests whether the shocked CGM itself
becomes radiative. A fully cooled 10⁴ K shell in pressure balance with ρv_bh² = 10⁻⁷
dyn cm⁻² would have n ~ 7×10⁴ cm⁻³: unresolved and thin-shell unstable, so read its
structure with care.

Predicted capture rate: ≈ 4.6 Msun/yr.

## 5. Parameter reference (`<problem>`)

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `M_bh` | required | Msun | BH mass |
| `epsilon` | required | code length | gravitational softening |
| `v_bh` | required | km/s | BH speed (CGM inflow speed) |
| `n_cgm`, `T_cgm` | required | cm⁻³, K | ambient CGM |
| `Mdot`, `v_w` | required | Msun/yr, km/s | wind mass-loss rate and speed |
| `T_w` | 1e4 | K | temperature of the injected wind |
| `r_inj` | required | code length | wind reset radius |
| `user_srcs` | — | — | must be `true` (gravity + wind cooling + reset) |
| `wind_cool` | false | — | momentum-driven wind cooling (§2.4) |
| `T_wind_cool` | 1e5 | K | temperature the shocked wind is reset to |
| `wind_cool_fmin` | 0.5 | — | minimum wind fraction for wind cooling |
| `heat_balance` | false | — | density-matched CGM heating (§3.1); needs `ism_cooling` |
| `T_heat_max` | 3e6 | K | no heating above this T |
| `r_diag` | 2 R_0 | code length | radius for the history diagnostics |
| `user_hist` | — | — | enables `CaptureHistory()` output |

`<hydro> nscalars` must be ≥ 1. Use 2 for the impact-parameter tracer.

## 6. Caveat: physical motivation for `wind_cool`

The instantaneous wind cooling is a **limiting case**, not a physical model. Using the
cooling rates of Faucher-Giguère & Quataert (2012, MNRAS 425, 605), the shocked wind in
`capture_cool` (n_sw ≈ 0.03 cm⁻³, T ≈ 10⁹ K at R_0 = 54 pc) has:

| Process | Cooling time |
|---|---|
| Inverse Compton (radiation energy density U_ph ≈ ρv_bh² when ṗ = L/c) | t_IC ≈ 3mₑc²/(4σ_T c ρ v_bh²) ≈ 100 Myr (10 Myr for n10) |
| Free–free | ~5 Gyr |
| Residence of shocked wind near the head (≈ 3R_0/c_s) | ~30 kyr |

So the shocked 0.03c wind is effectively adiabatic in this geometry. The `capture_cool`
runs should be read as the **momentum-driven lower bound**, with `fast_wind`-style
adiabatic runs as the energy-driven bracket.

Two consequences for the setup:
- Inverse Compton cooling cannot bring gas below the Compton temperature T_X ≈ 2×10⁷ K.
  If the cooling is attributed to Compton scattering, `T_wind_cool ≈ 2e7` is the
  consistent choice. The wind's thermal pressure is then still only ~2% of ρv_bh², so the
  run stays effectively momentum-driven.
- A self-consistently radiative shocked wind needs a slow wind at the same ṗ (paper
  §2.3: free–free and line cooling dominate for v_in ~ 1000 km/s). Rough estimates give
  t_cool < t_res for v_w ≲ 700 km/s at n = 1, and v_w ≲ 1000 km/s at n = 10. In that case
  `wind_cool` can be dropped and the normal cooling allowed to act on wind gas. The cost
  is that Mdot_w becomes comparable to the predicted capture rate.
