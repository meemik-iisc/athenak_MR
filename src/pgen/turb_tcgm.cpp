//========================================================================================
// AthenaK astrophysical MHD code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_tcgm.cpp
//! \brief Runaway BH moving through a time-dependent, turbulent CGM (3D Cartesian).
//!
//! 3D Cartesian port of Idefix problems/turb_tcgm (2D cylindrical). BH frame: the BH sits
//! at the origin and the CGM flows in through the outer-x1 face at -v_bh x_hat.
//!
//!  - CGM:   n(t) = n_cgm0 exp(-t/dt_ncgm) at fixed T_cgm, imposed in the outer-x1 ghosts.
//!  - Wind:  mass, momentum and energy injected uniformly in r < r_inj, radial speed
//!           v_inj = v_w/sqrt(2), with Mdot(t) = Mdot0 exp(-t/dt_mdot) and v_w = vw0.
//!           Passive scalar 0 is the wind mass fraction (1 in the wind, 0 in the CGM);
//!           the injected mass is added to it, so CGM crossing r < r_inj stays untagged.
//!  - BH:    softened point-mass gravity, G M_bh / (r^2 + eps^2)^{3/2}.
//!  - Turb:  frozen, solenoidal velocity field made of random plane waves,
//!               v_t(x) = sum_n a_n e_n cos(k_n . x' + phi_n),   e_n . k_n = 0,
//!           evaluated at x' = (x + v_bh t, y, z), so the pattern is carried in with the
//!           CGM. It replaces the Bessel(R) x Fourier(z) modes of the cylindrical version.
//!           |k_n| is log-spaced in [2pi/lmax, 2pi/lmin], directions and polarisations are
//!           isotropic, and a_n ~ k_n^{-1/3} gives E(k) ~ k^{-5/3} on the log-spaced shells.
//!           Normalised so <|v_t|^2> = sum a_n^2 / 2 = sigv^2 (3D rms = sigv).
//!  - Cooling: <hydro_srcterms> ism_cooling with cooling_subcycle = true, masked by
//!           passive scalar 0 (no cooling in pure CGM), as in the Idefix [Cooling] block.
//!
//! Input units: M_bh [Msun], Mdot0 [Msun/yr], n_cgm0 [cm^-3], T_cgm [K], sigv [km/s];
//! lengths, times and v_bh, vw0 in code units (kpc, kpc/(1000 km/s), 1000 km/s).

#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "coordinates/cell_locations.hpp"
#include "units/units.hpp"
#include "pgen.hpp"

namespace {

struct TurbTCGM {
  Real gm_bh, eps;           // G*M_bh and softening length (code units)
  Real v_bh;                 // BH speed; CGM flows in at -v_bh x_hat
  Real n_cgm0, dt_ncgm;      // n_cgm(t) = n_cgm0 exp(-t/dt_ncgm)   [cm^-3]
  Real rho_per_n;            // code density per cm^-3 (mu*amu/rho_unit)
  Real t_cgm_code;           // T_cgm / T_unit, so P = rho * t_cgm_code
  Real mdot0, dt_mdot;       // Mdot(t) = Mdot0 exp(-t/dt_mdot)    (code units)
  Real vw0;                  // wind speed (code units)
  Real r_inj, vol_inj;       // injection radius and analytic sphere volume
  Real gamma;
};
TurbTCGM tc;

// Plane-wave modes, one row per mode: kx, ky, kz, ax, ay, az, phase (a = a_n e_n).
// Held by pointer and freed in the finalize hook, before Kokkos::finalize().
constexpr int kNModeCols = 7;
struct TurbModes {
  int nmodes = 0;
  DvceArray2D<Real> modes;
};
TurbModes *turb = nullptr;

void AddUserSrcs(Mesh *pm, const Real bdt);
void InflowX1BC(Mesh *pm);
void FreeTurb(ParameterInput *pin, Mesh *pm);
void BuildTurb(Real sigv, Real lmin, Real lmax, int nmodes, int seed);

KOKKOS_INLINE_FUNCTION
Real NCGM(const Real t, const Real n0, const Real dt_n) { return n0*exp(-t/dt_n); }

KOKKOS_INLINE_FUNCTION
Real MdotT(const Real t, const Real mdot0, const Real dt_mdot) {
  return mdot0*exp(-t/dt_mdot);
}

// Turbulent velocity at pattern coordinates (xp, y, z)
KOKKOS_INLINE_FUNCTION
void TurbVel(const DvceArray2D<Real> &md, const int nm, const Real xp, const Real y,
             const Real z, Real &vx, Real &vy, Real &vz) {
  vx = 0.0; vy = 0.0; vz = 0.0;
  for (int n = 0; n < nm; ++n) {
    Real c = cos(md(n,0)*xp + md(n,1)*y + md(n,2)*z + md(n,6));
    vx += md(n,3)*c;
    vy += md(n,4)*c;
    vz += md(n,5)*c;
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_tcgm requires a <hydro> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->phydro->nscalars < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_tcgm needs <hydro>/nscalars >= 1 (scalar 0 = wind tracer)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  user_srcs_func = AddUserSrcs;
  user_bcs_func  = InflowX1BC;
  pgen_final_func = FreeTurb;

  // --- parameters -> code units ---
  auto *pu = pmbp->punit;
  Real m_bh  = pin->GetReal("problem", "M_bh");     // Msun
  Real mdot0 = pin->GetReal("problem", "Mdot0");    // Msun/yr
  Real sigv  = pin->GetReal("problem", "sigv");     // km/s
  Real t_cgm = pin->GetReal("problem", "T_cgm");    // K

  tc.gamma      = pin->GetReal("hydro", "gamma");
  tc.gm_bh      = pu->grav_constant()*m_bh*pu->msun();
  tc.eps        = pin->GetReal("problem", "epsilon");
  tc.v_bh       = pin->GetReal("problem", "v_bh");
  tc.n_cgm0     = pin->GetReal("problem", "n_cgm0");
  tc.dt_ncgm    = pin->GetReal("problem", "dt_ncgm");
  tc.rho_per_n  = pu->mu()*pu->atomic_mass_unit_cgs/pu->density_cgs();
  tc.t_cgm_code = t_cgm/pu->temperature_cgs();
  tc.mdot0      = mdot0*pu->msun()/pu->yr();
  tc.dt_mdot    = pin->GetReal("problem", "dt_mdot");
  tc.vw0        = pin->GetReal("problem", "vw0");
  tc.r_inj      = pin->GetReal("problem", "r_inj");
  tc.vol_inj    = 4.0*M_PI*tc.r_inj*tc.r_inj*tc.r_inj/3.0;

  Real lmin   = pin->GetReal("problem", "turb_lmin");
  Real lmax   = pin->GetReal("problem", "turb_lmax");
  int nmodes  = pin->GetOrAddInteger("problem", "turb_nmodes", 1024);
  int seed    = pin->GetOrAddInteger("problem", "turb_seed", 1234);
  // Modes are needed by the inflow BC, so build them on restarts too.
  BuildTurb(sigv*pu->km_s(), lmin, lmax, nmodes, seed);

  if (global_variable::my_rank == 0) {
    std::cout << "turb_tcgm: G*M_bh = " << tc.gm_bh << ", Mdot0 = " << tc.mdot0
              << ", rho_cgm0 = " << tc.n_cgm0*tc.rho_per_n
              << ", T_cgm[code] = " << tc.t_cgm_code
              << ", sigv[code] = " << sigv*pu->km_s() << std::endl;
  }
  if (restart) return;

  // --- initial conditions: CGM at -v_bh + turbulence, wind in r < r_inj ---
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  const int nhyd = pmbp->phydro->nhydro;
  const int nscal = pmbp->phydro->nscalars;
  const Real gm1 = tc.gamma - 1.0;

  const Real rho_cgm = NCGM(0.0, tc.n_cgm0, tc.dt_ncgm)*tc.rho_per_n;
  const Real p_cgm   = rho_cgm*tc.t_cgm_code;
  const Real v_bh    = tc.v_bh;
  const Real v_inj   = tc.vw0/std::sqrt(2.0);
  const Real rho_w   = MdotT(0.0, tc.mdot0, tc.dt_mdot)/(4.0*M_PI*SQR(tc.r_inj)*v_inj);
  const Real p_w     = rho_w*SQR(v_inj)/tc.gamma;
  const Real r_inj   = tc.r_inj;
  auto md = turb->modes;
  const int nm = turb->nmodes;

  par_for("turb_tcgm_ic", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real rad = sqrt(x*x + y*y + z*z);

    Real rho, vx, vy, vz, p, s;
    if (rad < r_inj) {
      Real rr = fmax(rad, 1.0e-20);
      rho = rho_w;  p = p_w;  s = 1.0;
      vx = v_inj*x/rr;  vy = v_inj*y/rr;  vz = v_inj*z/rr;
    } else {
      rho = rho_cgm;  p = p_cgm;  s = 0.0;
      TurbVel(md, nm, x, y, z, vx, vy, vz);
      vx -= v_bh;
    }
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = rho*vx;
    u0(m,IM2,k,j,i) = rho*vy;
    u0(m,IM3,k,j,i) = rho*vz;
    u0(m,IEN,k,j,i) = p/gm1 + 0.5*rho*(vx*vx + vy*vy + vz*vz);
    u0(m,nhyd,k,j,i) = rho*s;
    for (int n = nhyd+1; n < nhyd+nscal; ++n) u0(m,n,k,j,i) = 0.0;
  });
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void BuildTurb()
//! \brief Draw the plane-wave modes on the host (same seed -> same modes on every rank)
//! and copy them to the device.

void BuildTurb(Real sigv, Real lmin, Real lmax, int nmodes, int seed) {
  if (nmodes < 1 || lmin <= 0.0 || lmax <= lmin) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "turb_tcgm needs turb_nmodes >= 1 and 0 < turb_lmin < turb_lmax"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  const double kmin = 2.0*M_PI/lmax, kmax = 2.0*M_PI/lmin;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  std::vector<double> h(static_cast<size_t>(nmodes)*kNModeCols);
  double sum_a2 = 0.0;
  for (int n = 0; n < nmodes; ++n) {
    double kmag = kmin*std::pow(kmax/kmin, (n + 0.5)/nmodes);
    // isotropic direction
    double mu  = 2.0*unif(gen) - 1.0;
    double phk = 2.0*M_PI*unif(gen);
    double st  = std::sqrt(1.0 - mu*mu);
    double kh[3] = {st*std::cos(phk), st*std::sin(phk), mu};
    // orthonormal basis (e1, e2) of the plane perpendicular to k
    double ref[3] = {0.0, 0.0, 1.0};
    if (std::fabs(kh[2]) > 0.9) { ref[0] = 1.0; ref[2] = 0.0; }
    double e1[3] = {kh[1]*ref[2] - kh[2]*ref[1],
                    kh[2]*ref[0] - kh[0]*ref[2],
                    kh[0]*ref[1] - kh[1]*ref[0]};
    double e1n = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
    for (double &c : e1) c /= e1n;
    double e2[3] = {kh[1]*e1[2] - kh[2]*e1[1],
                    kh[2]*e1[0] - kh[0]*e1[2],
                    kh[0]*e1[1] - kh[1]*e1[0]};
    double psi   = 2.0*M_PI*unif(gen);   // polarisation angle
    double phase = 2.0*M_PI*unif(gen);
    double a     = std::pow(kmag, -1.0/3.0);
    sum_a2 += a*a;

    double *row = &h[static_cast<size_t>(n)*kNModeCols];
    for (int d = 0; d < 3; ++d) {
      row[d]     = kmag*kh[d];
      row[3 + d] = a*(std::cos(psi)*e1[d] + std::sin(psi)*e2[d]);
    }
    row[6] = phase;
  }
  const double norm = sigv/std::sqrt(0.5*sum_a2);   // <|v|^2> = sum a^2/2 = sigv^2

  if (turb == nullptr) turb = new TurbModes;
  turb->nmodes = nmodes;
  turb->modes = DvceArray2D<Real>("turb_modes", nmodes, kNModeCols);
  auto hm = Kokkos::create_mirror_view(turb->modes);
  for (int n = 0; n < nmodes; ++n) {
    for (int c = 0; c < kNModeCols; ++c) {
      double v = h[static_cast<size_t>(n)*kNModeCols + c];
      if (c >= 3 && c < 6) v *= norm;
      hm(n,c) = static_cast<Real>(v);
    }
  }
  Kokkos::deep_copy(turb->modes, hm);

  if (global_variable::my_rank == 0) {
    std::cout << "turb_tcgm: " << nmodes << " plane-wave modes, k in [" << kmin << ", "
              << kmax << "], sigv[code] = " << sigv << std::endl;
  }
}

void FreeTurb(ParameterInput *pin, Mesh *pm) {
  delete turb;
  turb = nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn void AddUserSrcs()
//! \brief BH gravity and wind injection. Source terms use primitives w0, update u0.

void AddUserSrcs(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const auto &w0 = pmbp->phydro->w0;
  auto &u0 = pmbp->phydro->u0;
  const int nhyd = pmbp->phydro->nhydro;

  const Real gm_bh = tc.gm_bh;
  const Real eps2  = tc.eps*tc.eps;
  const Real r_inj = tc.r_inj;
  const Real gm1   = tc.gamma - 1.0;
  const Real time  = pm->time;
  const Real v_inj = tc.vw0/std::sqrt(2.0);
  const Real s_rho = MdotT(time, tc.mdot0, tc.dt_mdot)/tc.vol_inj;
  // kinetic + thermal energy of the injected gas, with c_s = v_inj
  const Real s_e   = s_rho*SQR(v_inj)*(0.5 + 1.0/(tc.gamma*gm1));

  par_for("turb_tcgm_srcs", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real r2 = x*x + y*y + z*z;

    // softened point-mass gravity
    Real rho = w0(m,IDN,k,j,i);
    Real g_over_r = gm_bh/((r2 + eps2)*sqrt(r2 + eps2));
    Real vdotr = w0(m,IVX,k,j,i)*x + w0(m,IVY,k,j,i)*y + w0(m,IVZ,k,j,i)*z;
    u0(m,IM1,k,j,i) -= bdt*rho*g_over_r*x;
    u0(m,IM2,k,j,i) -= bdt*rho*g_over_r*y;
    u0(m,IM3,k,j,i) -= bdt*rho*g_over_r*z;
    u0(m,IEN,k,j,i) -= bdt*rho*g_over_r*vdotr;

    // wind injection; only the injected mass is tagged, so scalar 0 stays the true wind
    // mass fraction (CGM passing through r < r_inj is not relabelled as wind)
    Real rad = sqrt(r2);
    if (rad < r_inj) {
      Real rr = fmax(rad, 1.0e-20);
      u0(m,IDN,k,j,i) += bdt*s_rho;
      u0(m,IM1,k,j,i) += bdt*s_rho*v_inj*x/rr;
      u0(m,IM2,k,j,i) += bdt*s_rho*v_inj*y/rr;
      u0(m,IM3,k,j,i) += bdt*s_rho*v_inj*z/rr;
      u0(m,IEN,k,j,i) += bdt*s_e;
      u0(m,nhyd,k,j,i) += bdt*s_rho;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void InflowX1BC()
//! \brief Outer-x1 ghosts: CGM at n_cgm(t), T_cgm, velocity -v_bh + turbulence at
//! x' = x + v_bh t, tracer 0. Fills all j,k (incl. ghosts) of the outer-x1 ghost slab.
//! Sets w0 and u0 directly (ideal gas), only on blocks whose outer-x1 face is user.

void InflowX1BC(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const int n2 = (nx2 > 1) ? (nx2 + 2*ng) : 1;
  const int n3 = (nx3 > 1) ? (nx3 + 2*ng) : 1;
  auto &size = pmbp->pmb->mb_size;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &u0 = pmbp->phydro->u0;
  auto &w0 = pmbp->phydro->w0;
  const int nhyd = pmbp->phydro->nhydro;
  const int nscal = pmbp->phydro->nscalars;

  const Real time  = pm->time;
  const Real rho   = NCGM(time, tc.n_cgm0, tc.dt_ncgm)*tc.rho_per_n;
  const Real eint  = rho*tc.t_cgm_code/(tc.gamma - 1.0);
  const Real v_bh  = tc.v_bh;
  const Real shift = tc.v_bh*time;   // gas moves at -v_bh: frozen pattern f(x + v_bh t)
  auto md = turb->modes;
  const int nm = turb->nmodes;

  par_for("turb_tcgm_ox1", DevExeSpace(), 0, pmbp->nmb_thispack-1, 0, n3-1, 0, n2-1,
          0, ng-1,
  KOKKOS_LAMBDA(int m, int k, int j, int ii) {
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) != BoundaryFlag::user) return;
    const int i = ie + 1 + ii;
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real vx, vy, vz;
    TurbVel(md, nm, x + shift, y, z, vx, vy, vz);
    vx -= v_bh;

    w0(m,IDN,k,j,i) = rho;
    w0(m,IVX,k,j,i) = vx;
    w0(m,IVY,k,j,i) = vy;
    w0(m,IVZ,k,j,i) = vz;
    w0(m,IEN,k,j,i) = eint;
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = rho*vx;
    u0(m,IM2,k,j,i) = rho*vy;
    u0(m,IM3,k,j,i) = rho*vz;
    u0(m,IEN,k,j,i) = eint + 0.5*rho*(vx*vx + vy*vy + vz*vz);
    for (int n = nhyd; n < nhyd+nscal; ++n) {
      w0(m,n,k,j,i) = 0.0;
      u0(m,n,k,j,i) = 0.0;
    }
  });
}

}  // namespace
