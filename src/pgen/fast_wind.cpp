//========================================================================================
// AthenaK astrophysical MHD code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fast_wind.cpp
//! \brief Runaway BH with a fast, light wind moving through a uniform CGM (3D Cartesian).
//!
//! BH frame: the BH sits at the origin and a uniform CGM flows in through the outer-x1
//! face at -v_bh x_hat.
//!
//!  - CGM:     constant n_cgm, T_cgm, imposed in the outer-x1 ghosts and as the initial
//!             condition everywhere (tracer = 0).
//!  - Wind:    hard reset every RK stage inside r < r_inj to the free-wind solution
//!             rho = Mdot/(4 pi r_eff^2 v_w), v = v_w r_hat, T = T_w, tracer = 1, with
//!             r_eff = max(r, r_inj/4) to avoid the 1/r^2 divergence at the center.
//!             The mass flux through r_inj is therefore exactly Mdot.
//!  - BH:      softened point-mass gravity G M_bh / (r^2 + eps^2)^{3/2} outside r_inj.
//!  - Cooling: <hydro_srcterms> ism_cooling. Two ways to keep the ambient CGM in
//!             thermal equilibrium:
//!             (a) heat_balance = true: density-matched heating, per particle
//!                 Gamma = n_cgm Lambda(T), for T < T_heat_max (default 3e6 K), applied
//!                 inside the cooling kernels (SourceTerms::heat_ref_rho/heat_tmax,
//!                 same Lambda table; hrate is set to 0). Net cooling is then
//!                 n (n - n_cgm) Lambda(T): exactly zero in the ambient medium at any T
//!                 (a constant Gamma = n_cgm Lambda(T_cgm) makes the 1e6 K ambient
//!                 thermally unstable, e-folding in ~t_cool), compressed gas cools,
//!                 rarefied gas below T_heat_max is heated, and gas above T_heat_max
//!                 (e.g. shocked CGM) cools at the full n^2 Lambda(T). This law is
//!                 thermally unstable at the grid scale wherever t_cool << dx/c_s; with
//!                 heat_n_min = f >= 1, gas with n <= f n_cgm below T_heat_max neither
//!                 cools nor heats, so only clearly compressed gas cools;
//!             (b) cool_mask_scalar = 0: gas with wind tracer < threshold never cools.
//!  - Wind cooling (wind_cool = true): every stage, cells with wind fraction
//!             f_w > wind_cool_fmin and T > T_wind_cool are reset to T_wind_cool
//!             (kinetic energy kept). The shocked wind then has no pressure, so the
//!             wind acts on the CGM through its momentum only (momentum-driven limit,
//!             e.g. efficient Compton cooling).
//!  - Tracers: scalar 0 = wind mass fraction; scalar 1 (if nscalars >= 2) = impact
//!             parameter b = sqrt(y^2+z^2) of the inflowing CGM (code length; 0 for
//!             wind), to find where captured gas came from.
//!  - History (user_hist = true), CGM gas = wind fraction < 0.5:
//!             M_halt      CGM mass within r < r_diag with v_x > -v_bh/2 [Msun]
//!                         (halted: dragged along with the BH)
//!             Mdot_up     CGM mass flux outward (v_r > 0) through the upstream
//!                         hemisphere (x > 0) of the sphere r = r_diag [Msun/yr];
//!                         zero for undisturbed inflow
//!             Mh_b<i>     M_halt in bins of the impact parameter b/R_0:
//!                         [0,.5) [.5,1) [1,1.5) [1.5,2) [2,2.5) [2.5,3) [3,inf)
//!
//! Wind ram pressure balances the CGM total pressure at the stand-off radius
//!   R_0 = sqrt( Mdot v_w / (4 pi (rho v_bh^2 + P_cgm)) ),
//! which is printed at startup.
//!
//! Input units: M_bh [Msun], Mdot [Msun/yr], v_w and v_bh [km/s], n_cgm [cm^-3],
//! T_cgm and T_w [K]; epsilon and r_inj in code length units. Needs nscalars >= 1.
//!
//! Build in DOUBLE precision (make_build_file.sh fast_wind): in the free wind the
//! kinetic energy exceeds the thermal energy by ~1e6, so float loses the pressure.

#include <cmath>
#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "coordinates/cell_locations.hpp"
#include "units/units.hpp"
#include "outputs/outputs.hpp"
#include "srcterms/srcterms.hpp"
#include "srcterms/ismcooling.hpp"
#include "pgen.hpp"

namespace {

struct FastWind {
  Real gm_bh, eps;           // G*M_bh and softening length (code units)
  Real v_bh;                 // BH speed; CGM flows in at -v_bh x_hat
  Real rho_cgm, p_cgm;       // CGM density and pressure
  Real gamma;
  Real mdot, v_w, t_w;       // wind mass-loss rate, speed, temperature (code units)
  Real r_inj;                // injection radius
  bool wind_cool;            // reset shocked wind gas to t_wcool
  Real t_wcool, fw_min;      // wind cooling temperature (code) and min. wind fraction
  bool b_tracer;             // scalar 1 = impact parameter
  Real r0, r_diag;           // stand-off radius (ram+thermal) and diagnostic radius
  double msun_per_mass, msun_yr_per_mdot;
};
FastWind fw;

void AddUserSrcs(Mesh *pm, const Real bdt);
void ResetWind(Mesh *pm);
void CoolWind(Mesh *pm);
void InflowX1BC(Mesh *pm);
void CaptureHistory(HistoryData *pdata, Mesh *pm);

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->phydro->nscalars < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "fast_wind requires <hydro> with nscalars >= 1 (wind tracer)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  user_srcs_func = AddUserSrcs;
  user_bcs_func  = InflowX1BC;
  user_hist_func = CaptureHistory;

  // --- parameters -> code units (host math in double, cast once) ---
  auto *pu = pmbp->punit;
  double m_bh  = pin->GetReal("problem", "M_bh");     // Msun
  double n_cgm = pin->GetReal("problem", "n_cgm");    // cm^-3
  double t_cgm = pin->GetReal("problem", "T_cgm");    // K
  double v_bh  = pin->GetReal("problem", "v_bh");     // km/s
  double mdot  = pin->GetReal("problem", "Mdot");     // Msun/yr
  double v_w   = pin->GetReal("problem", "v_w");      // km/s
  double t_w   = pin->GetOrAddReal("problem", "T_w", 1.0e4);  // K

  double rho = n_cgm*pu->mu()*pu->atomic_mass_unit_cgs/pu->density_cgs();
  fw.gamma   = pin->GetReal("hydro", "gamma");
  fw.gm_bh   = static_cast<Real>(pu->grav_constant()*m_bh*pu->msun());
  fw.eps     = pin->GetReal("problem", "epsilon");
  fw.v_bh    = static_cast<Real>(v_bh*pu->km_s());
  fw.rho_cgm = static_cast<Real>(rho);
  fw.p_cgm   = static_cast<Real>(rho*t_cgm/pu->temperature_cgs());
  fw.mdot    = static_cast<Real>(mdot*pu->msun()/pu->yr());
  fw.v_w     = static_cast<Real>(v_w*pu->km_s());
  fw.t_w     = static_cast<Real>(t_w/pu->temperature_cgs());
  fw.r_inj   = pin->GetReal("problem", "r_inj");
  fw.wind_cool = pin->GetOrAddBoolean("problem", "wind_cool", false);
  fw.t_wcool = static_cast<Real>(pin->GetOrAddReal("problem", "T_wind_cool", 1.0e5)
                                 /pu->temperature_cgs());
  fw.fw_min  = pin->GetOrAddReal("problem", "wind_cool_fmin", 0.5);
  fw.b_tracer = (pmbp->phydro->nscalars >= 2);
  fw.msun_per_mass = 1.0/pu->msun();
  fw.msun_yr_per_mdot = pu->yr()/pu->msun();
  {
    double ptot = rho*fw.v_bh*fw.v_bh + fw.p_cgm;
    fw.r0 = static_cast<Real>(std::sqrt(fw.mdot*fw.v_w/(4.0*M_PI*ptot)));
  }
  fw.r_diag  = pin->GetOrAddReal("problem", "r_diag", 2.0*fw.r0);

  // density-matched heating of the ambient CGM, n_cgm Lambda(T) per particle below
  // T_heat_max, done inside the cooling kernels (see SourceTerms::heat_ref_rho)
  bool heat_balance = pin->GetOrAddBoolean("problem", "heat_balance", false);
  double t_heat_max = pin->GetOrAddReal("problem", "T_heat_max", 3.0e6);  // K
  // below T_heat_max, gas with n <= heat_n_min*n_cgm neither cools nor heats (0 keeps the
  // pure density-matched law; >= 1 also stops the heating of rarefied gas)
  double heat_n_min = pin->GetOrAddReal("problem", "heat_n_min", 0.0);
  if (heat_balance) {
    if (pmbp->phydro->psrc == nullptr || !pmbp->phydro->psrc->ism_cooling) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "heat_balance = true needs ism_cooling = true" << std::endl;
      exit(EXIT_FAILURE);
    }
    pmbp->phydro->psrc->hrate = 0.0;
    pmbp->phydro->psrc->heat_ref_rho = fw.rho_cgm;
    pmbp->phydro->psrc->heat_rho_min = static_cast<Real>(heat_n_min)*fw.rho_cgm;
    pmbp->phydro->psrc->heat_tmax = static_cast<Real>(t_heat_max);
  }

  if (global_variable::my_rank == 0) {
    double ptot = rho*fw.v_bh*fw.v_bh + fw.p_cgm;
    double r0 = std::sqrt(fw.mdot*fw.v_w/(4.0*M_PI*ptot));
    double r0_ram = std::sqrt(fw.mdot*fw.v_w/(4.0*M_PI*rho*fw.v_bh*fw.v_bh));
    double cs = std::sqrt(fw.gamma*fw.p_cgm/rho);
    std::cout << "fast_wind (code units): rho_cgm = " << fw.rho_cgm
              << ", p_cgm = " << fw.p_cgm << ", v_bh = " << fw.v_bh
              << " (Mach " << fw.v_bh/cs << ")" << std::endl
              << "  Mdot = " << fw.mdot << ", v_w = " << fw.v_w
              << ", rho_w(r_inj) = " << fw.mdot/(4.0*M_PI*fw.r_inj*fw.r_inj*fw.v_w)
              << ", r_inj = " << fw.r_inj << std::endl
              << "  R_0 (ram+thermal) = " << r0 << ", R_0 (ram only) = " << r0_ram
              << ", R_0/r_inj = " << r0/fw.r_inj << std::endl
              << "  pdot_w = Mdot v_w = " << fw.mdot*fw.v_w << " (code), "
              << fw.mdot*fw.v_w*pu->mass_cgs()*pu->length_cgs()/SQR(pu->time_cgs())
              << " dyn; R_cap = 2 R_0(ram) = " << 2.0*r0_ram << std::endl
              << "  wind_cool = " << fw.wind_cool << " (T_wind_cool[code] = "
              << fw.t_wcool << ", fmin = " << fw.fw_min << "), b tracer = "
              << fw.b_tracer << ", r_diag = " << fw.r_diag;
    if (heat_balance) {
      std::cout << std::endl << "  heating n_cgm Lambda(T) per particle (heat_ref_rho = "
                << fw.rho_cgm << ") for T < " << t_heat_max << " K; hrate = 0"
                << std::endl << "  no net cooling/heating for n <= " << heat_n_min
                << " n_cgm and T < T_heat_max";
    }
    std::cout << std::endl;
  }
  if (restart) return;

  // --- initial conditions: uniform CGM moving at -v_bh, then the wind sphere ---
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &u0 = pmbp->phydro->u0;
  const int nhyd = pmbp->phydro->nhydro;
  const int nscal = pmbp->phydro->nscalars;
  const Real rho_c = fw.rho_cgm;
  const Real eint  = fw.p_cgm/(fw.gamma - 1.0);
  const Real vx    = -fw.v_bh;
  const bool btr   = fw.b_tracer;
  auto &size = pmbp->pmb->mb_size;
  const int nx2 = indcs.nx2, nx3 = indcs.nx3;

  par_for("fast_wind_ic", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = rho_c;
    u0(m,IM1,k,j,i) = rho_c*vx;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = eint + 0.5*rho_c*vx*vx;
    for (int n = nhyd; n < nhyd+nscal; ++n) u0(m,n,k,j,i) = 0.0;
    if (btr) {
      Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      u0(m,nhyd+1,k,j,i) = rho_c*sqrt(y*y + z*z);
    }
  });
  ResetWind(pmy_mesh_);
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void AddUserSrcs()
//! \brief Softened BH gravity outside r_inj (from primitives w0), then the wind reset.
//! User sources run last in HydroSrcTerms, so the reset has the final say each stage.

void AddUserSrcs(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const auto &w0 = pmbp->phydro->w0;
  auto &u0 = pmbp->phydro->u0;

  const Real gm_bh = fw.gm_bh;
  const Real eps2  = fw.eps*fw.eps;
  const Real ri2   = fw.r_inj*fw.r_inj;

  if (gm_bh > 0.0) {
    par_for("fast_wind_grav", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke, js, je,
            is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real r2 = x*x + y*y + z*z;
      if (r2 < ri2) return;  // overwritten by the wind reset anyway
      Real s2 = r2 + eps2;
      Real rho = w0(m,IDN,k,j,i);
      Real g_over_r = gm_bh/(s2*sqrt(s2));
      Real vdotr = w0(m,IVX,k,j,i)*x + w0(m,IVY,k,j,i)*y + w0(m,IVZ,k,j,i)*z;
      u0(m,IM1,k,j,i) -= bdt*rho*g_over_r*x;
      u0(m,IM2,k,j,i) -= bdt*rho*g_over_r*y;
      u0(m,IM3,k,j,i) -= bdt*rho*g_over_r*z;
      u0(m,IEN,k,j,i) -= bdt*rho*g_over_r*vdotr;
    });
  }

  if (fw.wind_cool) CoolWind(pm);
  ResetWind(pm);
}

//----------------------------------------------------------------------------------------
//! \fn void CoolWind()
//! \brief Momentum-driven limit: wind-dominated cells (f_w > fw_min) hotter than
//! t_wcool are reset to t_wcool, keeping density and momentum. Works on u0 (the state
//! after this stage's update), so the energy removed is exactly the excess.

void CoolWind(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &u0 = pmbp->phydro->u0;
  const int nhyd = pmbp->phydro->nhydro;
  const Real gm1  = fw.gamma - 1.0;
  const Real tw   = fw.t_wcool;
  const Real fmin = fw.fw_min;

  par_for("fast_wind_cool", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke, js, je,
          is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real d = u0(m,IDN,k,j,i);
    if (u0(m,nhyd,k,j,i) < fmin*d) return;
    Real ke_ = 0.5*(SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i))
                    + SQR(u0(m,IM3,k,j,i)))/d;
    Real e_t = d*tw/gm1;
    if (u0(m,IEN,k,j,i) - ke_ > e_t) u0(m,IEN,k,j,i) = ke_ + e_t;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void ResetWind()
//! \brief Set u0 inside r < r_inj to the free-wind solution (tracer = 1, all scalars).

void ResetWind(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  const int nhyd = pmbp->phydro->nhydro;
  const int nscal = pmbp->phydro->nscalars;

  const Real r_inj = fw.r_inj;
  const Real r_min = 0.25*fw.r_inj;
  const Real v_w   = fw.v_w;
  const Real flux  = fw.mdot/(4.0*M_PI*v_w);  // rho*r^2
  const Real t_w   = fw.t_w;
  const Real gm1   = fw.gamma - 1.0;

  par_for("fast_wind_reset", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke, js, je,
          is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real r = sqrt(x*x + y*y + z*z);
    if (r >= r_inj) return;
    Real re  = fmax(r, r_min);
    Real rho = flux/(re*re);
    Real rr  = fmax(r, static_cast<Real>(1.0e-20));
    Real vx = v_w*x/rr, vy = v_w*y/rr, vz = v_w*z/rr;
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = rho*vx;
    u0(m,IM2,k,j,i) = rho*vy;
    u0(m,IM3,k,j,i) = rho*vz;
    u0(m,IEN,k,j,i) = rho*t_w/gm1 + 0.5*rho*v_w*v_w;
    u0(m,nhyd,k,j,i) = rho;                                 // wind fraction 1
    for (int n = nhyd+1; n < nhyd+nscal; ++n) u0(m,n,k,j,i) = 0.0;  // b = 0
  });
}

//----------------------------------------------------------------------------------------
//! \fn void InflowX1BC()
//! \brief Outer-x1 ghosts: uniform CGM at n_cgm, T_cgm, velocity -v_bh x_hat, tracer 0.

void InflowX1BC(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int ie = indcs.ie;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &u0 = pmbp->phydro->u0;
  auto &w0 = pmbp->phydro->w0;
  const int nhyd = pmbp->phydro->nhydro;
  const int nscal = pmbp->phydro->nscalars;

  const Real rho  = fw.rho_cgm;
  const Real eint = fw.p_cgm/(fw.gamma - 1.0);
  const Real vx   = -fw.v_bh;
  const bool btr  = fw.b_tracer;
  auto &size = pmbp->pmb->mb_size;
  const int js = indcs.js, ks = indcs.ks, nx2 = indcs.nx2, nx3 = indcs.nx3;

  par_for("fast_wind_ox1", DevExeSpace(), 0, pmbp->nmb_thispack-1, 0, n3-1, 0, n2-1,
          0, ng-1,
  KOKKOS_LAMBDA(int m, int k, int j, int ii) {
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) != BoundaryFlag::user) return;
    const int i = ie + 1 + ii;
    w0(m,IDN,k,j,i) = rho;
    w0(m,IVX,k,j,i) = vx;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IEN,k,j,i) = eint;
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = rho*vx;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = eint + 0.5*rho*vx*vx;
    for (int n = nhyd; n < nhyd+nscal; ++n) {
      w0(m,n,k,j,i) = 0.0;
      u0(m,n,k,j,i) = 0.0;
    }
    if (btr) {
      // ghost cells too: CellCenterX extrapolates linearly outside the block
      Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real b = sqrt(y*y + z*z);
      w0(m,nhyd+1,k,j,i) = b;
      u0(m,nhyd+1,k,j,i) = rho*b;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void CaptureHistory()
//! \brief Captured-gas diagnostics (rank-local sums; the history output sums ranks).

void CaptureHistory(HistoryData *pdata, Mesh *pm) {
  constexpr int NB = 7;  // impact-parameter bins
  pdata->nhist = 2 + NB;
  pdata->label[0] = "M_halt";
  pdata->label[1] = "Mdot_up";
  const char *bl[NB] = {"Mh_b0.0", "Mh_b0.5", "Mh_b1.0", "Mh_b1.5", "Mh_b2.0",
                        "Mh_b2.5", "Mh_b3.0"};
  for (int n = 0; n < NB; ++n) pdata->label[2+n] = bl[n];

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  const int nmkji = pmbp->nmb_thispack*nkji;
  auto &size = pmbp->pmb->mb_size;
  const auto &w0 = pmbp->phydro->w0;
  const int nhyd = pmbp->phydro->nhydro;
  const bool btr = fw.b_tracer;
  const Real vhalt = -0.5*fw.v_bh;
  const Real rd = fw.r_diag;
  const Real r0 = fw.r0;

  // q = 0: M_halt, 1: Mdot_up, 2..: M_halt in b bin q-2
  for (int q = 0; q < 2 + NB; ++q) {
    if (q >= 2 && !btr) { pdata->hdata[q] = 0.0; continue; }
    double sum = 0.0;
    Kokkos::parallel_reduce("fast_wind_hist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, double &s) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      if (w0(m,nhyd,k,j,i) >= 0.5) return;  // wind gas
      Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real r = sqrt(x*x + y*y + z*z);
      Real dx = size.d_view(m).dx1;
      Real vol = dx*size.d_view(m).dx2*size.d_view(m).dx3;
      Real rho_c = w0(m,IDN,k,j,i)*(1.0 - w0(m,nhyd,k,j,i));  // CGM mass density
      if (q == 1) {
        if (x <= 0.0 || fabs(r - rd) >= 0.5*dx) return;
        Real vr = (w0(m,IVX,k,j,i)*x + w0(m,IVY,k,j,i)*y + w0(m,IVZ,k,j,i)*z)/r;
        if (vr > 0.0) s += static_cast<double>(rho_c*vr*vol/dx);
        return;
      }
      if (r >= rd || w0(m,IVX,k,j,i) <= vhalt) return;
      if (q >= 2) {
        Real bb = w0(m,nhyd+1,k,j,i)/r0;
        int ib = static_cast<int>(2.0*bb);
        ib = (ib > NB-1) ? NB-1 : ib;
        if (ib != q-2) return;
      }
      s += static_cast<double>(rho_c*vol);
    }, Kokkos::Sum<double>(sum));
    double f = (q == 1) ? fw.msun_yr_per_mdot : fw.msun_per_mass;
    pdata->hdata[q] = static_cast<Real>(sum*f);
  }
}

}  // namespace
