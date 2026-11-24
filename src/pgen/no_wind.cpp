//========================================================================================
// AthenaK astrophysical MHD code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file no_wind.cpp
//! \brief Runaway BH moving through a uniform CGM, no wind (3D Cartesian).
//!
//! BH frame: the BH sits at the origin and a uniform CGM flows in through the outer-x1
//! face at -v_bh x_hat.
//!
//!  - CGM:     constant n_cgm, T_cgm, imposed in the outer-x1 ghosts and as the initial
//!             condition everywhere.
//!  - BH:      softened point-mass gravity, G M_bh / (r^2 + eps^2)^{3/2}.
//!  - Sink:    cells with r < r_sink are reset to rho_sink whenever rho > rho_sink, by
//!             scaling all conserved variables by rho_sink/rho (velocity and temperature
//!             unchanged). The removed mass is accumulated and written to the history
//!             file (user_hist = true) as M_acc [code mass] and M_acc_msun [Msun];
//!             Mdot = d(M_acc)/dt. M_acc restarts from 0 after a restart.
//!  - Cooling: <hydro_srcterms> ism_cooling (explicit, optionally tabulated with
//!             cool_table = true). No heating, no wind injection, no passive scalars.
//!
//! Input units: M_bh [Msun], n_cgm [cm^-3], T_cgm [K], v_bh [km/s]; epsilon and r_sink
//! in code length units, rho_sink as a fraction of rho_cgm. Everything is converted with <units>, so the same pgen works for any
//! choice of code scales (e.g. kpc box and pc zoom-in box).
//!
//! Intended for single-precision builds (make_build_single.sh no_wind), but works in
//! double precision as well.

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
#include "pgen.hpp"

namespace {

struct NoWind {
  Real gm_bh, eps;           // G*M_bh and softening length (code units)
  Real v_bh;                 // BH speed (code units); CGM flows in at -v_bh x_hat
  Real rho_cgm;              // CGM density (code units)
  Real p_cgm;                // CGM pressure (code units)
  Real gamma;
  Real r_sink;               // sink radius (code units)
  Real rho_sink;             // density inside the sink after removal (code units)
  double m_acc = 0.0;        // rank-local accreted mass (code units), double on purpose
  double msun_per_mass;      // Msun per code mass unit
};
NoWind nw;

void AddUserSrcs(Mesh *pm, const Real bdt);
void InflowX1BC(Mesh *pm);
void NoWindHistory(HistoryData *pdata, Mesh *pm);

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "no_wind requires a <hydro> block" << std::endl;
    exit(EXIT_FAILURE);
  }

  user_srcs_func = AddUserSrcs;
  user_bcs_func  = InflowX1BC;
  user_hist_func = NoWindHistory;

  // --- parameters -> code units (host math in double, cast once) ---
  auto *pu = pmbp->punit;
  double m_bh  = pin->GetReal("problem", "M_bh");     // Msun
  double n_cgm = pin->GetReal("problem", "n_cgm");    // cm^-3
  double t_cgm = pin->GetReal("problem", "T_cgm");    // K
  double v_bh  = pin->GetReal("problem", "v_bh");     // km/s

  double rho = n_cgm*pu->mu()*pu->atomic_mass_unit_cgs/pu->density_cgs();
  nw.gamma   = pin->GetReal("hydro", "gamma");
  nw.gm_bh   = static_cast<Real>(pu->grav_constant()*m_bh*pu->msun());
  nw.eps     = pin->GetReal("problem", "epsilon");
  nw.v_bh    = static_cast<Real>(v_bh*pu->km_s());
  nw.rho_cgm = static_cast<Real>(rho);
  nw.p_cgm   = static_cast<Real>(rho*t_cgm/pu->temperature_cgs());
  nw.r_sink  = pin->GetReal("problem", "r_sink");
  nw.rho_sink = static_cast<Real>(rho*pin->GetOrAddReal("problem", "rho_sink_frac", 1.0e-2));
  nw.msun_per_mass = 1.0/pu->msun();
  nw.m_acc   = 0.0;

  if (global_variable::my_rank == 0) {
    std::cout << "no_wind: G*M_bh = " << nw.gm_bh << ", v_bh = " << nw.v_bh
              << ", rho_cgm = " << nw.rho_cgm << ", p_cgm = " << nw.p_cgm
              << ", R_acc = 2GM/v^2 = " << 2.0*nw.gm_bh/(nw.v_bh*nw.v_bh)
              << ", r_sink = " << nw.r_sink << ", rho_sink = " << nw.rho_sink
              << " (code units)" << std::endl;
  }
  if (restart) return;

  // --- initial conditions: uniform CGM moving at -v_bh ---
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &u0 = pmbp->phydro->u0;
  const int nhyd = pmbp->phydro->nhydro;
  const int nscal = pmbp->phydro->nscalars;
  const Real rho_c = nw.rho_cgm;
  const Real eint  = nw.p_cgm/(nw.gamma - 1.0);
  const Real vx    = -nw.v_bh;

  par_for("no_wind_ic", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = rho_c;
    u0(m,IM1,k,j,i) = rho_c*vx;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = eint + 0.5*rho_c*vx*vx;
    for (int n = nhyd; n < nhyd+nscal; ++n) u0(m,n,k,j,i) = 0.0;
  });
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void AddUserSrcs()
//! \brief Softened BH point-mass gravity (from primitives w0) and the sink (on u0).
//! Called once per RK stage; the sink removes whatever mass has entered r < r_sink.

void AddUserSrcs(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  const int nmkji = pmbp->nmb_thispack*nkji;
  auto &size = pmbp->pmb->mb_size;
  const auto &w0 = pmbp->phydro->w0;
  auto &u0 = pmbp->phydro->u0;
  const int nvar = u0.extent_int(1);

  const Real gm_bh = nw.gm_bh;
  const Real eps2  = nw.eps*nw.eps;
  const Real rs2   = nw.r_sink*nw.r_sink;
  const Real rho_s = nw.rho_sink;

  double dm_tot = 0.0;
  Kokkos::parallel_reduce("no_wind_grav_sink",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, double &dm) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real r2 = x*x + y*y + z*z;
    Real s2 = r2 + eps2;

    // softened point-mass gravity
    Real rho = w0(m,IDN,k,j,i);
    Real g_over_r = gm_bh/(s2*sqrt(s2));
    Real vdotr = w0(m,IVX,k,j,i)*x + w0(m,IVY,k,j,i)*y + w0(m,IVZ,k,j,i)*z;
    u0(m,IM1,k,j,i) -= bdt*rho*g_over_r*x;
    u0(m,IM2,k,j,i) -= bdt*rho*g_over_r*y;
    u0(m,IM3,k,j,i) -= bdt*rho*g_over_r*z;
    u0(m,IEN,k,j,i) -= bdt*rho*g_over_r*vdotr;

    // sink: scale all conserved variables down to rho_sink (keeps v and T)
    if (r2 < rs2) {
      Real d = u0(m,IDN,k,j,i);
      if (d > rho_s) {
        Real f = rho_s/d;
        for (int n = 0; n < nvar; ++n) u0(m,n,k,j,i) *= f;
        Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
        dm += static_cast<double>(d - rho_s)*static_cast<double>(vol);
      }
    }
  }, Kokkos::Sum<double>(dm_tot));
  nw.m_acc += dm_tot;
}

//----------------------------------------------------------------------------------------
//! \fn void NoWindHistory()
//! \brief Accreted mass (rank-local; history output sums over ranks).

void NoWindHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 2;
  pdata->label[0] = "M_acc";
  pdata->label[1] = "M_acc_msun";
  pdata->hdata[0] = static_cast<Real>(nw.m_acc);
  pdata->hdata[1] = static_cast<Real>(nw.m_acc*nw.msun_per_mass);
}

//----------------------------------------------------------------------------------------
//! \fn void InflowX1BC()
//! \brief Outer-x1 ghosts: uniform CGM at n_cgm, T_cgm, velocity -v_bh x_hat. Fills all
//! j,k (incl. ghosts) of the outer-x1 ghost slab. Sets w0 and u0 directly (ideal gas),
//! only on blocks whose outer-x1 face is user.

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

  const Real rho  = nw.rho_cgm;
  const Real eint = nw.p_cgm/(nw.gamma - 1.0);
  const Real vx   = -nw.v_bh;

  par_for("no_wind_ox1", DevExeSpace(), 0, pmbp->nmb_thispack-1, 0, n3-1, 0, n2-1,
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
  });
}

}  // namespace
