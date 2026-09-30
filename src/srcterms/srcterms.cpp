//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.cpp
//  Implements various (physics) source terms to be added to the Hydro or MHD eqns.
//  Source terms objects are stored in the respective fluid class, so that
//  Hydro/MHD can have different source terms
//
//  ISM cooling has two modes, selected by "cooling_subcycle" in the srcterms block:
//   false (default): explicit update inside every RK stage (ISMCooling). The global dt is
//                    limited by cfl_number * t_cool_min (see srcterms_newdt.cpp).
//   true:            cooling is removed from the RK stages. After the hydro step the
//                    Driver calls SubcycleISMCooling(), which integrates cooling over the
//                    full hydro dt with n_sub uniform explicit sub-steps, n_sub being the
//                    SAME for every cell (no GPU warp divergence).

#include "srcterms.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <limits>
#include <string> // string

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "gravity/gravity.hpp"
#include "hydro/hydro.hpp"
#include "ismcooling.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "units/units.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters
// Only source terms specified in input file are initialized.
// Block name passed to constructor will be one of "hydro_srcterms", "mhd_srcterms",
// or "rad_srcterms"

SourceTerms::SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  // Read flags for each source term implemented (default false)
  const_accel = pin->GetOrAddBoolean(block, "const_accel", false);
  ism_cooling = pin->GetOrAddBoolean(block, "ism_cooling", false);
  rel_cooling = pin->GetOrAddBoolean(block, "rel_cooling", false);
  rad_beam = pin->GetOrAddBoolean(block, "rad_beam", false);
  self_gravity = pin->GetOrAddBoolean(block, "self_gravity", false);

  // defaults for sub-cycled cooling (only read from input when ism_cooling = true)
  ism_cooling_subcycle = false;
  cool_nsub_max = 1;
  cool_cfl = 1.0;
  last_nsub = 1;
  cool_table = false;
  cool_tab_logt0 = 0.0f;
  cool_tab_dinv = 1.0f;
  cool_tab_n = 0;
  tcool_min_post = static_cast<Real>(std::numeric_limits<float>::max());
  have_tcool_min_post = false;

  // (1) read data for (constant) gravitational acceleration
  if (const_accel) {
    const_accel_val = pin->GetReal(block, "const_accel_val");
    const_accel_dir = pin->GetInteger(block, "const_accel_dir");
    if (const_accel_dir < 1 || const_accel_dir > 3) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "const_accle_dir must be 1,2, or 3" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // (2) Optically thin ISM cooling
  if (ism_cooling) {
    hrate = pin->GetReal(block, "hrate");
    ism_cooling_subcycle = pin->GetOrAddBoolean(block, "cooling_subcycle", false);
    if (ism_cooling_subcycle) {
      cool_nsub_max = pin->GetOrAddInteger(block, "cool_nsub_max", 100);
      cool_cfl      = pin->GetOrAddReal(block, "cool_cfl", 0.5);
      if (cool_nsub_max < 1 || cool_cfl <= 0.0) {
        std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__
                  << std::endl << "cool_nsub_max must be >= 1 and cool_cfl > 0"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      cool_table = pin->GetOrAddBoolean(block, "cool_table", true);
      if (cool_table) {
        // Lambda(T) tabulated on a uniform grid in log10 T from 10 K to 10^9 K. Stored as
        // float: on GPUs with slow FP64 (e.g. RTX/Ada workstation cards, 1/64 rate) this
        // replaces the double-precision log10/pow/exp in ISMCoolFn() by one log10f and
        // two cached loads. Below/above the range the end values are used.
        const double lt0 = 1.0, lt1 = 9.0, dlt = 0.002;
        cool_tab_n     = static_cast<int>(std::lround((lt1 - lt0)/dlt)) + 1;
        cool_tab_logt0 = static_cast<float>(lt0);
        cool_tab_dinv  = static_cast<float>(1.0/dlt);
        cool_tab = DvceArray1D<float>("cool_tab", cool_tab_n);
        auto htab = Kokkos::create_mirror_view(cool_tab);
        for (int n = 0; n < cool_tab_n; ++n) {
          htab(n) = static_cast<float>(ISMCoolFn(std::pow(10.0, lt0 + n*dlt)));
        }
        Kokkos::deep_copy(cool_tab, htab);
        // report interpolation accuracy at interval midpoints, skipping the two intervals
        // that straddle the (discontinuous) branch boundaries of ISMCoolFn at 4.2 and 8.15
        double maxerr = 0.0;
        for (int n = 0; n < cool_tab_n - 1; ++n) {
          double a = lt0 + n*dlt, b = a + dlt;
          const double tol = 1.0e-9;
          if ((a <= 4.2 + tol && b >= 4.2 - tol) ||
              (a <= 8.15 + tol && b >= 8.15 - tol)) continue;
          double exact = ISMCoolFn(std::pow(10.0, a + 0.5*dlt));
          double interp = 0.5*(static_cast<double>(htab(n)) + static_cast<double>(htab(n+1)));
          if (exact > 0.0) maxerr = std::max(maxerr, std::fabs(interp/exact - 1.0));
        }
        if (global_variable::my_rank == 0) {
          std::cout << "ISM cooling: tabulated Lambda(T), " << cool_tab_n
                    << " entries, max rel. interpolation error = " << maxerr << std::endl;
        }
      }
      if (global_variable::my_rank == 0) {
        std::cout << "ISM cooling: operator-split sub-cycling, cool_nsub_max="
                  << cool_nsub_max << " cool_cfl=" << cool_cfl << std::endl;
      }
    }
  }

  // (3) optically thin relativistic cooling
  if (rel_cooling) {
    crate_rel = pin->GetReal(block, "crate_rel");
    cpower_rel = pin->GetOrAddReal(block, "cpower_rel", 1.);
  }

  // (4) radiation beam source (radiation)
  if (rad_beam) {
    dii_dt = pin->GetReal(block, "dii_dt");
    pos1 = pin->GetReal(block, "pos_1");
    pos2 = pin->GetReal(block, "pos_2");
    pos3 = pin->GetReal(block, "pos_3");
    dir1 = pin->GetReal(block, "dir_1");
    dir2 = pin->GetReal(block, "dir_2");
    dir3 = pin->GetReal(block, "dir_3");
    width = pin->GetReal(block, "width");
    spread = pin->GetReal(block, "spread");
  }
}

//----------------------------------------------------------------------------------------
// destructor

SourceTerms::~SourceTerms() {
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::ApplySrcTerms
//! \brief Applies selected source terms to input arrays. Two different versions are
//! implemented for fluid and radiation fields, distinguished by their argument lists

void SourceTerms::ApplySrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                                const Real bdt, DvceArray5D<Real> &u0) {
  // NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars
  if (const_accel) ConstantAccel(w0, eos_data,  bdt, u0);
  // with sub-cycling, cooling is NOT applied in the RK stages; the Driver applies it
  // once per cycle after the hydro update (see SubcycleISMCooling below)
  if (ism_cooling && !ism_cooling_subcycle) ISMCooling(w0, eos_data, bdt, u0);
  if (rel_cooling) RelCooling(w0, eos_data, bdt, u0);
  if (self_gravity) SelfGravity(w0, eos_data, bdt, u0);
  return;
}

void SourceTerms::ApplySrcTerms(DvceArray5D<Real> &i0, const Real bdt) {
  if (rad_beam) BeamSource(i0, bdt);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::ConstantAccel
//! \brief Add constant acceleration
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::ConstantAccel(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                                const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  Real &g = const_accel_val;
  int &dir = const_accel_dir;

  par_for("const_acc", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real src = bdt*g*w0(m,IDN,k,j,i);
    u0(m,dir,k,j,i) += src;
    if (eos_data.is_ideal) { u0(m,IEN,k,j,i) += src*w0(m,dir,k,j,i); }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::ISMCooling()
//! \brief Add explict ISM cooling and heating source terms in the energy equations.
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::ISMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                             const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real heating_rate = hrate;
  Real temp_unit = pmy_pack->punit->temperature_cgs();
  Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                /pmy_pack->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                      /n_unit/n_unit;
  Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()/n_unit;

  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // temperature in cgs unit
    Real temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
    Real gamma_heating = heating_rate/heating_unit;

    u0(m,IEN,k,j,i) -= bdt * w0(m,IDN,k,j,i) *
                        (w0(m,IDN,k,j,i) * lambda_cooling - gamma_heating);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::RelCooling()
//! \brief Add explict relativistic cooling in the energy and momentum equations.
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::RelCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                             const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real cooling_rate = crate_rel;
  Real cooling_power = cpower_rel;

  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // temperature in cgs unit
    Real temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;

    auto &ux = w0(m,IVX,k,j,i);
    auto &uy = w0(m,IVY,k,j,i);
    auto &uz = w0(m,IVZ,k,j,i);
    Real ut = sqrt(1.0 + ux*ux + uy*uy + uz*uz);

    u0(m,IEN,k,j,i) -= bdt*w0(m,IDN,k,j,i)*ut*pow((temp*cooling_rate), cooling_power);
    u0(m,IM1,k,j,i) -= bdt*w0(m,IDN,k,j,i)*ux*pow((temp*cooling_rate), cooling_power);
    u0(m,IM2,k,j,i) -= bdt*w0(m,IDN,k,j,i)*uy*pow((temp*cooling_rate), cooling_power);
    u0(m,IM3,k,j,i) -= bdt*w0(m,IDN,k,j,i)*uz*pow((temp*cooling_rate), cooling_power);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::SelfGravity
//! \brief Adds source terms for self-gravitational acceleration to conserved variables
//! \note
//! This implements the source term formula in Mullen, Hanawa and Gammie 2020, but only
//! for the momentum part. The energy source term is not conservative in this version.
//! Also note that this implementation is not exactly conservative when the potential
//! contains a residual error (Multigrid has small but non-zero residual).

void SourceTerms::SelfGravity(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                              const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  // Get gravitational potential - check if gravity is enabled
  if (pmy_pack->pgrav == nullptr) {
    std::cout << "### ERROR in SourceTerms::SelfGravity" << std::endl
              << "self_gravity source term enabled but pgrav is null" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &phi = pmy_pack->pgrav->phi;

  // Get Godunov density fluxes from Riemann solver
  // (following Mullen, Hanawa & Gammie 2020 for the energy source term)
  DvceArray5D<Real> flx1, flx2, flx3;
  if (pmy_pack->pmhd != nullptr) {
    flx1 = pmy_pack->pmhd->uflx.x1f;
    flx2 = pmy_pack->pmhd->uflx.x2f;
    flx3 = pmy_pack->pmhd->uflx.x3f;
  } else {
    flx1 = pmy_pack->phydro->uflx.x1f;
    flx2 = pmy_pack->phydro->uflx.x2f;
    flx3 = pmy_pack->phydro->uflx.x3f;
  }

  // x1-direction momentum and energy source terms
  par_for("selfgrav_x1",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dx1 = mbsize.d_view(m).dx1;
    Real hdtodx1 = 0.5*bdt/dx1;
    Real dpl = -(phi(m,0,k,j,i  ) - phi(m,0,k,j,i-1));
    Real dpr = -(phi(m,0,k,j,i+1) - phi(m,0,k,j,i  ));

    // Add momentum source term
    u0(m,IM1,k,j,i) += hdtodx1 * w0(m,IDN,k,j,i) * (dpl + dpr);

    // Add energy source term using Godunov fluxes (ideal EOS only)
    if (eos_data.is_ideal) {
      u0(m,IEN,k,j,i) += hdtodx1 * (flx1(m,IDN,k,j,i  ) * dpl
                                    + flx1(m,IDN,k,j,i+1) * dpr);
    }
  });

  if (multi_d) {
    // x2-direction momentum and energy source terms
    par_for("selfgrav_x2",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx2 = mbsize.d_view(m).dx2;
      Real hdtodx2 = 0.5*bdt/dx2;
      Real dpl = -(phi(m,0,k,j,  i) - phi(m,0,k,j-1,i));
      Real dpr = -(phi(m,0,k,j+1,i) - phi(m,0,k,j,  i));

      // Add momentum source term
      u0(m,IM2,k,j,i) += hdtodx2 * w0(m,IDN,k,j,i) * (dpl + dpr);

      // Add energy source term using Godunov fluxes (ideal EOS only)
      if (eos_data.is_ideal) {
        u0(m,IEN,k,j,i) += hdtodx2 * (flx2(m,IDN,k,j,  i) * dpl
                                      + flx2(m,IDN,k,j+1,i) * dpr);
      }
    });
  }

  if (three_d) {
    // x3-direction momentum and energy source terms
    par_for("selfgrav_x3",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx3 = mbsize.d_view(m).dx3;
      Real hdtodx3 = 0.5*bdt/dx3;
      Real dpl = -(phi(m,0,k,  j,i) - phi(m,0,k-1,j,i));
      Real dpr = -(phi(m,0,k+1,j,i) - phi(m,0,k,  j,i));

      // Add momentum source term
      u0(m,IM3,k,j,i) += hdtodx3 * w0(m,IDN,k,j,i) * (dpl + dpr);

      // Add energy source term using Godunov fluxes (ideal EOS only)
      if (eos_data.is_ideal) {
        u0(m,IEN,k,j,i) += hdtodx3 * (flx3(m,IDN,k,  j,i) * dpl
                                      + flx3(m,IDN,k+1,j,i) * dpr);
      }
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::BeamSource()
//! \brief Add beam of radiation at position (pos1,pos2,pos3) moving in direction
//! (dir1,dir2,dir3) with physical width and angular spread (width,spread)

void SourceTerms::BeamSource(DvceArray5D<Real> &i0, const Real bdt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = (pmy_pack->nmb_thispack-1);
  int nang1 = (pmy_pack->prad->prgeo->nangles-1);

  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;

  Real &p1 = pos1, &p2 = pos2, &p3 = pos3;
  Real &d1 = dir1, &d2 = dir2, &d3 = dir3;
  Real &dii_dt_ = dii_dt;
  Real &width_ = width;
  Real &spread_ = spread;

  auto &tc = pmy_pack->prad->tetcov_c;
  auto &nh_c_ = pmy_pack->prad->nh_c;
  auto &tet_c_ = pmy_pack->prad->tet_c;
  par_for("rad_beam",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
    Real dgx[4][4], dgy[4][4], dgz[4][4];
    ComputeMetricDerivatives(x1v,x2v,x3v,flat,spin,dgx,dgy,dgz);
    Real e[4][4], e_cov[4][4], omega[4][4][4];
    ComputeTetrad(x1v,x2v,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);

    // Calculate proper distance to beam origin and minimum angle between directions
    Real dx1 = x1v - p1;
    Real dx2 = x2v - p2;
    Real dx3 = x3v - p3;
    Real dx_sq = glower[1][1]*dx1*dx1 +2.0*glower[1][2]*dx1*dx2 + 2.0*glower[1][3]*dx1*dx3
               + glower[2][2]*dx2*dx2 +2.0*glower[2][3]*dx2*dx3
               + glower[3][3]*dx3*dx3;
    Real mu_min = cos(spread_/2.0*M_PI/180.0);

    // Calculate contravariant time component of direction
    Real temp_a = glower[0][0];
    Real temp_b = 2.0*(glower[0][1]*d1 + glower[0][2]*d2 + glower[0][3]*d3);
    Real temp_c = glower[1][1]*d1*d1 + 2.0*glower[1][2]*d1*d2 + 2.0*glower[1][3]*d1*d3
                + glower[2][2]*d2*d2 + 2.0*glower[2][3]*d2*d3
                + glower[3][3]*d3*d3;
    Real d0 = ((-temp_b - sqrt(SQR(temp_b) - 4.0*temp_a*temp_c))/(2.0*temp_a));

    // lower indices
    Real dc0 = glower[0][0]*d0 + glower[0][1]*d1 + glower[0][2]*d2 + glower[0][3]*d3;
    Real dc1 = glower[0][1]*d0 + glower[1][1]*d1 + glower[1][2]*d2 + glower[1][3]*d3;
    Real dc2 = glower[0][2]*d0 + glower[1][2]*d1 + glower[2][2]*d2 + glower[2][3]*d3;
    Real dc3 = glower[0][3]*d0 + glower[1][3]*d1 + glower[2][3]*d2 + glower[3][3]*d3;

    // Calculate covariant direction in tetrad frame
    Real dtc0 = (tet_c_(m,0,0,k,j,i)*dc0 + tet_c_(m,0,1,k,j,i)*dc1 +
                 tet_c_(m,0,2,k,j,i)*dc2 + tet_c_(m,0,3,k,j,i)*dc3);
    Real dtc1 = (tet_c_(m,1,0,k,j,i)*dc0 + tet_c_(m,1,1,k,j,i)*dc1 +
                 tet_c_(m,1,2,k,j,i)*dc2 + tet_c_(m,1,3,k,j,i)*dc3)/(-dtc0);
    Real dtc2 = (tet_c_(m,2,0,k,j,i)*dc0 + tet_c_(m,2,1,k,j,i)*dc1 +
                 tet_c_(m,2,2,k,j,i)*dc2 + tet_c_(m,2,3,k,j,i)*dc3)/(-dtc0);
    Real dtc3 = (tet_c_(m,3,0,k,j,i)*dc0 + tet_c_(m,3,1,k,j,i)*dc1 +
                 tet_c_(m,3,2,k,j,i)*dc2 + tet_c_(m,3,3,k,j,i)*dc3)/(-dtc0);

    // Go through angles
    for (int n=0; n<=nang1; ++n) {
      Real mu = (nh_c_.d_view(n,1) * dtc1
               + nh_c_.d_view(n,2) * dtc2
               + nh_c_.d_view(n,3) * dtc3);
      if ((dx_sq < SQR(width_/2.0)) && (mu > mu_min)) {
        Real n0 = tet_c_(m,0,0,k,j,i);
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        i0(m,n,k,j,i) += n0*n_0*dii_dt_*bdt;
      }
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn CoolLambda()
//! \brief Lambda(T) [erg cm^3/s], either by linear interpolation of the float table
//! (uniform in log10 T, one log10f per call) or by the analytic ISMCoolFn().

KOKKOS_INLINE_FUNCTION
Real CoolLambda(const Real temp, const bool use_tab, const DvceArray1D<float> &tab,
                const float lt0, const float dinv, const int ntab) {
  if (!use_tab) return ISMCoolFn(temp);
  float x = (log10f(static_cast<float>(temp)) - lt0)*dinv;
  x = fminf(fmaxf(x, 0.0f), static_cast<float>(ntab - 1) - 1.0e-3f);
  int   i0 = static_cast<int>(x);
  float f  = x - static_cast<float>(i0);
  return static_cast<Real>(tab(i0) + f*(tab(i0+1) - tab(i0)));
}

//----------------------------------------------------------------------------------------
//! \fn Real SourceTerms::MinCoolingTime()
//! \brief Rank-local minimum of t_cool = e_int / |n^2 Lambda(T) - n Gamma| over active
//! cells. Cells at (or within 1% of) the temperature floor are excluded: they are clamped
//! by the floor and cannot cool further, so they must not drive the number of sub-steps.
//! Returns FLT_MAX if no cell contributes. The caller performs any MPI reduction.

Real SourceTerms::MinCoolingTime(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji  = nx3*nx2*nx1;
  const int nji   = nx2*nx1;

  Real gm1          = eos_data.gamma - 1.0;
  Real tfloor       = eos_data.tfloor;     // floor on P/rho, code units
  Real heating_rate = hrate;
  Real temp_unit    = pmy_pack->punit->temperature_cgs();
  Real n_unit       = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                      /pmy_pack->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                      /n_unit/n_unit;
  Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                      /n_unit;

  const bool use_tab = cool_table;
  auto tab = cool_tab;
  const float lt0 = cool_tab_logt0, dinv = cool_tab_dinv;
  const int ntab = cool_tab_n;

  Real tmin = static_cast<Real>(std::numeric_limits<float>::max());
  Kokkos::parallel_reduce("srcterms_min_tcool",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_t) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real rho  = w0(m,IDN,k,j,i);
    Real eint = w0(m,IEN,k,j,i);
    Real p_over_rho = gm1*eint/rho;
    if (p_over_rho > 1.01*tfloor) {
      Real temp = temp_unit*p_over_rho;
      Real lam  = CoolLambda(temp, use_tab, tab, lt0, dinv, ntab);
      Real net  = FLT_MIN + fabs(rho*(rho*lam/cooling_unit - heating_rate/heating_unit));
      min_t = fmin(eint/net, min_t);
    }
  }, Kokkos::Min<Real>(tmin));

  return tmin;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::SubcycleISMCooling()
//! \brief Operator-split explicit ISM cooling over the full hydro step dt.
//!
//! (1) t_min = global minimum cooling time of the post-hydro state (one MPI_Allreduce)
//! (2) n_sub = min(ceil(dt/(cool_cfl*t_min)), cool_nsub_max), identical on every rank
//! (3) one kernel: every cell performs exactly n_sub explicit sub-steps of dt/n_sub.
//!     The loop count is uniform, so there is no warp divergence, and e_int stays in a
//!     register (one global read + one write per cell regardless of n_sub).
//!
//! The kernel runs over active AND ghost cells. After the final ConToPrim of the RK step,
//! ghost cells hold copies of neighbouring active cells; since n_sub and dt_sub are
//! global, cooling a ghost reproduces the neighbour's result, so no extra boundary
//! exchange is needed before the next cycle's flux calculation (exact on uniform grids;
//! approximate at AMR fine/coarse boundaries).
//!
//! Must be called when u0 and w0 are consistent (after the RK stages). Both are updated:
//! only the internal energy changes, so w0(IEN) = e_new and u0(IEN) += (e_new - e_old).

void SourceTerms::SubcycleISMCooling(DvceArray5D<Real> &w0, DvceArray5D<Real> &u0,
                                     const EOS_Data &eos_data, const Real dt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int n1m1 = indcs.nx1 + 2*(indcs.ng) - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng) - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng) - 1) : 0;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // (1) global minimum cooling time
  Real tmin = MinCoolingTime(w0, eos_data);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &tmin, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif

  // (2) number of sub-steps, same for every cell
  int nsub = 1;
  if (tmin < static_cast<Real>(std::numeric_limits<float>::max())) {
    Real ratio = dt/(cool_cfl*tmin);
    // guard the double->int conversion against overflow before clamping
    nsub = (ratio >= static_cast<Real>(cool_nsub_max)) ? cool_nsub_max :
           static_cast<int>(std::ceil(ratio));
  }
  nsub = std::max(1, std::min(nsub, cool_nsub_max));
  last_nsub = nsub;
  const Real dts = dt/static_cast<Real>(nsub);

  // (3) constants captured by value in the kernel
  Real gm1          = eos_data.gamma - 1.0;
  Real pfloor       = eos_data.pfloor;
  Real tfloor       = eos_data.tfloor;
  Real heating_rate = hrate;
  Real temp_unit    = pmy_pack->punit->temperature_cgs();
  Real n_unit       = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                      /pmy_pack->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                      /n_unit/n_unit;
  Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                      /n_unit;

  const bool use_tab = cool_table;
  auto tab = cool_tab;
  const float lt0 = cool_tab_logt0, dinv = cool_tab_dinv;
  const int ntab = cool_tab_n;

  // flattened loop over ALL cells (active + ghost), fused with the reduction that gives
  // the post-cooling t_cool,min for the next timestep (saves a separate pass)
  const int nc1 = n1m1 + 1, nc2 = n2m1 + 1, nc3 = n3m1 + 1;
  const int nji = nc2*nc1, nkji = nc3*nji, nmkji = (nmb1 + 1)*nkji;
  Real tpost = static_cast<Real>(std::numeric_limits<float>::max());

  Kokkos::parallel_reduce("srcterms_cool_subcycle",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_t) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nc1;
    int i = (idx - m*nkji - k*nji - j*nc1);

    Real rho  = w0(m,IDN,k,j,i);
    Real e0   = w0(m,IEN,k,j,i);                 // internal energy density (post C2P)
    Real efl  = fmax(pfloor, rho*tfloor)/gm1;    // same floors as ConsToPrim
    Real heat = heating_rate/heating_unit;
    Real e    = e0;

    for (int s = 0; s < nsub; ++s) {
      Real temp = temp_unit*gm1*e/rho;
      Real lam  = CoolLambda(temp, use_tab, tab, lt0, dinv, ntab);
      Real net  = rho*(rho*lam/cooling_unit - heat);
      e = fmax(e - dts*net, efl);
    }

    w0(m,IEN,k,j,i)  = e;             // primitives stay consistent, no extra C2P needed
    u0(m,IEN,k,j,i) += (e - e0);      // total energy changes by the same amount

    // post-cooling cooling time of this cell (same definition as MinCoolingTime)
    Real p_over_rho = gm1*e/rho;
    if (p_over_rho > 1.01*tfloor) {
      Real temp = temp_unit*p_over_rho;
      Real lam  = CoolLambda(temp, use_tab, tab, lt0, dinv, ntab);
      Real net  = FLT_MIN + fabs(rho*(rho*lam/cooling_unit - heat));
      min_t = fmin(e/net, min_t);
    }
  }, Kokkos::Min<Real>(tpost));

  tcool_min_post = tpost;              // rank-local; Mesh::NewTimeStep reduces over ranks
  have_tcool_min_post = true;

  return;
}