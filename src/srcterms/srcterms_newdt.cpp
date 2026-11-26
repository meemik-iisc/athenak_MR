//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms_newdt.cpp
//! \brief function to compute timestep for source terms across all MeshBlock(s) in a
//! MeshBlockPack
//!
//! ISM cooling:
//!  explicit (default):     dtnew = t_cool_min; Mesh::NewTimeStep() applies cfl_number.
//!  sub-cycled (cooling_subcycle = true): cooling no longer needs dt < t_cool, it only
//!     needs n_sub <= cool_nsub_max. So dtnew is the cap
//!         dt <= cool_nsub_max * cool_cfl * t_cool_min
//!     divided by cfl_number, because Mesh::NewTimeStep() multiplies dtnew by it.

#include <float.h>

#include <algorithm>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "ismcooling.hpp"
#include "srcterms.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::NewTimeStep()
//! \brief Compute new timestep for source terms.

void SourceTerms::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());

  if (ism_cooling && ism_cooling_subcycle) {
    // reuse the post-cooling minimum computed inside the cooling kernel if available
    // (every cycle after the first); only the initial call needs a separate reduction
    Real tmin = have_tcool_min_post ? tcool_min_post : MinCoolingTime(w0, eos_data);
    if (tmin < static_cast<Real>(std::numeric_limits<float>::max())) {
      Real cap = static_cast<Real>(cool_nsub_max)*cool_cfl*tmin
                 /(pmy_pack->pmesh->cfl_no);
      dtnew = std::min(dtnew, cap);
    }
  } else if (ism_cooling) {
    Real gamma = eos_data.gamma;
    Real gm1 = gamma - 1.0;
    Real heating_rate = hrate;
    // unit factors in double (rho_unit and P_unit/t_unit can over/underflow float),
    // cast to Real once so the kernels run in the build precision
    const units::Units *pu = pmy_pack->punit;
    const double n_unit_d = pu->density_cgs()/pu->mu()/pu->atomic_mass_unit_cgs;
    Real temp_unit    = static_cast<Real>(pu->temperature_cgs());
    Real cooling_unit = static_cast<Real>(pu->pressure_cgs()/pu->time_cgs()
                                          /n_unit_d/n_unit_d);
    Real heating_unit = static_cast<Real>(pu->pressure_cgs()/pu->time_cgs()/n_unit_d);
    const bool mask = cool_mask;
    const int imask = cool_mask_index;
    const Real mthr = cool_mask_threshold;
    const Real href = heat_ref_rho, hrmin = heat_rho_min, htmax = heat_tmax;
    const bool use_tab = cool_table;
    auto tab = cool_tab;
    const float lt0 = cool_tab_logt0, dinv = cool_tab_dinv;
    const int ntab = cool_tab_n;

    // find smallest (e/cooling_rate) in each cell
    Kokkos::parallel_reduce("srcterms_cooling_newdt",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      if (mask && w0(m,imask,k,j,i) < mthr) return;

      // temperature in cgs unit
      Real temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
      Real &eint = w0(m,IEN,k,j,i);

      Real lambda_cooling = CoolLambda(temp, use_tab, tab, lt0, dinv, ntab)/cooling_unit;
      Real rr = HeatBalanceRho(w0(m,IDN,k,j,i), temp, href, hrmin, htmax);

      // add a tiny number
      Real cooling_heating = FLT_MIN + fabs(w0(m,IDN,k,j,i) *
                             (rr * lambda_cooling - heating_rate/heating_unit));

      min_dt = fmin((eint/cooling_heating), min_dt);
    }, Kokkos::Min<Real>(dtnew));
  }

  if (rel_cooling) {
    Real gamma = eos_data.gamma;
    Real gm1 = gamma - 1.0;
    Real cooling_rate = crate_rel;
    Real cooling_power = cpower_rel;

    // find smallest (e/cooling_rate) in each cell
    Kokkos::parallel_reduce("srcterms_cooling_newdt",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      // temperature in cgs unit
      Real temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
      Real &eint = w0(m,IEN,k,j,i);

      auto &ux = w0(m, IVX, k, j, i);
      auto &uy = w0(m, IVY, k, j, i);
      auto &uz = w0(m, IVZ, k, j, i);
      Real ut = sqrt(1.0 + ux * ux + uy * uy + uz * uz);

      // The following should be approximately correct
      // add a tiny number
      Real cooling_heating = FLT_MIN + fabs(w0(m,IDN,k,j,i) * ut *
                             pow(temp*cooling_rate, cooling_power));

      min_dt = fmin((eint/cooling_heating), min_dt);
    }, Kokkos::Min<Real>(dtnew));
  }

  return;
}