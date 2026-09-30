#ifndef SRCTERMS_SRCTERMS_HPP_
#define SRCTERMS_SRCTERMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.hpp
//! \brief Data, functions, and classes to implement various source terms in the hydro
//! and/or MHD equations of motion.  Currently implemented:
//!  (1) constant (gravitational) acceleration - for RTI
//!  (2) shearing box in 2D (x-z), for both hydro and MHD
//!  (3) random forcing to drive turbulence - implemented in TurbulenceDriver class
//!  (4) optically thin ISM cooling, either explicit inside the RK stages (default) or
//!      operator-split and sub-cycled after the hydro step (cooling_subcycle = true)

#include <map>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class SourceTerms
//! \brief data and functions for physical source terms

class SourceTerms {
 public:
  SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~SourceTerms();

  // data
  // flags for various source terms
  bool const_accel;
  bool ism_cooling;
  bool rel_cooling;
  bool rad_beam;
  bool self_gravity;

  // new timestep
  Real dtnew;

  // data for constant accel
  Real const_accel_val;   // magnitude of accn
  int const_accel_dir;    // direction of accn

  // data for ISM cooling
  Real hrate;

  // data for sub-cycled ISM cooling (operator split, applied by Driver after RK stages)
  bool ism_cooling_subcycle;  // if true: cooling removed from RK stages, done by Driver
  int  cool_nsub_max;         // maximum number of cooling sub-steps per hydro step
  Real cool_cfl;              // each sub-step satisfies dt_sub <= cool_cfl * t_cool_min
  int  last_nsub;             // number of sub-steps used in the most recent cycle
  // tabulated cooling function (float, uniform in log10 T) used by the sub-cycled solver
  bool cool_table;            // if true: table lookup instead of ISMCoolFn() (default)
  DvceArray1D<float> cool_tab;
  float cool_tab_logt0;       // log10(T) of first entry
  float cool_tab_dinv;        // 1/dlog10(T)
  int   cool_tab_n;           // number of entries
  // post-cooling t_cool,min computed inside the cooling kernel (reused by NewTimeStep)
  Real tcool_min_post;
  bool have_tcool_min_post;

  // data for relativistic cooling
  Real crate_rel;
  Real cpower_rel;

  // data for radiation beam source
  Real dii_dt;            // injection rate
  Real pos1, pos2, pos3;  // position of source
  Real dir1, dir2, dir3;  // direction of source
  Real width, spread;     // spatial width of source region, spread in angles

  // functions
  void ApplySrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                     const Real bdt, DvceArray5D<Real> &u0);
  void ApplySrcTerms(DvceArray5D<Real> &i0, const Real bdt);
  void ConstantAccel(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                     const Real bdt, DvceArray5D<Real> &u0);
  void ISMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real bdt, DvceArray5D<Real> &u0);
  void RelCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real bdt, DvceArray5D<Real> &u0);
  void SelfGravity(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                   const Real bdt, DvceArray5D<Real> &u0);
  void BeamSource(DvceArray5D<Real> &i0, const Real bdt);
  void NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos);
  Real MinCoolingTime(const DvceArray5D<Real> &w0, const EOS_Data &eos);
  void SubcycleISMCooling(DvceArray5D<Real> &w0, DvceArray5D<Real> &u0,
                          const EOS_Data &eos, const Real dt);

 private:
  MeshBlockPack *pmy_pack;
};

#endif  // SRCTERMS_SRCTERMS_HPP_