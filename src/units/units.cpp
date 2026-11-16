//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file units.cpp

#include <string>

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "units.hpp"

namespace units {
namespace {
// Read a unit scale in double precision. GetReal()/GetOrAddReal() cast to Real, which in
// a single-precision build overflows scales such as mass_cgs ~ 1e40 (float max ~ 3.4e38).
double GetOrAddDouble(ParameterInput *pin, const std::string &block,
                      const std::string &name, const std::string &def_value) {
  return std::stod(pin->GetOrAddString(block, name, def_value));
}
double GetDouble(ParameterInput *pin, const std::string &block, const std::string &name) {
  return std::stod(pin->GetString(block, name));
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \brief Units constructor
Units::Units(ParameterInput *pin) :
  length_cgs_(GetOrAddDouble(pin, "units", "length_cgs", "1.0")),
  mass_cgs_(GetOrAddDouble(pin, "units", "mass_cgs", "1.0")),
  time_cgs_(GetOrAddDouble(pin, "units", "time_cgs", "1.0")),
  mu_(GetOrAddDouble(pin, "units", "mu", "1.0")) {
  // NOTE(@pdmullen): In GR, it is oft preferrable to specify the (1) black hole mass and
  // (2) density unit directly instead of specifying MLT units.  The below resets MLT
  // units for GR calculations (or sets them if they were omitted from the input file).
  if (pin->GetOrAddBoolean("coord","general_rel",false)) {
    // Get input density scale and black hole mass in cgs units
    double density_scale = GetDouble(pin, "units", "density_cgs");       // (cgs)
    double bhmass_cgs = GetDouble(pin, "units", "bhmass_msun")*msun_cgs;  // BH mass (cgs)

    // (Re)set MLT units
    length_cgs_ = grav_constant_cgs*bhmass_cgs/SQR(speed_of_light_cgs);  // GM/c^2
    mass_cgs_ = density_scale*(length_cgs_*length_cgs_*length_cgs_);     // dens_cgs*[L]^3
    time_cgs_ = length_cgs_/speed_of_light_cgs;                          // GM/c^3
  }
}

//----------------------------------------------------------------------------------------
//! \brief Units destructor
Units::~Units() {
}

//----------------------------------------------------------------------------------------
// Code scales in cgs units
double Units::length_cgs() const { return length_cgs_; }
double Units::mass_cgs() const { return mass_cgs_; }
double Units::time_cgs() const { return time_cgs_; }

// mean molecular weight
double Units::mu() const { return mu_; }

//----------------------------------------------------------------------------------------
// Derived code scales in cgs units
// Converting variables from code unit to physical (cgs) unit
// i.e., variable_in_physical_unit = variable_in_code_unit * code_scale
double Units::velocity_cgs() const {
  return Units::length_cgs() / Units::time_cgs();
}
double Units::density_cgs() const {
  return Units::mass_cgs() /
         (Units::length_cgs() * Units::length_cgs() * Units::length_cgs());
}
double Units::energy_cgs() const {
  return Units::mass_cgs() * Units::velocity_cgs() * Units::velocity_cgs();
}
double Units::pressure_cgs() const {
  return Units::energy_cgs() /
         (Units::length_cgs() * Units::length_cgs() * Units::length_cgs());
}
double Units::temperature_cgs() const {
  return Units::velocity_cgs() * Units::velocity_cgs() * Units::mu() *
         Units::atomic_mass_unit_cgs / Units::k_boltzmann_cgs;
}

//----------------------------------------------------------------------------------------
// Code unit per X, or X in code units
// Converting variables from physical (cgs) unit to code unit
// i.e., variable_in_code_unit = variable_in_physical_unit * X

// Length
double Units::cm() const { return cm_cgs / Units::length_cgs(); }
double Units::pc() const { return pc_cgs / Units::length_cgs(); }
double Units::kpc() const { return kpc_cgs / Units::length_cgs(); }

// Mass
double Units::g() const { return g_cgs / Units::mass_cgs(); }
double Units::msun() const { return msun_cgs / Units::mass_cgs(); }
double Units::atomic_mass_unit() const { return atomic_mass_unit_cgs / Units::mass_cgs(); }

// Time
double Units::s() const { return s_cgs / Units::time_cgs(); }
double Units::yr() const { return yr_cgs / Units::time_cgs(); }
double Units::myr() const { return myr_cgs / Units::time_cgs(); }

// Velocity
double Units::cm_s() const { return cm_s_cgs / Units::velocity_cgs(); }
double Units::km_s() const { return km_s_cgs / Units::velocity_cgs(); }

// Density
double Units::g_cm3() const { return g_cm3_cgs / Units::density_cgs(); }

// Energy
double Units::erg() const { return erg_cgs / Units::energy_cgs(); }

// Pressure
double Units::dyne_cm2() const { return dyne_cm2_cgs / Units::pressure_cgs(); }

// Temperature
double Units::kelvin() const { return kelvin_cgs / Units::temperature_cgs(); }

//----------------------------------------------------------------------------------------
// Physical Constants in code units
double Units::k_boltzmann() const {
  return k_boltzmann_cgs / (Units::energy_cgs() / Units::temperature_cgs());
}

double Units::grav_constant() const {
  return grav_constant_cgs * Units::density_cgs() * Units::time_cgs() * Units::time_cgs();
}

double Units::speed_of_light() const {
  return speed_of_light_cgs / Units::velocity_cgs();
}

} // namespace units
