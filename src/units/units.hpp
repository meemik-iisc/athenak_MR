#ifndef UNITS_UNITS_HPP_
#define UNITS_UNITS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file units.hpp
//! \brief definitions for Units class

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"

namespace units {

//----------------------------------------------------------------------------------------
//! \class Units
//! \brief data and definitions of functions used to store and access units
//  Functions are implemented in units.cpp

class Units {
 public:
  explicit Units(ParameterInput *pin);
  ~Units();

  // data
  // CGS unit per X
  static constexpr double cm_cgs = 1.0;                           // cm
  static constexpr double pc_cgs = 3.0856775809623245e+18;        // cm
  static constexpr double kpc_cgs = 3.0856775809623245e+21;       // cm
  static constexpr double g_cgs = 1.0;                            // g
  static constexpr double msun_cgs = 1.98841586e+33;              // g
  static constexpr double atomic_mass_unit_cgs = 1.660538921e-24; // g
  static constexpr double s_cgs = 1.0;                            // s
  static constexpr double yr_cgs = 3.15576e+7;                    // s
  static constexpr double myr_cgs = 3.15576e+13;                  // s
  static constexpr double cm_s_cgs = 1.0;                         // cm/s
  static constexpr double km_s_cgs = 1.0e5;                       // cm/s
  static constexpr double g_cm3_cgs = 1.0;                        // g/cm^3
  static constexpr double erg_cgs = 1.0;                          // erg
  static constexpr double dyne_cm2_cgs = 1.0;                     // dyne/cm^2
  static constexpr double kelvin_cgs = 1.0;                       // k

  // PHYSICAL CONSTANTS
  static constexpr double k_boltzmann_cgs = 1.3806488e-16;          // erg/k
  static constexpr double grav_constant_cgs = 6.67408e-8;           // cm^3/(g*s^2)
  static constexpr double speed_of_light_cgs = 2.99792458e10;       // cm/s
  static constexpr double rad_constant_cgs = 7.56573325e-15;        // erg/(cm^3*K^4)
  static constexpr double electron_rest_mass_energy_cgs = 5.93e9;   // k

  // Power Law Coefficients (cgs)
  static constexpr double rosseland_coef_cgs = 1.95705e24;
  static constexpr double planck_minus_rosseland_coef_cgs = 7.04536e25;

  // functions
  // Code scales in cgs
  double length_cgs() const;
  double mass_cgs() const;
  double time_cgs() const;

  // mean molecular weight
  double mu() const;

  // Derived code scales in cgs
  double velocity_cgs() const;
  double density_cgs() const;
  double energy_cgs() const;
  double pressure_cgs() const;
  double temperature_cgs() const;

  // Code unit per X, or X in code units
  double cm() const;
  double pc() const;
  double kpc() const;
  double g() const;
  double msun() const;
  double atomic_mass_unit() const;
  double s() const;
  double yr() const;
  double myr() const;
  double cm_s() const;
  double km_s() const;
  double g_cm3() const;
  double erg() const;
  double dyne_cm2() const;
  double kelvin() const;

  // Physical Constants in code units
  double k_boltzmann() const;
  double grav_constant() const;
  double speed_of_light() const;

 private:
  // Specified code scales in cgs units
  // (Multiply code units to get quantities in cgs units)
  // (cgs unit per code unit)
  double length_cgs_, mass_cgs_, time_cgs_;

  // mean molecular weight
  const double mu_;
};

} // namespace units

#endif // UNITS_UNITS_HPP_
