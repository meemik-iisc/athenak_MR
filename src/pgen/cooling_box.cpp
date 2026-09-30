//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cooling_box.cpp
//! \brief Problem generator for a constant density box with radiative cooling
//! \ref (arXiv:2401.00446v1) Dissipation of AGN Jets in clumpy interstellar medium

#include <algorithm>
#include <cmath>
#include <sstream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen.hpp"
#include "srcterms/srcterms.hpp"
#include "srcterms/ismcooling.hpp"
#include "globals.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for jets in a uniform medium
namespace {
    // made global to share with source terms   
    // made global to share with source terms   
    struct CoolBox{
        Real n_amb, T_amb, n_clump, T_clump, gamma_gas;
        KOKKOS_INLINE_FUNCTION CoolBox() = default;
    };
    CoolBox cbox;  // Host global
    // void AddUserSrcs(Mesh *pm, const Real bdt);
} // namespace
  

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
    // user_srcs_func = AddUserSrcs;
     // User boundary function
    MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

    // capture variables for the kernel
    auto &indcs = pmbp->pmesh->mb_indcs;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    auto &size = pmbp->pmb->mb_size;
    // get initial parameters from input file
    Real gamma_gas_     = pin->GetReal("hydro", "gamma");
    Real n_amb_         = pin->GetReal("problem", "n_amb");
    Real T_amb_         = pin->GetReal("problem", "T_amb");
    Real n_clump_       = pin->GetReal("problem", "n_clump");
    Real T_clump_       = pin->GetReal("problem", "T_clump");


    cbox.gamma_gas      = gamma_gas_;
    cbox.n_amb          = n_amb_;
    cbox.T_amb          = T_amb_;
    cbox.n_clump        = n_clump_;
    cbox.T_clump        = T_clump_;
                 

    Real const &gm1     = gamma_gas_ - 1;  
    Real time           = pmbp->pmesh->time;        
    if (restart) return;

    // Select either Hydro or MHD
    if (pmbp->phydro!=nullptr){

        auto &u0                = pmbp->phydro->u0;
        Real rho_amb_code       = n_amb_*pmbp->punit->mu()*pmbp->punit->atomic_mass_unit_cgs/pmbp->punit->density_cgs();
        Real T_amb_code         = T_amb_/pmbp->punit->temperature_cgs();
        Real pres_amb_code      = rho_amb_code*T_amb_code; 
        Real rho_clump_code     = n_clump_*pmbp->punit->mu()*pmbp->punit->atomic_mass_unit_cgs/pmbp->punit->density_cgs();
        Real T_clump_code       = T_clump_/pmbp->punit->temperature_cgs();
        Real pres_clump_code    = rho_clump_code*T_clump_code;

        par_for("rad_bondi",DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(int m,int k,int j,int i) {
            Real &xmin = size.d_view(m).x1min;
            Real &xmax = size.d_view(m).x1max;
            int nx1 = indcs.nx1;
            Real x1v = CellCenterX(i-is, nx1, xmin, xmax);

            Real &ymin = size.d_view(m).x2min;
            Real &ymax = size.d_view(m).x2max;
            int nx2 = indcs.nx2;
            Real x2v = CellCenterX(j-js, nx2, ymin, ymax);

            Real &zmin = size.d_view(m).x3min;
            Real &zmax = size.d_view(m).x3max;
            int nx3 = indcs.nx3;
            Real x3v = CellCenterX(k-ks, nx3, zmin, zmax);

            // in cooling_box.cpp, inside the par_for
            Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
            if (rad < 1.0) {
                u0(m,IDN,k,j,i) = rho_clump_code;
                u0(m,IEN,k,j,i) = pres_clump_code / gm1;
            } else {
                u0(m,IDN,k,j,i) = rho_amb_code;
                u0(m,IEN,k,j,i) = pres_amb_code / gm1;
            }
            u0(m,IM1,k,j,i) = 0;
            u0(m,IM2,k,j,i) = 0;
            u0(m,IM3,k,j,i) = 0;
        });
        return;
    }
}

// namespace{
//     //----------------------------------------------------------------------------------------
//     //! \fn void AddUserSrcs()
//     //! \brief Add User Source Terms
//     // NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
//     void AddUserSrcs(Mesh *pm, const Real bdt) {
//         MeshBlockPack *pmbp = pm->pmb_pack;
//         const auto &w0 = pmbp->phydro->w0;
//         auto &u0 = pmbp->phydro->u0;
//         const EOS_Data &eos_data = pmbp->phydro->peos->eos_data;
//         AddCGMHeating(pm,bdt,u0,w0,eos_data);
//         return;
//     }
    
//     void AddCGMHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
//                 const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
//         MeshBlockPack *pmbp = pm->pmb_pack;
//         auto &indcs = pmbp->pmesh->mb_indcs;
//         int is = indcs.is;
//         int ie = indcs.ie;
//         int js = indcs.js;
//         int je = indcs.je;
//         int ks = indcs.ks;
//         int ke = indcs.ke;
//         int nmb1 = pmbp->nmb_thispack - 1;
//         auto size = pmbp->pmb->mb_size;
//         // First passive scalar index
//         int nfluid = pmbp->phydro->nhydro;

//         Real gamma_gas = eos_data.gamma;
//         Real gm1 = gamma_gas - 1.0;
//         Real temp_unit = pmbp->punit->temperature_cgs();

//         Real n_unit = pmbp->punit->density_cgs()/(pmbp->punit->mu()*pmbp->punit->atomic_mass_unit_cgs);

//         Real cooling_unit = pmbp->punit->pressure_cgs()/(pmbp->punit->time_cgs()*n_unit * n_unit);

//         // Choose this threshold in terms of wind mass fraction.
//         // f_wind < 1e-3 means essentially pure CGM.
//         Real wind_threshold = 1e-3;

//         par_for("heating", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
//         KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
//             Real rho = w0(m, IDN, k, j, i);
//             if (rho <= 0.0) {
//             return;
//             }
//             Real temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
//             Real cgm_heating_lambda = ISMCoolFn(temp)/cooling_unit;
//             // Primitive passive scalar = wind mass fraction
//             Real f_wind = w0(m, nfluid, k, j, i);
//             // Guard against numerical overshoots
//             f_wind =Kokkos::fmax(0.0, Kokkos::fmin(1.0, f_wind));
//             if (f_wind < wind_threshold) {
//             u0(m, IEN, k, j, i) += bdt*rho*rho*cgm_heating_lambda;
//             }
//             // if (m == 0 && k == ks && j == js && i == is) {
//             //     Kokkos::printf(
//             //         "rho=%e T=%e fwind=%e lambda=%e\n",
//             //         rho, temp_cgs, f_wind, cgm_heating_lambda);
//             // }
//         });
//         return;
//     }

// }; //namespace
