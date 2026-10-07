//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file wind_outflow_time_dependent_cgm.cpp
//! \brief Problem generator for time-dependent CGM winds blowing in a stationary BH
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
    struct PgenBh{
        Real M_bh, v_bh, epsilon,
        n_cgm, T_cgm,
        gamma_gas;
        KOKKOS_INLINE_FUNCTION PgenBh() = default;
    };
    PgenBh pbh;  // Host global
    void AddUserSrcs(Mesh *pm, const Real bdt);
    void AddBHGrav(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                      const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
    void WindX1BC(Mesh *pm);
} // namespace
  

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
    user_srcs_func = AddUserSrcs;
     // User boundary function
    user_bcs_func = WindX1BC;
    MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

    // capture variables for the kernel
    auto &indcs = pmbp->pmesh->mb_indcs;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    auto &size = pmbp->pmb->mb_size;

    // get initial parameters from input file
    Real gamma_gas_     = pin->GetReal("hydro", "gamma");
    Real n_cgm_         = pin->GetReal("problem","n_cgm");
    Real T_cgm_         = pin->GetReal("problem","T_cgm");
    Real M_bh_          = pin->GetReal("problem","M_bh");
    Real v_bh_          = pin->GetReal("problem","v_bh");
    Real epsilon_       = pin->GetReal("problem","epsilon");

    
    pbh.gamma_gas       = gamma_gas_;
    pbh.n_cgm           = n_cgm_;
    pbh.T_cgm           = T_cgm_;
    pbh.M_bh            = M_bh_;
    pbh.v_bh            = v_bh_;
    pbh.epsilon         = epsilon_;
                 

    Real const &gm1     = gamma_gas_ - 1;  
    Real time           = pmbp->pmesh->time;        
    if (restart) return;

    // Select either Hydro or MHD
    if (pmbp->phydro!=nullptr){

        auto &u0 = pmbp->phydro->u0;
        Real rho_cgm_   = n_cgm_*pmbp->punit->mu();
        Real T_cgm_code = T_cgm_/pmbp->punit->temperature_cgs();
        Real pres_code  = rho_cgm_*T_cgm_code;
        // std::cout << "rho_cgm = "<<rho_cgm<<",T_code"<<T_cgm_code<<",pres_code = " << pres_code << "\n";

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

            Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
            u0(m,IDN,k,j,i) = rho_cgm_;
            u0(m,IM1,k,j,i) = -1*rho_cgm_*v_bh_;
            u0(m,IM2,k,j,i) = 0.0;
            u0(m,IM3,k,j,i) = 0.0;
            u0(m,IEN,k,j,i) = pres_code/gm1 + 0.5*(SQR(u0(m,IM1,k,j,i))+SQR(u0(m,IM2,k,j,i))+SQR(u0(m,IM3,k,j,i)))/u0(m,IDN,k,j,i);
        });
        return;
    }
}

namespace{
    //----------------------------------------------------------------------------------------
    //! \fn void AddUserSrcs()
    //! \brief Add User Source Terms
    // NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
    void AddUserSrcs(Mesh *pm, const Real bdt) {
        MeshBlockPack *pmbp = pm->pmb_pack;
        const auto &w0 = pmbp->phydro->w0;
        auto &u0 = pmbp->phydro->u0;
        const EOS_Data &eos_data = pmbp->phydro->peos->eos_data;
        AddBHGrav(pm,bdt,u0,w0,eos_data);
        return;
    }
    
    void AddBHGrav(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                const DvceArray5D<Real> &w0, const EOS_Data &eos_data) { //Apply BH Grav at all timesteps
        MeshBlockPack *pmbp = pm->pmb_pack;
        auto &indcs = pmbp->pmesh->mb_indcs;
        int is = indcs.is, ie = indcs.ie;
        int js = indcs.js, je = indcs.je;
        int ks = indcs.ks, ke = indcs.ke;
        int nmb1 = pmbp->nmb_thispack - 1;
        auto size = pmbp->pmb->mb_size;

        Real gamma_gas_         = pbh.gamma_gas;
        Real M_bh_              = pbh.M_bh;
        Real epsilon_           = pbh.epsilon;
        Real G_code             = pmbp->punit->grav_constant();
        Real const &gm1 = gamma_gas_ - 1;

        par_for("bh_gravity", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {

            Real &x1min = size.d_view(m).x1min;
            Real &x1max = size.d_view(m).x1max;
            int nx1 = indcs.nx1;
            Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

            Real &x2min = size.d_view(m).x2min;
            Real &x2max = size.d_view(m).x2max;
            int nx2 = indcs.nx2;
            Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

            Real &x3min = size.d_view(m).x3min;
            Real &x3max = size.d_view(m).x3max;
            int nx3 = indcs.nx3;
            Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

            Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));   
            Real vx     = w0(m,IVX,k,j,i);
            Real vy     = w0(m,IVY,k,j,i);
            Real vz     = w0(m,IVZ,k,j,i);
            Real rho    = w0(m,IDN,k,j,i);

            Real grad_phi_by_r  = (G_code*M_bh_)/(pow((SQR(rad)+SQR(epsilon_)),1.5));
            
            u0(m,IM1,k,j,i) -= rho*grad_phi_by_r*bdt*x1v;
            u0(m,IM2,k,j,i) -= rho*grad_phi_by_r*bdt*x2v;
            u0(m,IM3,k,j,i) -= rho*grad_phi_by_r*bdt*x3v;
            Real v_dot_r    = (w0(m,IVX,k,j,i)*x1v)+(w0(m,IVY,k,j,i)*x2v)+(w0(m,IVZ,k,j,i)*x3v);
            u0(m,IEN,k,j,i) -= rho*v_dot_r*grad_phi_by_r*bdt;
        });
        return;
    }
    
    void WindX1BC(Mesh *pm) {
        MeshBlockPack *pmbp = pm->pmb_pack;
        auto &indcs     = pm->mb_indcs;
        int &ng         = indcs.ng;
        int n1          = indcs.nx1 + 2*ng;
        int n2          = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
        int n3          = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
        int &is         = indcs.is;  int &ie  = indcs.ie;
        int &js         = indcs.js;  int &je  = indcs.je;
        int &ks         = indcs.ks;  int &ke  = indcs.ke;
        auto &mb_bcs    = pm->pmb_pack->pmb->mb_bcs;

        DvceArray5D<Real> u0_, w0_;
        u0_             = pm->pmb_pack->phydro->u0;
        w0_             = pm->pmb_pack->phydro->w0;
        int nmb         = pm->pmb_pack->nmb_thispack;
        int nvar        = u0_.extent_int(1);

        Real time       = pm->pmb_pack->pmesh->time;
        Real gamma_gas_ = pbh.gamma_gas;

        Real n_cgm_     = pbh.n_cgm;
        Real T_cgm_     = pbh.T_cgm;
        Real v_bh_      = pbh.v_bh;

        Real const &gm1 = gamma_gas_ - 1; 
        Real rho_cgm_   = n_cgm_*pmbp->punit->mu();
        Real T_cgm_code = T_cgm_/pmbp->punit->temperature_cgs();
        Real pres_code  = rho_cgm_*T_cgm_code;


        // ConsToPrim over all X1 ghost zones
        pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, is-ng, ie+ng, 0, n2-1, 0, n3-1);

        // Set outer X1 ghost zones to fixed CGM inflow
        par_for("wind_x1_bc", DevExeSpace(), 0, nmb-1, ks, ke, js, je,
        KOKKOS_LAMBDA(int m, int k, int j) {
            if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
                for (int i = 0; i < ng; ++i) {
                    int ig = ie + i + 1;

                    // Density
                    w0_(m, IDN, k, j, ig) = rho_cgm_;

                    // Velocity: inflow with -v_bh along x
                    w0_(m, IVX, k, j, ig) = -1*v_bh_;
                    w0_(m, IVY, k, j, ig) = 0.0;
                    w0_(m, IVZ, k, j, ig) = 0.0;

                    // Pressure = pres_cgm
                    w0_(m, IEN, k, j, ig) = pres_code/gm1;
                    }
            }
        });

        // PrimToCons on X1 ghost zones
        pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));

        return;
    }

}; //namespace
