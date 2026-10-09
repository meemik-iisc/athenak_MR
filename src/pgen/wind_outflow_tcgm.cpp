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
        Mdot_0, dt_mdot,
        vw_0, dt_vw, r_inj,
        n_cgm0, n_ism0, H_ism, H_cgm, T_cgm,t_tr, dt_ncgm, 
        gamma_gas;
        KOKKOS_INLINE_FUNCTION PgenBh() = default;
    };
    PgenBh pbh;  // Host global
    void AddUserSrcs(Mesh *pm, const Real bdt);
    void AddCGMHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                      const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
    void AddBHGrav(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                      const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
    void AddOutflow(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                      const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
    void WindX1BC(Mesh *pm);
    // Functions Time dependent CGM
    KOKKOS_INLINE_FUNCTION
    static Real N_CGM(const Real t,const  Real t_tr, const Real dt_ncgm, 
                        const Real n_ism0, const Real n_cgm0, const Real H_ism, const Real H_cgm, const Real v_bh) {
        Real n_ism_t = n_ism0*exp((-1*t)/(H_ism/v_bh));
        Real n_cgm_t = n_cgm0*exp((-1*(t-t_tr))/(H_cgm/v_bh));
        Real w          = 0.5*(1.0+tanh((t-t_tr)/dt_ncgm));
        // Real n_cgm = ((1-w)*n_ism_t)+(w*n_cgm_t);
        Real n_cgm = n_ism0*exp(-t/dt_ncgm);
        return n_cgm;
    }
    // Functions Time dependent Mdot
    KOKKOS_INLINE_FUNCTION
    static Real Mdot_t(const Real t, const Real Mdot_0, const Real dt_mdot) {
        return Mdot_0*exp(t/dt_mdot);
    }
    // Functions Time dependent v_wind
    KOKKOS_INLINE_FUNCTION
    static Real Vw_t(const Real t, const Real vw_0, const Real dt_vw) {
        // return vw_0*exp(exp((-1*t)/dt_vw));
        return vw_0;
    }
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

    int nfluid   = pmbp->phydro->nhydro;   

    // get initial parameters from input file
    Real gamma_gas_     = pin->GetReal("hydro", "gamma");
    Real n_cgm0_        = pin->GetReal("problem","n_cgm0");
    Real n_ism0_        = pin->GetReal("problem","n_ism0");
    Real H_ism_         = pin->GetReal("problem","H_ism");
    Real H_cgm_         = pin->GetReal("problem","H_cgm");
    Real T_cgm_         = pin->GetReal("problem","T_cgm");
    Real t_tr_          = pin->GetReal("problem","t_tr");
    Real dt_ncgm_       = pin->GetReal("problem","dt_ncgm");
    Real M_bh_          = pin->GetReal("problem","M_bh");
    Real v_bh_          = pin->GetReal("problem","v_bh");
    Real epsilon_       = pin->GetReal("problem","epsilon");
    Real Mdot_0_        = pin->GetReal("problem","Mdot_0");
    Real dt_mdot_       = pin->GetReal("problem","dt_mdot");
    Real vw_0_          = pin->GetReal("problem","vw_0");
    Real dt_vw_         = pin->GetReal("problem","dt_vw");
    Real r_inj_         = pin->GetReal("problem","r_inj");

    
    pbh.gamma_gas       = gamma_gas_;
    pbh.n_cgm0          = n_cgm0_;
    pbh.n_ism0          = n_ism0_;
    pbh.H_ism           = H_ism_;
    pbh.H_cgm           = H_cgm_;
    pbh.T_cgm           = T_cgm_;
    pbh.t_tr            = t_tr_;
    pbh.dt_ncgm         = dt_ncgm_;
    pbh.M_bh            = M_bh_;
    pbh.v_bh            = v_bh_;
    pbh.epsilon         = epsilon_;
    pbh.Mdot_0          = Mdot_0_;
    pbh.dt_mdot         = dt_mdot_;
    pbh.vw_0            = vw_0_;
    pbh.dt_vw           = dt_vw_;
    pbh.r_inj           = r_inj_;
                 

    Real const &gm1     = gamma_gas_ - 1;  
    Real time           = pmbp->pmesh->time;        
    if (restart) return;

    // Select either Hydro or MHD
    if (pmbp->phydro!=nullptr){

        auto &u0 = pmbp->phydro->u0;
        Real vw_t       = Vw_t(time, vw_0_, dt_vw_);
        Real v_inj      = vw_t/std::sqrt(2);
        Real mdot_t     = Mdot_t(time, Mdot_0_, dt_mdot_);
        Real rho_w      = (mdot_t)/(4*M_PI*SQR(r_inj_)*v_inj);
        Real pres_w     = (rho_w*SQR(v_inj))/gamma_gas_;
        Real n_cgm_t    = N_CGM(time, t_tr_, dt_ncgm_, n_ism0_, n_cgm0_, H_ism_, H_cgm_, v_bh_);
        Real rho_cgm_t  = n_cgm_t*pmbp->punit->mu();
        Real T_cgm_code = T_cgm_/pmbp->punit->temperature_cgs();
        Real pres_code  = rho_cgm_t*T_cgm_code;
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

            if (rad<r_inj_){
                u0(m,IDN,k,j,i) = rho_w;
                u0(m,IM1,k,j,i) = (rho_w*v_inj*x1v)/rad;
                u0(m,IM2,k,j,i) = (rho_w*v_inj*x2v)/rad;
                u0(m,IM3,k,j,i) = (rho_w*v_inj*x3v)/rad;
                u0(m,IEN,k,j,i) = pres_w/gm1 + 0.5*(SQR(u0(m,IM1,k,j,i))+SQR(u0(m,IM2,k,j,i))+SQR(u0(m,IM3,k,j,i)))/u0(m,IDN,k,j,i);
                //Add outflow tracer
                u0(m, nfluid, k, j, i) = rho_w*1.0;
            }
            else{
                u0(m,IDN,k,j,i) = rho_cgm_t;
                u0(m,IM1,k,j,i) = -1*rho_cgm_t*v_bh_;
                u0(m,IM2,k,j,i) = 0.0;
                u0(m,IM3,k,j,i) = 0.0;
                u0(m,IEN,k,j,i) = pres_code/gm1 + 0.5*(SQR(u0(m,IM1,k,j,i))+SQR(u0(m,IM2,k,j,i))+SQR(u0(m,IM3,k,j,i)))/u0(m,IDN,k,j,i);
                //Add tracer = 0 outside
                u0(m, nfluid, k, j, i) = 0.0;
            }
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
        AddOutflow(pm,bdt,u0,w0,eos_data);
        AddCGMHeating(pm,bdt,u0,w0,eos_data);
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
    void AddOutflow(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                const DvceArray5D<Real> &w0, const EOS_Data &eos_data) { //Apply Radial Outflow at all timesteps
        MeshBlockPack *pmbp = pm->pmb_pack;
        auto &indcs = pmbp->pmesh->mb_indcs;
        int is = indcs.is, ie = indcs.ie;
        int js = indcs.js, je = indcs.je;
        int ks = indcs.ks, ke = indcs.ke;
        int nmb1 = pmbp->nmb_thispack - 1;
        auto size = pmbp->pmb->mb_size;

        // number of hydro vars and passive scalars
        int nfluid   = pmbp->phydro->nhydro;
        int nscalars = pmbp->phydro->nscalars;
        Real time    = pmbp->pmesh->time;

        Real gamma_gas_         = pbh.gamma_gas;
        Real Mdot_0_            = pbh.Mdot_0;
        Real dt_mdot_           = pbh.dt_mdot;
        Real vw_0_              = pbh.vw_0;
        Real dt_vw_             = pbh.dt_vw;
        Real r_inj_             = pbh.r_inj;
        Real const &gm1         = gamma_gas_ - 1; 

        Real vw_t       = Vw_t(time, vw_0_, dt_vw_);
        Real v_inj      = vw_t/std::sqrt(2);
        Real mdot_t     = Mdot_t(time, Mdot_0_, dt_mdot_);
        Real rho_w      = (mdot_t)/(4*M_PI*SQR(r_inj_)*v_inj);
        Real pres_w     = (rho_w*SQR(v_inj))/gamma_gas_;
        Real Vol_inj    = (4*M_PI*std::pow(r_inj_,3))/3;    
        Real S_rho      = mdot_t/Vol_inj;
        Real S_e        = (S_rho*SQR(v_inj))*(0.5+(1/(gamma_gas_*gm1))); 

        par_for("outflow", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {

            Real &x1min = size.d_view(m).x1min;
            Real &x1max = size.d_view(m).x1max;
            int nx1     = indcs.nx1;
            Real x1v    = CellCenterX(i-is, nx1, x1min, x1max);

            Real &x2min = size.d_view(m).x2min;
            Real &x2max = size.d_view(m).x2max;
            int nx2     = indcs.nx2;
            Real x2v    = CellCenterX(j-js, nx2, x2min, x2max);

            Real &x3min = size.d_view(m).x3min;
            Real &x3max = size.d_view(m).x3max;
            int nx3     = indcs.nx3;
            Real x3v    = CellCenterX(k-ks, nx3, x3min, x3max);

            Real rad    = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
            if (rad<r_inj_){
                Real dens       = w0(m,IDN,k,j,i);     
                u0(m,IDN,k,j,i) += S_rho*bdt;
                u0(m,IM1,k,j,i) += (S_rho*v_inj*x1v/rad)*bdt;
                u0(m,IM2,k,j,i) += (S_rho*v_inj*x2v/rad)*bdt;
                u0(m,IM3,k,j,i) += (S_rho*v_inj*x3v/rad)*bdt;
                u0(m,IEN,k,j,i) += S_e*bdt;
                //Add tracer
                u0(m,nfluid,k,j,i) = dens*1.0;
            }
        });
        return;
    }
    
    void AddCGMHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
        MeshBlockPack *pmbp = pm->pmb_pack;
        auto &indcs = pmbp->pmesh->mb_indcs;
        int is = indcs.is;
        int ie = indcs.ie;
        int js = indcs.js;
        int je = indcs.je;
        int ks = indcs.ks;
        int ke = indcs.ke;
        int nmb1 = pmbp->nmb_thispack - 1;
        auto size = pmbp->pmb->mb_size;
        // First passive scalar index
        int nfluid = pmbp->phydro->nhydro;

        Real gamma_gas = eos_data.gamma;
        Real gm1 = gamma_gas - 1.0;
        Real temp_unit = pmbp->punit->temperature_cgs();

        Real n_unit = pmbp->punit->density_cgs()/(pmbp->punit->mu()*pmbp->punit->atomic_mass_unit_cgs);

        Real cooling_unit = pmbp->punit->pressure_cgs()/(pmbp->punit->time_cgs()*n_unit * n_unit);

        // Choose this threshold in terms of wind mass fraction.
        // f_wind < 1e-3 means essentially pure CGM.
        Real wind_threshold = 1e-3;

        par_for("heating", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            Real rho = w0(m, IDN, k, j, i);
            if (rho <= 0.0) {
            return;
            }
            Real temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
            Real cgm_heating_lambda = ISMCoolFn(temp)/cooling_unit;
            // Primitive passive scalar = wind mass fraction
            Real f_wind = w0(m, nfluid, k, j, i);
            // Guard against numerical overshoots
            f_wind =Kokkos::fmax(0.0, Kokkos::fmin(1.0, f_wind));
            if (f_wind < wind_threshold) {
            u0(m, IEN, k, j, i) += bdt*rho*rho*cgm_heating_lambda;
            }
            // if (m == 0 && k == ks && j == js && i == is) {
            //     Kokkos::printf(
            //         "rho=%e T=%e fwind=%e lambda=%e\n",
            //         rho, temp_cgs, f_wind, cgm_heating_lambda);
            // }
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
        int nfluid_bc   = pm->pmb_pack->phydro->nhydro;
        int nscalars_bc = pm->pmb_pack->phydro->nscalars;

        Real gamma_gas_ = pbh.gamma_gas;
        Real n_ism0_    = pbh.n_ism0;
        Real n_cgm0_    = pbh.n_cgm0;
        Real H_ism_     = pbh.H_ism;
        Real H_cgm_     = pbh.H_cgm;
        Real v_bh_      = pbh.v_bh;
        Real t_tr_      = pbh.t_tr;
        Real dt_ncgm_   = pbh.dt_ncgm;
        Real T_cgm_     = pbh.T_cgm;

        Real const &gm1 = gamma_gas_ - 1; 
        Real n_cgm_t    = N_CGM(time, t_tr_, dt_ncgm_, n_ism0_, n_cgm0_, H_ism_, H_cgm_, v_bh_);
        Real rho_cgm_t  = n_cgm_t*pmbp->punit->mu();
        Real T_cgm_code = T_cgm_/pmbp->punit->temperature_cgs();
        Real pres_code  = rho_cgm_t*T_cgm_code;


        // ConsToPrim over all X1 ghost zones
        pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, is-ng, ie+ng, 0, n2-1, 0, n3-1);

        // Set outer X1 ghost zones to fixed CGM inflow
        par_for("wind_x1_bc", DevExeSpace(), 0, nmb-1, ks, ke, js, je,
        KOKKOS_LAMBDA(int m, int k, int j) {
            if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
                for (int i = 0; i < ng; ++i) {
                    int ig = ie + i + 1;

                    // Density
                    w0_(m, IDN, k, j, ig) = rho_cgm_t;

                    // Velocity: inflow with -v_bh along x
                    w0_(m, IVX, k, j, ig) = -1*v_bh_;
                    w0_(m, IVY, k, j, ig) = 0.0;
                    w0_(m, IVZ, k, j, ig) = 0.0;

                    // Pressure = pres_cgm
                    w0_(m, IEN, k, j, ig) = pres_code/gm1;
                    //Add tracer = 0
                    if (nscalars_bc > 0) {
                        w0_(m, nfluid_bc, k, j, ig) = 0.0;  // tracer=0 for CGM inflow
                    }
                }
            }
        });

        // PrimToCons on X1 ghost zones
        pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));

        return;
    }

}; //namespace
