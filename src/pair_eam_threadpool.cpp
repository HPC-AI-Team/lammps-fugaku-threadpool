// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "pair_eam_threadpool.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "suffix.h"

#include <cmath>

#include "omp_compat.h"
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairEAMThreadpool::PairEAMThreadpool(LAMMPS *lmp) :
  PairEAM(lmp), ThrThreadpool(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
  _first = false;
  for(int i = 0; i < T_THREAD; i++){
    _tfirst[i] = false;
  }
}

/* ---------------------------------------------------------------------- */

void PairEAMThreadpool::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(rho);
    memory->destroy(fp);
    memory->destroy(numforce);
    nmax = atom->nmax;
    memory->create(rho,nthreads*nmax,"pair:rho");
    memory->create(fp,nmax,"pair:fp");
    memory->create(numforce,nmax,"pair:numforce");
  }

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    // thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    if (force->newton_pair)
      thr->init_eam(nall, rho);
    else
      thr->init_eam(atom->nlocal, rho);

    if (evflag) {
      if (eflag) {
        if (force->newton_pair) eval<1,1,1>(ifrom, ito, thr);
        else eval<1,1,0>(ifrom, ito, thr);
      } else {
        if (force->newton_pair) eval<1,0,1>(ifrom, ito, thr);
        else eval<1,0,0>(ifrom, ito, thr);
      }
    } else {
      if (force->newton_pair) eval<0,0,1>(ifrom, ito, thr);
      else eval<0,0,0>(ifrom, ito, thr);
    }

    // thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

void PairEAMThreadpool::compute(int eflag, int vflag, int tid)
{
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  if(!_tfirst[tid]) {
    if(tid == THREAD_NUM) {
      if (atom->nmax > nmax) {
        memory->destroy(rho);
        memory->destroy(fp);
        memory->destroy(numforce);
        nmax = atom->nmax;
        memory->create(rho,nthreads*nmax,"pair:rho");
        memory->create(fp,nmax,"pair:fp");
        memory->create(numforce,nmax,"pair:numforce");
      }
    }
    _tfirst[tid] = true;
    PairEAM::lmp->parral_barrier(12, tid);
  }

  int ifrom, ito;

  const int idelta = 1 + inum / nthreads;
  ifrom = tid * idelta;
  ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;

  if(DEBUG_MSG) utils::logmesg(PairEAM::lmp,"Threadpool::compute tid {} ifrom {} ito {} nthreads  {} flag {} {} nlocal {}\n", 
          tid, ifrom, ito, nthreads, eflag, vflag, atom->nlocal);

  ThrData *thr = fix->get_thr(tid);
  // thr->timer(Timer::START);
  ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

  if (force->newton_pair)
    thr->init_eam(nall, rho);
  else
    thr->init_eam(atom->nlocal, rho);

  if (evflag) {
    if (eflag) {
      if (force->newton_pair) eval<1,1,1>(ifrom, ito, thr);
      else eval<1,1,0>(ifrom, ito, thr);
    } else {
      if (force->newton_pair) eval<1,0,1>(ifrom, ito, thr);
      else eval<1,0,0>(ifrom, ito, thr);
    }
  } else {
    if (force->newton_pair) eval<0,0,1>(ifrom, ito, thr);
    else eval<0,0,0>(ifrom, ito, thr);
  }

  if(DEBUG_MSG) utils::logmesg(PairEAM::lmp,"tid {} eng_vdwl  {} {} \n", 
          tid, thr->eng_vdwl, thr->eng_coul);

  // thr->timer(Timer::PAIR);
  reduce_thr(this, eflag, vflag, thr);
}

template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
void PairEAMThreadpool::eval(int iifrom, int iito, ThrData * const thr)
{
  int i,j,ii,jj,m,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r,p,rhoip,rhojp,z2,z2p,recip,phip,psip,phi;
  double *coeff;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;

  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  double * const rho_t = thr->get_rho();
  const int tid = thr->get_tid();
  const int nthreads = comm->nthreads;

  const int * _noalias const type = atom->type;
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;

  double fxtmp,fytmp,fztmp;

  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // rho = density at each atom
  // loop over neighbors of my atoms

  for (ii = iifrom; ii < iito; ii++) {
    i = ilist[ii];
    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutforcesq) {
        jtype = type[j];
        p = sqrt(rsq)*rdr + 1.0;
        m = static_cast<int> (p);
        m = MIN(m,nr-1);
        p -= m;
        p = MIN(p,1.0);
        coeff = rhor_spline[type2rhor[jtype][itype]][m];
        rho_t[i] += ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
        if (NEWTON_PAIR || j < nlocal) {
          coeff = rhor_spline[type2rhor[itype][jtype]][m];
          rho_t[j] += ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
        }
      }
    }
  }

  // wait until all threads are done with computation
  // PairEAM::lmp->parral_barrier(12, tid);

  // if(DEBUG_MSG) utils::logmesg(PairEAM::lmp,"Threadpool::compute tid {}  finish first barrier \n", tid);

  // communicate and sum densities

  if (NEWTON_PAIR) {
    // reduce per thread density
    // thr->timer(Timer::PAIR);
    data_reduce_thr_threadpool(rho, nall, nthreads, 1, tid, PairEAM::lmp);

    // wait until reduction is complete
    PairEAM::lmp->parral_barrier(12, tid);

    if(tid < TNI_NUM)
      comm->reverse_comm_parral(this, tid); 

    // if(DEBUG_MSG) utils::logmesg(PairEAM::lmp,"Threadpool::compute tid {}  finish reverse_comm_parral \n", tid);


    // wait until master thread is done with communication
    PairEAM::lmp->parral_barrier(12, tid);

    if(tid == THREAD_NUM) {
      comm->reverse_comm_parral_unpack(this);
      comm->c_vcq = (comm->c_vcq + 1) % VCQ_NUM;
    }


    PairEAM::lmp->parral_barrier(12, tid);
    // if(DEBUG_MSG) utils::logmesg(PairEAM::lmp,"Threadpool::compute tid {}  finish reverse_comm_parral_unpack \n", tid);

  } else {
    // thr->timer(Timer::PAIR);
    data_reduce_thr_threadpool(rho, nlocal, nthreads, 1, tid, PairEAM::lmp);

    PairEAM::lmp->parral_barrier(12, tid);
  }

  // fp = derivative of embedding energy at each atom
  // phi = embedding energy at each atom
  // if rho > rhomax (e.g. due to close approach of two atoms),
  //   will exceed table, so add linear term to conserve energy

  // std::string mesg = " pi ";
    
  for (ii = iifrom; ii < iito; ii++) {
    i = ilist[ii];
    p = rho[i]*rdrho + 1.0;
    m = static_cast<int> (p);
    m = MAX(1,MIN(m,nrho-1));
    p -= m;
    p = MIN(p,1.0);
    coeff = frho_spline[type2frho[type[i]]][m];
    fp[i] = (coeff[0]*p + coeff[1])*p + coeff[2];
    if (EFLAG) {
      phi = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];

      // mesg += fmt::format(" {:<10.5g} ", phi);

      if (rho[i] > rhomax) phi += fp[i] * (rho[i]-rhomax);
      e_tally_thr(this, i, i, nlocal, NEWTON_PAIR, scale[type[i]][type[i]]*phi, 0.0, thr);
    }
  }
  // if(DEBUG_MSG && EFLAG) utils::logmesg(PairEAM::lmp," tid {}  {} \n", tid, mesg);

  // wait until all theads are done with computation
  PairEAM::lmp->parral_barrier(12, tid);

  // if(DEBUG_MSG) utils::logmesg(PairEAM::lmp,"Threadpool::compute tid {}  finish second barrier \n", tid);


  // communicate derivative of embedding function
  // MPI communication only on master thread
  if(tid < TNI_NUM)
    comm->forward_comm_parral(this, tid);

  // if(DEBUG_MSG) utils::logmesg(PairEAM::lmp,"Threadpool::compute tid {}  finish forward_comm_parral\n", tid);


  // wait until master thread is done with communication
  PairEAM::lmp->parral_barrier(12, tid);

  if(tid == THREAD_NUM) {
    comm->c_vcq = (comm->c_vcq + 1) % VCQ_NUM;
  }

  // if(DEBUG_MSG) utils::logmesg(PairEAM::lmp,"Threadpool::compute tid {}  finish forward_comm_parral_unpack\n", tid);


  // compute forces on each atom
  // loop over neighbors of my atoms

  for (ii = iifrom; ii < iito; ii++) {
    i = ilist[ii];
    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    itype = type[i];
    fxtmp = fytmp = fztmp = 0.0;
    const double * _noalias const scale_i = scale[itype];

    jlist = firstneigh[i];
    jnum = numneigh[i];
    numforce[i] = 0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutforcesq) {
        ++numforce[i];
        jtype = type[j];
        r = sqrt(rsq);
        p = r*rdr + 1.0;
        m = static_cast<int> (p);
        m = MIN(m,nr-1);
        p -= m;
        p = MIN(p,1.0);

        coeff = rhor_spline[type2rhor[itype][jtype]][m];
        rhoip = (coeff[0]*p + coeff[1])*p + coeff[2];
        coeff = rhor_spline[type2rhor[jtype][itype]][m];
        rhojp = (coeff[0]*p + coeff[1])*p + coeff[2];
        coeff = z2r_spline[type2z2r[itype][jtype]][m];
        z2p = (coeff[0]*p + coeff[1])*p + coeff[2];
        z2 = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];

        recip = 1.0/r;
        phi = z2*recip;
        phip = z2p*recip - phi*recip;
        psip = fp[i]*rhojp + fp[j]*rhoip + phip;
        fpair = -scale_i[jtype]*psip*recip;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;
        if (NEWTON_PAIR || j < nlocal) {
          f[j].x -= delx*fpair;
          f[j].y -= dely*fpair;
          f[j].z -= delz*fpair;
        }

        if (EFLAG) evdwl = scale_i[jtype]*phi;
        if (EVFLAG) ev_tally_thr(this, i,j,nlocal,NEWTON_PAIR,
                                 evdwl,0.0,fpair,delx,dely,delz,thr);
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}

/* ---------------------------------------------------------------------- */

double PairEAMThreadpool::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairEAM::memory_usage();

  return bytes;
}
