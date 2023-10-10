// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
   OpenMP based threading support for LAMMPS
------------------------------------------------------------------------- */

#include "fix_threadpool.h"
#include "thr_data.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "universe.h"
#include "update.h"

#include "pair_hybrid.h"
#include "bond_hybrid.h"
#include "angle_hybrid.h"
#include "dihedral_hybrid.h"
#include "improper_hybrid.h"
#include "kspace.h"

#include <cstring>

#include "omp_compat.h"

#include "suffix.h"

using namespace LAMMPS_NS;
using namespace FixConst;

static int get_tid()
{
  int tid = 0;
#if defined(_OPENMP)
  tid = omp_get_thread_num();
#endif
  return tid;
}

/* ---------------------------------------------------------------------- */

FixThreadpool::FixThreadpool(LAMMPS *lmp, int narg, char **arg)
  :  Fix(lmp, narg, arg),
     thr(nullptr), last_omp_style(nullptr), last_pair_hybrid(nullptr),
     _nthr(-1), _neighbor(true), _mixed(false), _reduced(true),
     _pair_compute_flag(false), _kspace_compute_flag(false)
{
  int nthreads = 1;


  if (getenv("THREAD_POOL_FLAG") != nullptr && atoi(getenv("THREAD_POOL_FLAG")) == 1) {
    nthreads = 12;
  }
  
  if (nthreads != comm->nthreads) {
    comm->nthreads = nthreads;
  }

  if(DEBUG_MSG) utils::logmesg(lmp," FixThreadpool nthreads {} \n", nthreads);



  // allocate list for per thread accumulator manager class instances
  // and then have each thread create an instance of this class to
  // encourage the OS to use storage that is "close" to each thread's CPU.

  thr = new ThrData *[nthreads];
  _nthr = nthreads;

  for(int tid = 0; tid < nthreads; tid++)
  {
    auto t = new Timer(lmp);
    thr[tid] = new ThrData(tid,t);
  }
}

/* ---------------------------------------------------------------------- */

FixThreadpool::~FixThreadpool()
{
  for (int i=0; i < _nthr; ++i)
    delete thr[i];

  delete[] thr;
}

/* ---------------------------------------------------------------------- */

int FixThreadpool::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  mask |= MIN_PRE_FORCE;
  // mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixThreadpool::init()
{

  if(DEBUG_MSG) utils::logmesg(lmp,"FixThreadpool init \n");
  // OPENMP package cannot be used with atom_style template

  // adjust number of data objects when the number of OpenMP
  // threads has been changed somehow
  const int nthreads = comm->nthreads;
  if (_nthr != nthreads) {
    if (comm->me == 0)
      utils::logmesg(lmp,"Re-init threadpool for {} threadpool thread(s)\n", nthreads);

    for (int i=0; i < _nthr; ++i)
      delete thr[i];

    thr = new ThrData *[nthreads];
    _nthr = nthreads;
    for(int i = 0; i < nthreads; i++)
    {
      const int tid = get_tid();
      auto t = new Timer(lmp);
      thr[tid] = new ThrData(tid,t);
    }
  }

  // reset per thread timer
  for (int i=0; i < nthreads; ++i) {
    thr[i]->_timer_active=1;
    thr[i]->timer(Timer::RESET);
    thr[i]->_timer_active=-1;
  }

  if (utils::strmatch(update->integrate_style,"^respa")
      && !utils::strmatch(update->integrate_style,"^respa/omp"))
    error->all(FLERR,"Must use respa/omp for r-RESPA with /omp styles");

  _pair_compute_flag = force->pair && force->pair->compute_flag;
  _kspace_compute_flag = force->kspace && force->kspace->compute_flag;

  int check_hybrid, kspace_split;
  last_pair_hybrid = nullptr;
  last_omp_style = nullptr;
  const char *last_omp_name = nullptr;
  const char *last_hybrid_name = nullptr;
  const char *last_force_name = nullptr;

  // support for verlet/split operation.
  // kspace_split == 0 : regular processing
  // kspace_split < 0  : master partition, does not do kspace
  // kspace_split > 0  : slave partition, only does kspace

  if (strstr(update->integrate_style,"verlet/split") != nullptr) {
    if (universe->iworld == 0) kspace_split = -1;
    else kspace_split = 1;
  } else {
    kspace_split = 0;
  }

// determine which is the last force style with OpenMP
// support as this is the one that has to reduce the forces

#define CheckStyleForOMP(name)                                          \
  check_hybrid = 0;                                                     \
  if (force->name) {                                                    \
    if ( (strcmp(force->name ## _style,"hybrid") == 0) ||               \
         (strcmp(force->name ## _style,"hybrid/overlay") == 0) )        \
      check_hybrid=1;                                                   \
    if (force->name->suffix_flag & Suffix::OMP) {                       \
      last_force_name = (const char *) #name;                           \
      last_omp_name = force->name ## _style;                            \
      last_omp_style = (void *) force->name;                            \
    }                                                                   \
  }

#define CheckHybridForOMP(name,Class) \
  if (check_hybrid) {                                         \
    Class ## Hybrid *style = (Class ## Hybrid *) force->name; \
    for (int i=0; i < style->nstyles; i++) {                  \
      if (style->styles[i]->suffix_flag & Suffix::OMP) {      \
        last_force_name = (const char *) #name;               \
        last_omp_name = style->keywords[i];                   \
        last_omp_style = style->styles[i];                    \
      }                                                       \
    }                                                         \
  }

  if (_pair_compute_flag && (kspace_split <= 0)) {
    CheckStyleForOMP(pair);
    CheckHybridForOMP(pair,Pair);
    if (check_hybrid) {
      last_pair_hybrid = last_omp_style;
      last_hybrid_name = last_omp_name;
    }

    CheckStyleForOMP(bond);
    CheckHybridForOMP(bond,Bond);

    CheckStyleForOMP(angle);
    CheckHybridForOMP(angle,Angle);

    CheckStyleForOMP(dihedral);
    CheckHybridForOMP(dihedral,Dihedral);

    CheckStyleForOMP(improper);
    CheckHybridForOMP(improper,Improper);
  }

  if (_kspace_compute_flag && (kspace_split >= 0)) {
    CheckStyleForOMP(kspace);
  }

#undef CheckStyleForOMP
#undef CheckHybridForOMP
  neighbor->set_omp_neighbor(_neighbor ? 1 : 0);

  // diagnostic output
  if (comm->me == 0) {
    if (last_omp_style) {
      if (last_pair_hybrid)
        utils::logmesg(lmp,"Hybrid pair style last /threadpool style {}\n",last_hybrid_name);
      utils::logmesg(lmp,"Last active /threadpool style is {}_style {}\n",last_force_name,last_omp_name);
    } else {
      utils::logmesg(lmp,"No /threadpool style for force computation currently active\n");
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixThreadpool::setup(int)
{
  // we are post the force compute in setup. turn on timers

  if(DEBUG_MSG) utils::logmesg(lmp,"[info] FixThreadpool::setup _nthr {} \n", _nthr);

  for (int i=0; i < _nthr; ++i)
    thr[i]->_timer_active=0;
}

/* ---------------------------------------------------------------------- */

// adjust size and clear out per thread accumulator arrays
void FixThreadpool::pre_force(int tid = 0)
{
  const int nall = atom->nlocal + atom->nghost;

  double **f = atom->f;
  double **torque = atom->torque;
  double *erforce = atom->erforce;
  double *desph = atom->desph;
  double *drho = atom->drho;

  if(DEBUG_MSG) utils::logmesg(lmp,"FixThreadpool::pre_force tid {} \n", tid);


  thr[tid]->check_tid(tid);
  thr[tid]->init_force(nall,f,torque,erforce,desph,drho);

  _reduced = false;
}

// adjust size and clear out per thread accumulator arrays
void FixThreadpool::post_force(int tid = 0)
{
  // comm->opt_reverse_comm_parrel_unpack();
}

/* ---------------------------------------------------------------------- */

double FixThreadpool::memory_usage()
{
  double bytes = (double)_nthr * (sizeof(ThrData *) + sizeof(ThrData));
  bytes += (double)_nthr * thr[0]->memory_usage();

  return bytes;
}
