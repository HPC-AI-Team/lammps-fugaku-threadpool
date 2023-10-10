/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_COMM_BRICK_H
#define LMP_COMM_BRICK_H

#include "comm.h"

namespace LAMMPS_NS {

class CommBrick : public Comm {
 public:
  CommBrick(class LAMMPS *);
  CommBrick(class LAMMPS *, class Comm *);

  ~CommBrick() override;

  void init() override;
  void setup() override;                        // setup 3d comm pattern
  void forward_comm(int dummy = 0) override;    // forward comm of atom coords
  void reverse_comm() override;                 // reverse comm of forces
  void exchange() override;                     // move atoms to new procs
  void borders() override;                      // setup list of atoms to comm

  void borders_one_parral_sendlist() override;                      // setup list of atoms to comm
  void borders_one_parral_xmit(int) override;                      // setup list of atoms to comm
  void borders_one_parral_firstrecv() override;                      // setup list of atoms to comm
  void borders_one_parral_xmit_pos(int) override;                      // setup list of atoms to comm
  void borders_one_parral_finish() override;                      // setup list of atoms to comm

  void forward_comm_parral(int tid) override;                 // forward comm from a Pair
  void reverse_comm_parral(int tid) override;                 // reverse comm from a Pair
  void reverse_comm_parral_unpack() override;                 // reverse comm of forces
  

  void utofu_init() override;

  double box_distance(int*);

  void forward_comm(class Pair *) override;                 // forward comm from a Pair
  void reverse_comm(class Pair *) override;                 // reverse comm from a Pair
  void forward_comm_parral(class Pair *, int tid) override;                 // forward comm from a Pair
  void reverse_comm_parral(class Pair *, int tid) override;                 // reverse comm from a Pair
  void forward_comm_parral_unpack(class Pair *) override;                 // forward comm from a Pair
  void reverse_comm_parral_unpack(class Pair *) override;                 // reverse comm from a Pair

  void forward_comm(class Bond *) override;                 // forward comm from a Bond
  void reverse_comm(class Bond *) override;                 // reverse comm from a Bond
  void forward_comm(class Fix *, int size = 0) override;    // forward comm from a Fix
  void reverse_comm(class Fix *, int size = 0) override;    // reverse comm from a Fix
  void reverse_comm_variable(class Fix *) override;         // variable size reverse comm from a Fix
  void forward_comm(class Compute *) override;              // forward from a Compute
  void reverse_comm(class Compute *) override;              // reverse from a Compute
  void forward_comm(class Dump *) override;                 // forward comm from a Dump
  void reverse_comm(class Dump *) override;                 // reverse comm from a Dump

  void forward_comm_array(int, double **) override;            // forward comm of array
  int exchange_variable(int, double *, double *&) override;    // exchange on neigh stencil
  void *extract(const char *, int &) override;
  double memory_usage() override;

 protected:
  int nswap;                            // # of swaps to perform = sum of maxneed
  int recvneed[3][2];                   // # of procs away I recv atoms from
  int sendneed[3][2];                   // # of procs away I send atoms to
  int maxneed[3];                       // max procs away any proc needs, per dim
  int maxswap;                          // max # of swaps memory is allocated for
  int *sendnum, *recvnum;               // # of atoms to send/recv in each swap
  int *sendproc, *recvproc;             // proc to send/recv to/from at each swap
  int *size_forward_recv;               // # of values to recv in each forward comm
  int *size_reverse_send;               // # to send in each reverse comm
  int *size_reverse_recv;               // # to recv in each reverse comm
  double *slablo, *slabhi;              // bounds of slab to send at each swap
  double **multilo, **multihi;          // bounds of slabs for multi-collection swap
  double **multioldlo, **multioldhi;    // bounds of slabs for multi-type swap
  double **cutghostmulti;               // cutghost on a per-collection basis
  double **cutghostmultiold;            // cutghost on a per-type basis
  int *pbc_flag;                        // general flag for sending atoms thru PBC
  int **pbc;                            // dimension flags for PBC adjustments

  int *firstrecv;        // where to put 1st recv atom in each swap
  int **sendlist;        // list of atoms to send in each swap
  int *localsendlist;    // indexed list of local sendlist atoms
  int *maxsendlist;      // max size of send list for each swap

  double *buf_send;        // send buffer for all comm
  double *buf_recv;        // recv buffer for all comm
  int maxsend, maxrecv;    // current size of send/recv buffer
  int smax, rmax;          // max size in atoms of single borders send/recv

  std::mutex mtx_reverse;

  bool first_init_flag;

  int bin2swap[27][SWAP_NUM];
  int bin2swap_ptr[27];
  
  double bin_split_line[3][2]; 


  int opt_maxdirct;
  int opt_maxswap;                          // max # of swaps memory is allocated for

  uint64_t *opt_sendnum, *opt_recvnum;               // # of atoms to send/recv in each swap 每个方向要发送的原子数量
  uint64_t *opt_forw_pos;
  int **opt_sendlist;        // list of atoms to send in each swap. 二维数组，第一维是方向，第二维是该方向需要传输的原子的下标

  int *remaind_iswap;
  int *opt_firstrecv;
  int *opt_sendproc, *opt_recvproc;             // proc to send/recv to/from at each swap
  int *opt_size_forward_recv;               // # of values to recv in each forward comm
  int opt_size_forward_recv_bcast[27][27];               // # of values to recv in each forward comm
  int *opt_size_reverse_send;               // # to send in each reverse comm
  int *opt_size_reverse_recv;               // # to recv in each reverse comm
  
  int *opt_reverse_send_pos;               // # to send in each reverse comm
  int *opt_forward_send_pos;               // # to send in each reverse comm
  double **opt_slablo, **opt_slabhi;              // bounds of slab to send at each swap
  int *opt_pbc_flag;                        // general flag for sending atoms thru PBC
  int **opt_pbc;                            // dimension flags for PBC 
  double **opt_buf_send[VCQ_NUM];        // send buffer for all comm
  double **opt_buf_recv[VCQ_NUM];        // recv buffer for all comm  

  int *opt_maxsendlist;      // max size of send list for each swap
  int opt_maxforward;

  double **neigh_sublo, **neigh_subhi;

  int *opt_maxsend;
  int *opt_maxrecv;    // current size of send/recv buffer
  MPI_Datatype utofu_comm_type;
  utofu_tni_id_t  tni_id, *tni_ids;




  utofu_vcq_hdl_t             vcq_hdl_send[VCQ_NUM][TNI_NUM],         vcq_hdl_recv[VCQ_NUM][TNI_NUM];;
  utofu_vcq_id_t              lcl_vcq_id_send[VCQ_NUM][TNI_NUM],      lcl_vcq_id_recv[VCQ_NUM][TNI_NUM];  
  struct utofu_onesided_caps *onesided_caps_send[VCQ_NUM][TNI_NUM],   *onesided_caps_recv[VCQ_NUM][TNI_NUM];;

  unsigned long int post_flags;
  utofu_stadd_t *lcl_send_stadd[VCQ_NUM], *lcl_recv_stadd[VCQ_NUM];
  utofu_stadd_t *lcl_send_f_stadd[VCQ_NUM], *lcl_recv_x_stadd[VCQ_NUM];
  Utofu_comm *lcl_comms[VCQ_NUM], *rmt_comms[VCQ_NUM], *lcl_recv_comms[VCQ_NUM];
  uint8_t   *edata;
  uintptr_t *cbvalue;
  uint64_t  *cbvalue_send;
  uint64_t  *edata_send;
  uint64_t  *edata_recvs;


  int directions[62][3];
  int swap_direct[124][3];
  int ndims;


  int con_direction[62][3] = {
    {0, 0, 1},    {0, 1, 0},    {1, 0, 0},
    {0, 1, 1},    {0, -1, 1},    {1, 0, 1},
    {-1, 0, 1},    {1, 1, 0},    {-1, 1, 0},
    {1, 1, 1},    {1, -1, 1},    {-1, 1, 1},
    {-1, -1, 1},  // 13 

    {0, 0, 2},    {0, 2, 0},    {2, 0, 0},  // 3

    {-2, -2, 2}, {-1, -2, 2}, {0, -2, 2}, {1, -2, 2}, {2, -2, 2}, 
    {-2, -1, 2}, {-1, -1, 2}, {0, -1, 2}, {1, -1, 2}, {2, -1, 2},  // 10
    {-2, 0, 2}, {-1, 0, 2},  {1, 0, 2}, {2, 0, 2},  // 4
    {-2, 1, 2}, {-1, 1, 2}, {0, 1, 2}, {1, 1, 2}, {2, 1, 2}, 
    {-2, 2, 2}, {-1, 2, 2}, {0, 2, 2}, {1, 2, 2}, {2, 2, 2},

    {-2, 2, 1}, {-1, 2, 1}, {0, 2, 1}, {1, 2, 1}, {2, 2, 1}, 
    {-2, -2, 1}, {-1, -2, 1}, {0, -2, 1}, {1, -2, 1}, {2, -2, 1},  // 20

    {-2, -1, 1}, {2, -1, 1}, 
    {-2, 0, 1}, {2, 0, 1}, 
    {-2, 1, 1}, {2, 1, 1},  

    {-2, 2, 0}, {-1, 2, 0},  {1, 2, 0}, {2, 2, 0},   
    {-2, 1, 0}, {2, 1, 0}   // 12
  };

  const int send_direction[26] = {
    24,25,22,23,20,21,18,19,
    16,17,14,15,12,13,10,11,8,9,6,7,
    0,1,4,5,2,3
  };
  

  // NOTE: init_buffers is called from a constructor and must not be made virtual
  void init_buffers();
    void init_buffers_value();
  void buildMPIType();

  void warp_utofu_put(utofu_vcq_hdl_t vcq_hdl, utofu_vcq_id_t rmt_vcq_id,
      utofu_stadd_t lcl_send_stadd, utofu_stadd_t rmt_recv_stadd, size_t length,
      uint64_t edata, uintptr_t cbvalue, unsigned long int post_flags,void *piggydata);

  void warp_utofu_poll_tcq(utofu_vcq_hdl_t vcq_hdl, 
              uintptr_t &cbvalue, unsigned long int post_flags);

  void warp_utofu_poll_mrq(utofu_vcq_hdl_t vcq_hdl, 
      uint64_t &edata, unsigned long int post_flags,struct utofu_mrq_notice &in_notice);

  void utofu_recv(utofu_vcq_hdl_t vcq_hdl, uint64_t &edata, unsigned long int post_flags,struct utofu_mrq_notice &in_notice);

  int updown(int, int, int, double, int, double *);
  // compare cutoff to procs
  virtual void grow_send(int, int);       // reallocate send buffer
  virtual void grow_recv(int);            // free/allocate recv buffer
  virtual void grow_list(int, int);       // reallocate one sendlist
  virtual void grow_swap(int);            // grow swap, multi, and multi/old arrays
  virtual void allocate_swap(int);        // allocate swap arrays
  virtual void allocate_multi(int);       // allocate multi arrays
  virtual void allocate_multiold(int);    // allocate multi/old arrays
  virtual void free_swap();               // free swap arrays
  virtual void free_multi();              // free multi arrays
  virtual void free_multiold();           // free multi/old arrays
};

}    // namespace LAMMPS_NS

#endif
