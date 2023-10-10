
# if [ $# -eq 0 ] ; then
#     echo "no input parameter, please specify a parameter between threadpool or omp"
#     exit
# fi

# (make -j64 $1 2>&1) | tee compile.log

# if [ "$1" = "threadpool" ] ; then
#   cp lmp_$1 BIN_threadpool_lj
#   cp lmp_$1 BIN_threadpool_eam
# elif [ "$1" = "omp" ] ; then
#   cp lmp_$1 BIN_6tni_singlethread_lj
#   cp lmp_$1 BIN_6tni_singlethread_eam
# fi

(make -j48 threadpool 2>&1) | tee compile.log

mv lmp_threadpool lmp_execute
cp lmp_execute BIN_threadpool_lj
cp lmp_execute BIN_threadpool_eam

# (make -j64 threadpool_bigbig 2>&1) | tee compile.log
# mv lmp_threadpool_bigbig lmp_execute
# cp lmp_execute BIN_opt_weak_scaling_lj
# cp lmp_execute BIN_opt_weak_scaling_eam