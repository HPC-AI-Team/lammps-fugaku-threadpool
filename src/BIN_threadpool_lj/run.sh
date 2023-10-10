# (mpiFCCpx utofu_pp_1to1.cpp -o utofu_pp -ltofucom) && (cp utofu_pp output1/) \
#     &&  (cp utofu_pp_pjsub output1/) && (cd output1 && pjsub -s utofu_pp_pjsub)

# (mpiFCCpx utofu_pp.cpp -o utofu_pp -ltofucom) && (cp utofu_pp output/) \
#     &&  (cp utofu_pp_pjsub output/) && (cd output && pjsub -s utofu_pp_pjsub)



# if [ $# -eq 0 ] ; then
#     echo "no input parameter"
#     exit
# fi

if [ ! -d output  ];then
  mkdir output
fi


# if [ "$1" = "clean" ] ; then
#     echo "clean all"
#     rm -rf output/*
#     exit
# fi

outfile="output/60k"

if [ ! -d $outfile  ];then
  mkdir $outfile
fi

outfile="output/1.7M"

if [ ! -d $outfile  ];then
  mkdir $outfile
fi

echo "cd $outfile and run"

outfile="output/60k"
(echo "cd $outfile and run") && (cp lmp_execute $outfile/) &&  (cp lmp_threadpool_pjsub_60k $outfile/) &&  (cp in.threadpool.lj.60k $outfile/) && \
         (cd $outfile && pjsub -s lmp_threadpool_pjsub_60k)
outfile="output/1.7M"
(echo "cd $outfile and run") && (cp lmp_execute $outfile/) &&  (cp lmp_threadpool_pjsub_1.7M $outfile/) &&  (cp in.threadpool.lj.1.7M $outfile/) && \
         (cd $outfile && pjsub -s lmp_threadpool_pjsub_1.7M)
