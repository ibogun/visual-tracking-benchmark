

Wu. et al (2013), Wu. et al (2015) tracking benchmark
===================================
See original [README](https://github.com/jwlim/tracker_benchmark). Provides functions to run tracker evaluation in parallel.


Running original
--------------------



Usage
- Default (for all trackers, all sequences, all evaltypes(OPE, SRE, TRE))
    - command : python run_trackers.py
    - same with
- For specific trackers, sequences, evaltypes    
    - command : python run_trackers.py -t "tracker" -s "sequence" -e "evaltype"
    - e.g : python run_trackers.py -t IVT,TLD -s Couple,Crossing -e OPE,SRE)


Libraries
- Matlab Engine for python (only needed for executing matlab script files of trackers)

    http://kr.mathworks.com/help/matlab/matlab-engine-for-python.html
- matplotlib
- numpy
- Python Imaging Library (PIL)

    http://www.pythonware.com/products/pil/

Running parallel
-------------------------

Original benchmark evaluation has 2 drawbacks. First, it only evaluates the trackers sequentially. Second, results are saved only in the end. `run_trackers_cached_parallel.py` allows to save evaluation for each video independently; it also allows to run the evaluation in parallel. Syntax is as follows:


    python run_trackers_cached_parallel.py -t RobStruck -e OPE -p 16 # runs evaluation of the RobStruck tracker using OPE protocol with 16 threads

    # it is possible to run more than one tracker

    python run_trackers_cached_parallel.py -t RobStruck,ObStruck,MBestStruck -e OPE -p 16

Limitations
-----------------------
  -   Sequences have to be downloaded either manually or using original benchmark ([original](https://github.com/jwlim/tracker_benchmark)). Once its done a symlink ./data/ -> place where sequences are downloaded would suffice.

  -   Evaluations on SRE,TRE have different parameters than the MATLAB toolbox. This results that during evaluation results needed to be trimmed.
