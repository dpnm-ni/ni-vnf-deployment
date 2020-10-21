#!/bin/bash

# Debug mode - use 0 for regular operation and i to return only the first i requests.
DBG=0

# Number of CPU cores to use.
NCORES=18

# Topology file.
# Currently available options: inet2, mec, testbed.
TOPOFILE="inet2"

# Distribution of request interarrival time and flow duration.
# Currently available options: exp, norm.
DIST="norm"

# Week identifier, used as seed (one run is typically used to generate
# requests over the span of one week ~ 10k minutes).
WEEK=1
NMINUTES=2

# Suffix used for consistent file naming scheme.
SUFFIX="wk$WEEK""_sfccat_$DIST""_$TOPOFILE"

# File names of generated output files.
REQSRDS="requests-$SUFFIX.rds"
REQSTXT="requests-$SUFFIX.txt"
REQTG="requests-$SUFFIX-tg.txt"

# Compression factors -- note that this also means an increase in the arrival density and decrease of flow durations (the number of active flows remains).
#   week -> hour:   168
#   week -> day:      7
COMPRESSTIME=7

# Control load with these parameters. Avg. number of flows at peak rate = REQSPERMIN / 60 * MEANDURATION.
REQSPERMIN=1
MEANDURATION=1

# Bandwidth range of requests.
MINBW=20
MAXBW=100

# Input CSVs.
SFCCATFILE="sfccat-$SUFFIX.csv"
SFCNODESFILE="sfcnodes-$SUFFIX.csv"
IPMAPFILE="ipmap-$SUFFIX.csv"

# Location of CPLEX log files.
LOGFILEPATH="../solver_src/logs/"


# Run the individual steps from generating the request trace to obtaining solver results.
Rscript 001_generateSFCRTrace.R --rdsOutfile $REQSRDS --txtOutfile $REQSTXT --tgOutfile $REQTG --topoFile $TOPOFILE --nMinutes $NMINUTES --seed 23 --iatDist $DIST --durDist $DIST --compressTime $COMPRESSTIME --reqsPerMin $REQSPERMIN --meanDuration $MEANDURATION --minBandwidth $MINBW --maxBandwidth $MAXBW --sfccatFile $SFCCATFILE --sfcnodesFile $SFCNODESFILE --ipmapFile $IPMAPFILE --dbg $DBG
python3 002_sfcrTrace2CPLEX.py --sfcrTraceFile $REQSTXT --outFilePath $LOGFILEPATH --suffix $SUFFIX
Rscript 006_export_gnn_input_fn_regression.r
python3 sfclist_regression.py
