source("common.R")
library(abind)
library(igraph)
library(reticulate)
library(stringr)

### General config + global vars.

topoFileName <- "inet2"
week <- 1

experiment <- sprintf("wk%d_sfccat_norm_%s", week, topoFileName)
outputCSV <- F

# Get physical topology and extract relevant node and edge information.

theTopo <- readTopo(topoFileName)
theTopo.graph <- 
    graph_from_edgelist(as.matrix(theTopo[[4]] %>% select(from, to)), F)
nNodes <- theTopo[[1]][1]
dfTopo <- as.data.frame(theTopo[3])

# List of all shortest paths for each pair of nodes.
allPaths <- 
    expand.grid(to = 1:nNodes, from = 1:nNodes, path = 0)
for (ii in 1:nNodes) {
    curPaths <- all_shortest_paths(graph = theTopo.graph, from = ii)
    for (jj in 1:nNodes) {
        allPaths[(ii - 1) * nNodes + jj,]$path <- curPaths$res[[jj]] %>% as.numeric() %>% list()
    }
}

adjMatrix <- diag(nrow = nNodes, ncol = nNodes)
adjMatrix.bw <- matrix(nrow = nNodes, ncol = nNodes, data = 0)
adjMatrix.delay <- matrix(nrow = nNodes, ncol = nNodes, data = 0)
for (ii in 1:nrow(theTopo[[4]])) {
    adjMatrix[theTopo[[4]]$from[ii], theTopo[[4]]$to[ii]] <- 1
    adjMatrix[theTopo[[4]]$to[ii], theTopo[[4]]$from[ii]] <- 1
    
    adjMatrix.bw[theTopo[[4]]$from[ii], theTopo[[4]]$to[ii]] <- theTopo[[4]]$bw[ii]
    adjMatrix.bw[theTopo[[4]]$to[ii], theTopo[[4]]$from[ii]] <- theTopo[[4]]$bw[ii]
    
    adjMatrix.delay[theTopo[[4]]$from[ii], theTopo[[4]]$to[ii]] <- theTopo[[4]]$delay[ii]
    adjMatrix.delay[theTopo[[4]]$to[ii], theTopo[[4]]$from[ii]] <- theTopo[[4]]$delay[ii]
}


# Only consider a subset of requests.
ridrange <- 1:80000
# pathPrefix <- "example-data/"
pathPrefix <- ""

reqs <- 
    readRDS(sprintf("%srequests-%s.rds", pathPrefix, experiment)) %>%
    mutate(rid = 1:nrow(.)) %>%
    filter(rid %in% ridrange)

solverres <-
    readRDS(sprintf("%ssolver-results-%s.rds", pathPrefix, experiment)) %>%
    filter(rid %in% ridrange)


cat(sprintf("number of request using rds : %s\n", nrow(reqs)))
cat(sprintf("number of solver using rds : %s\n", nrow(solverres)))

############################################################################################## (ver. 20200504)
# SFC request dataframe - requests definition No.1
# reqs.edit <- reqs %>% select(duration, src, dst,traffic, maxlat, sfcid)

reqs.edit <- right_join(reqs, solverres, by=c('rid'='rid')) %>% select(duration.x, src, dst,traffic.x, maxlat, sfcid.x)

reqs.edit[,sprintf("srcNode%d", 1:nNodes)] <- 0
reqs.edit[,sprintf("dstNode%d", 1:nNodes)] <- 0
reqs.edit[,sprintf("sfcid%d", 1:max(reqs$sfcid))] <- 0

for (i in 1:nrow(reqs.edit)){
    reqs.edit[i,sprintf("srcNode%d", reqs.edit$src[i])] <- 1
    reqs.edit[i,sprintf("dstNode%d", reqs.edit$dst[i])] <- 1
    reqs.edit[i,sprintf("sfcid%d", reqs.edit$sfcid.x[i])] <- 1
}

reqs.def.1 <- subset(reqs.edit, select=-src) %>% subset(select=-dst) %>% subset(select=-sfcid.x)
############################################################################################## (ver. 20200504)

# FIXME: only two cases - would significantly increase the size of our matrices otherwise..
if (length(which(solverres$nids1 > 5))) {
    cat("Caution: lowering solverres$nids1 from >5 to 5.\n")
    solverres$nids1[which(solverres$nids1 > 5)] <- 5
}

routeres <-
    readRDS(sprintf("%sroute-results-%s.rds", pathPrefix, experiment)) %>%
    arrange(trafid, rid) %>% 
    filter(trafid %in% ridrange)

noderes <- 
    readRDS(sprintf("%snodecapacity-results-%s.rds", pathPrefix, experiment)) %>% 
    arrange(trafid, node)

nSFCTypes <-
    reqs %>%
    pull(sfcid) %>%
    unique() %>%
    length()

VNFTypes <-
    reqs %>%
    select(starts_with("vnf")) %>%
    unlist() %>%
    na.omit() %>%
    unique() %>%
    sort()

nVNFTypes <-
    VNFTypes %>% 
    length()

# For a given request ID, extract the corresponding GNN input matrices, i.e.,
# node features, graph, and edge features for SFC requests, as well as
# node features, graph, and edge features for the network.
getGNNInput <- function(curRid) {
    
    curReq <-
        reqs %>%
        filter(rid == curRid)
    
    curSolverRes <-
        solverres %>% 
        filter(rid == curRid)
    
    curReqs <- 
        reqs %>% 
        filter(rid %in% curSolverRes$reqlist[[1]])
    
    curRoutes <- 
        routeres %>% 
        filter(trafid == curRid)
    
    ############################################################################################## (ver. 20200506)    
    # Topology -- node features.
    #                                                 전체 cpu core수, used cpu core수, vnftype별 갯수, capacity, remaining capacity, totaltraffic
    nodeFeatures.topo <- matrix(nrow = nNodes, ncol = 1 + 2*6, data = 0)
    # column.names <- c('usedcpu')
    # for(i in 1:nVNFTypes){
    #     column.names <- c(column.names, paste(vnfcat$name[i],"_nVNF", sep=""))
    #     column.names <- c(column.names, paste(vnfcat$name[i],"_capacity", sep=""))
    #     column.names <- c(column.names, paste(vnfcat$name[i],"_remainingcapacity", sep=""))
    #     column.names <- c(column.names, paste(vnfcat$name[i],"_totaltraffic", sep=""))
    #     
    # }
    # colnames(nodeFeatures.topo) <- column.names
    
    noderesCur <- noderes %>% filter(trafid == curRid)
    # First nN nodes represent servers, see for loop below for VNF nodes.
    for (ii in 1:nrow(dfTopo)){
        nodeFeatures.topo[ii, 1] <- dfTopo$cpu[ii]
    }
    
    for (ii in 1:nrow(noderesCur)){
         curRow <- noderesCur[ii,]
         curVNF <- curRow$nf
         #Position of current VNF in VNF list.
         curPos <- which(VNFTypes == curVNF)
         #cat(sprintf("test1 : %s\n", curRow$node))
         #cat(sprintf("test2 : %s\n",  2 + (curPos-1)*2 + 2))
         #nodeFeatures.topo[curRow$node, 2] <- nodeFeatures.topo[curRow$node, 2] + curRow$usedcpu
         #nodeFeatures.topo[curRow$node, 2 + (curPos-1)*2 + 1] <- nodeFeatures.topo[curRow$node, 2 + (curPos-1)*2 + 1] + curRow$ninst
         #nodeFeatures.topo[curRow$node, 2 + (curPos-1)*2 + 2] <- nodeFeatures.topo[curRow$node, 2 + (curPos-1)*2 + 2] + curRow$remainingcapacity
    }
    
    # Topology -- graph.
    graph.topo <- diag(nrow = nNodes, ncol = nNodes)
    graph.topo[1:nNodes, 1:nNodes] <- adjMatrix
   
    
    # Topology -- edge features.
    #                                                     delay,bw
    edgeFeatures.topo <- array(0, dim = c(nNodes, nNodes, 2))
    # Delay of physical links.
    edgeFeatures.topo[1:nNodes,1:nNodes,1] <- adjMatrix.delay
    # Initialize bandwidth with link capacity, determine edge usage, and subtract.
    edgeFeatures.topo[1:nNodes,1:nNodes,2] <- adjMatrix.bw
    # edgeUsage <- reqs2edgeUsage(curReqs, curRoutes, allPaths)
    # for (ii in 1:nrow(edgeUsage)) {
    #     edgeFeatures.topo[edgeUsage[ii, ]$from, edgeUsage[ii, ]$to, 4] <- edgeFeatures.topo[edgeUsage[ii, ]$from, edgeUsage[ii, ]$to, 4] - edgeUsage[ii, ]$traffic
    # }
   
    # Labeling 
    # vnf타입 별 deployment를 이전 단계와과 비교하여 Labeling 
    # 전체 데이터 수 * 노드 수 * VNF Type 수 * 액션 수(3)
    
    deployment.label <- array(0, dim = c(nNodes, nVNFTypes))
    # curRid <- 4
    #current deployment
    deployment.cur <- matrix(nrow = nNodes, ncol = nVNFTypes, data = 0)
    noderesCur <- noderes %>% filter(trafid == curRid)
    # First nN nodes represent servers, see for loop below for VNF nodes.
    for (ii in 1:nrow(noderesCur)){
        curRow <- noderesCur[ii,]
        curVNF <- curRow$nf
        # Position of current VNF in VNF list.
        curPos <- which(VNFTypes == curVNF)
        deployment.cur[curRow$node, curPos] <- deployment.cur[curRow$node, curPos] + curRow$ninst
    }

    #previous deployment
    deployment.prev <- matrix(nrow = nNodes, ncol = nVNFTypes, data = 0)
    curRidIndex <- which(solverres$rid == curRid)
    #curRid가 첫번째가 아닌 경우, previous rid
    if (curRidIndex > 1){
        prevRid <- solverres$rid[curRidIndex-1]
        noderesPrev <- noderes %>% filter(trafid == prevRid)
        # First nN nodes represent servers, see for loop below for VNF nodes.
        for (ii in 1:nrow(noderesPrev)){
            prevRow <- noderesPrev[ii,]
            prevVNF <- prevRow$nf
            # Position of previous VNF in VNF list.
            prevPos <- which(VNFTypes == prevVNF)
            deployment.prev[prevRow$node, prevPos] <- deployment.prev[prevRow$node, prevPos] + prevRow$ninst
        }
    }

    
    #deployment 차이 : current deployment - previous deployment에 따라 Labeling
    deployment.diff <- deployment.cur
    for(i in 1:nrow(deployment.diff)){
        for(j in 1:ncol(deployment.diff)){
            deployment.label[i,j] <-deployment.diff[i,j]
        }

    }
    ############################################################################################## (ver. 20200506)
    
    return(list(
        nodeFeatures.topo,
        graph.topo,
        edgeFeatures.topo,
        deployment.label
    ))
}


# Convert 3D matrix to 2D matrix by stacking.
make2D <- function(mat) {
    return(apply(mat, MARGIN = 2, function(a) as.matrix(a)))
}

if (outputCSV) {
    # FIXME: does not consider the fact that matrices 1-3 should have a shift of one time
    #        step compared to matrices 4-6 (see argument below).
    pb <- txtProgressBar(style = 3)
    
    ## CSV output of individual matrices.
    # Skip first entry since we don't have its predecessor for labeling.
    for (ii in 1:nrow(solverres)) {
        setTxtProgressBar(pb, ii / nrow(solverres))
        curReqID <- solverres$rid[ii]
        curGNNInput <- getGNNInput(curReqID)
        fwrite(curGNNInput[[1]] %>% as.data.table(), sprintf("data/gnn_input/req_%05d_1-nodefeatures_topo.csv", curReqID), col.names = F)
        fwrite(curGNNInput[[2]] %>% as.data.table(), sprintf("data/gnn_input/req_%05d_2-graph_topo.csv", curReqID), col.names = F)
        fwrite(curGNNInput[[3]] %>% make2D() %>% as.data.table(), sprintf("data/gnn_input/req_%05d_3-edgefeatures_topo.csv", curReqID), col.names = F)
        fwrite(curGNNInput[[4]] %>% make2D() %>% as.data.table(), sprintf("data/gnn_input/req_%05d_0-deployment_label.csv", curReqID), col.names = F)
    }
}

## Summarize matrices and export as pickle.
# FIXME: for some reason, reticulate's numpy check requires having used py_save_object first..
py_save_object(array(1:10, dim = c(2, 5)), "tmp/tmp.pickle")
if (reticulate::py_numpy_available()) {

    # Get matrices for first entry to get matrix dimensions.
    gnn1 <- getGNNInput(solverres[2, ]$rid)
    dims <- lapply(gnn1, dim)
    nEntries <- nrow(solverres) - 1
    
    mats1 <- array(0, dim = c(nEntries, dims[[1]]))
    mats2 <- array(0, dim = c(nEntries, dims[[2]]))
    mats3 <- array(0, dim = c(nEntries, dims[[3]]))
    mats4 <- array(0, dim = c(nEntries, dims[[4]]))
    
    pb <- txtProgressBar(style = 3)
    for (ii in 1:nrow(solverres)) {
        setTxtProgressBar(pb, ii / nrow(solverres))
        curReqID <- solverres$rid[ii]
        curGNNInput <- getGNNInput(curReqID)


        if (ii != nrow(solverres)) {
           mats1[ii,,]  <- curGNNInput[[1]]
           mats2[ii,,]  <- curGNNInput[[2]]
           mats3[ii,,,] <- curGNNInput[[3]]
        }
        if (ii != 1) {
            mats4[ii - 1,,]  <- curGNNInput[[4]]
        }
    }

    py_save_object(mats1, filename = sprintf("data/gnn_input/%s-1-nodefeatures_topo.pickle", experiment))
    py_save_object(mats2, filename = sprintf("data/gnn_input/%s-2-graph_topo.pickle", experiment))
    py_save_object(mats3, filename = sprintf("data/gnn_input/%s-3-edgefeatures_topo.pickle", experiment))
    py_save_object(mats4, filename = sprintf("data/gnn_input/%s-0-deployment_label.pickle", experiment))
    
    
} else {
    cat("NumPy not available - no pickle output.\n")
}

reqs.array <- as.matrix(reqs.def.1)
py_save_object(reqs.array, filename = sprintf("data/gnn_input/%s-4-dataframe_req.pickle",experiment))
reqs.colnames <- colnames(reqs.def.1)
cat(reqs.colnames, file="data/gnn_input/reqs.colnames.txt", append = TRUE)
cat(nrow(solverres), file="nsol.txt")
# Example for creating large matrices and pickle export.
# library(abind)
# library(reticulate)
# a1 <- array(1:24, dim = c(2,3,4))
# a2 <- array(25:48, dim = c(2,3,4))
# a3 <- abind(a1, a2, along = 0)
# py_save_object(a3, filename = "test.pickle")


# confirm<-py_load_object("data/gnn_input/wk1_sfccat_norm_testbed-0-deployment_label.pickle", pickle = "pickle")
# confirm[10,,,]
