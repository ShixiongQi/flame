# Flame in Knative (flaik)

## Overview

flaik is a serverless deployment for flame.
The flaik system consists of three sub-systems: flame control plane, Kubernetes, and Knative. 
Knative is an Open-Source solution to build Serverless Applications.

The `flame/flaik` folder contains several scripts to configure and set up the flaik environment.
Thus, the working directory for this guideline is `flame/flaik`.

The flaik env is tested under NSD Cloudlab. NSF Cloudlab provides flexible, scientific infrastructure for research on the future of cloud computing. 

## flaik installation guideline
This guideline is mainly for deploying the aggregator in Flame as a serverless service. Currently Flame provides support for Knative integration. We demonstrate this capability on NSF Cloudlab, using a multi-node cluster. We select the physical node type as xl170 in Utah and set OS image as Ubuntu 20.04. **Note:** As Cloudlab only allocates 16GB disk space by default, please check *Temp Filesystem Max Space* to maximize the disk space configuration and keep *Temporary Filesystem Mount Point* as default (/mydata)

Follow steps below to set up flaik:

* [Creating a multi-node cluster on Cloudlab](01-create-cluster-on-cloudlab.md)
* [Setting up Kubernetes & Knative](02-setup-k8s-kn.md)
* [Setting up Flame](03-setup-flame.md)
