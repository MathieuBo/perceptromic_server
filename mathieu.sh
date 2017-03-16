#!/bin/sh

#############################
# les directives PBS vont ici:

# Your job name (displayed by the queue)
#PBS -N MBPercepPark

# Specify the working directory
#PBS -d ./perceptron_avakas

# walltime (hh:mm::ss)
#PBS -l walltime=06:00:00

# Specify the number of nodes(nodes=) and the number of cores per nodes(ppn=) to be used
#PBS -l nodes=1:ppn=6

# Specify physical memory: kb for kilobytes, mb for megabytes, gb for gigabytes
#PBS -l mem=200mb

#PBS -m abe 
#PBS -M mbourdenx@me.com
# fin des directives PBS
#############################

# modules cleaning
module purge
module add torque
pyenv local 3.5.2 
# For using Python 3.5.2
# module add python3
# module add gcc/4.8.2

# useful informations to print
echo "#############################" 
echo "User:" $USER
echo "Date:" `date`
echo "Host:" `hostname`
echo "Directory:" `pwd`
echo "PBS_JOBID:" $PBS_JOBID
echo "PBS_O_WORKDIR:" $PBS_O_WORKDIR
echo "PBS_NODEFILE: " `cat $PBS_NODEFILE | uniq`
echo "#############################" 

#############################

# What you actually want to launch
echo "Start the job"
python combinations.py > output.txt

# all done
echo "Job finished" 
