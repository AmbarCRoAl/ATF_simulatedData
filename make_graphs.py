#In-progress script to sequentally read a collection of h5 monitor data (POP#ID.h5) and 
# return arrays with usable data for machine learning algorithms. 


#IMPORTS
import numpy as np
from math import pi,sin, cos, exp, sqrt
import matplotlib.pyplot as plt
import scipy
import os
import shutil
import h5py as h5
import collections.abc



#GLOBAL VARIABLES ---------------------------------------------------------------
x_variables = ['x']
y_variables = ['y', 'energy']
  # E^2 = (p*c)^2 + (mc^2)^2 and p^2 = px^2 + py^2 + pz^2
input_variables = ['linac1phase', 'linac2phase', 'gunphase', 'solenoidcurrent']
all_x, all_y, all_E = [], [], []

c = 299792458 #m/s
m = 9.1093837e-31 #kg
m = 0.5109989461e6 #eV

# This section is to generate the ID names of the files (pretty similar to generate_simulData.py code)
phase1_optimal = 62/180*pi   #radians
phase2_optimal = 96/180*pi   #radians
gunphase_optimal = 50/180*pi   #radians
current_optimal = 0.206 #amperes
iphase1 = 3      #with ivariabley=x, there will be (x-1)^2+1 variances of the variabley
iphase2 = 3
igunphase = 3
start_step = 0.02 #we will do    x +- start_step*x
phase1_range = []
phase2_range = []
gunphase_range = []
current_range = [0.206,round(0.206-0.206*0.15,3), round(0.206+0.206*0.15,3), round(0.206+2*0.206*0.15,3)]
dx = start_step*phase1_optimal
for i in range(iphase1):
   phase1_range.append(phase1_optimal +dx*i**2)
   if i != 0:
    phase1_range.append(phase1_optimal -dx*i**2)
dx = start_step*phase2_optimal
for i in range(iphase2):
   phase2_range.append(phase2_optimal +dx*i**2)
   if i != 0:
    phase2_range.append(phase2_optimal -dx*i**2)
dx = start_step*gunphase_optimal
for i in range(igunphase):
   gunphase_range.append(gunphase_optimal +dx*i**2)
   if i != 0:
    gunphase_range.append(gunphase_optimal -dx*i**2)

filenames = []
lpops1 = []
hpops1 = []
hpops2 = []
for i in phase1_range:
  for j in phase2_range:
    for k in gunphase_range:
      for l in current_range:
        dataID = str(round(i,3))+"_"+str(round(j,3))+"_"+str(round(k,3))+"_"+str(round(l,3))
        dataID = dataID.replace('.', 'dot')
        lpop1 = "LPOP1f" + dataID + ".h5"
        hpop1 = "HPOP1f" + dataID + ".h5"
        hpop2 = "HPOP2f" + dataID + ".h5"
        lpops1.append(lpop1)
        hpops1.append(hpop1)
        hpops2.append(hpop2)
filenames.append(lpops1)
filenames.append(hpops1)
filenames.append(hpops2)
#print('filenames lenght and lenght of content:', len(filenames), len(filenames[0]))
#print(filenames[0][1])



#FUNCTIONS ----------------------------------------------------------------------
def get_array(h5dataset, keyname):
  x = h5dataset[keyname]
  x = x[:]
  x = x.astype(float)
  return x

def get_energy(px, py, pz):
  if hasattr(px, "__len__"):
    energy = []
    for i in range(len(px)):
      p2 = px[i]**2 + py[i]**2 + pz[i]**2
      energy.append(sqrt(abs(p2*c**2 + m**2*c**4)))
  else:
    p2 = px**2 + py**2 + pz**2
    energy = sqrt(abs(p2*c**2 + m**2*c**4))
  return energy




#DATA HANDLING ------------------------------------------------------------------
count = 0
for i in range(3):
  data = [] #for each file, this will contain [x, y, energy]
  for j in filenames[i]:
    # Read H5 file
    f = h5.File(j, "r")
    # Get list of datasets within the H5 file, for ATF#ID.h5 and POP#ID.h5 file it's the # of steps in the format 'Step#X' were X is an integer >=0
    steps = [n for n in f.keys()]
    len_steps = len(steps)
    step_content = [n for n in f['Step#0'].keys()] #The file should contain at the very least the 1st step
      #string array from 0 to 9 sequentially: id, m, px, py, pz, q, time, x, y, z   (for POP#ID.h5 files)

    for n in steps:
      step = f[n]

      x = step[step_content[7]]
        #the number of datapoints for the sampled POP1f#ID.h5 file was 20000 (len(x)=20000)
      x = x[:]
      x = x.astype(float)
      all_x.append(x)
      y = step[step_content[8]]
      y= y[:]
      y= y.astype(float)
      all_y.append(y)


      px = step[step_content[2]]
      px = px[:]
      px = px.astype(float)
      py = step[step_content[3]]
      py = py[:]
      py = py.astype(float)
      pz = step[step_content[4]]
      pz = pz[:]
      pz = pz.astype(float)
      all_E.append(get_energy(px,py,pz))

      plt.scatter(x,y, facecolors='none', edgecolor='b')
      plt.xlabel("x position")
      plt.ylabel('y position')
      name = 'position_' + str(count)
      plt.savefig(name)
      plt.clf()
      plt.scatter(x,all_E[count], facecolors='none', edgecolor='b')
      plt.xlabel("x position")
      plt.ylabel('y position')
      name = 'energy_' + str(count)
      plt.savefig(name)
      plt.clf()
      count +=1



