#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:55:29 2019

@author: DI
"""
import numpy as np
from GA_class_200619 import system
import datetime
from decimal import Decimal
import os


################################# SYSTEM PARAMETERS ###########################
cent_freq = 1.9e12
theta_i = [0]

freq_min = 1.8e12 
freq_max = 2e12
steps = 10000
freq = np.linspace(freq_min, freq_max, steps)
polar = 'p' # 's' or 'p'


################################# DESIGN PARAMETERS ###########################

### library ###

n_Si = 3.417
n_air = 1
n_kapton = 1.77
n_Al2O3 = 3.1
n_Mylar =  1.75 # for Thz , 1.65 for GHz 

library = {'Si':[[1 + 0.1 * n_Si, n_Si], [1e-5, 5e-5]], 
           'air':[[1,1], [300e-6, 400e-6]], 
           'AR':[[1.84867560, 1.84867560], [10e-7, 10e-5]], 
           'Mylar':[[n_Mylar, n_Mylar], [3e-6, 120e-6]],
           'Kapton':[[n_kapton, n_kapton], [0.375e-3, 0.375e-3]], 
           'Silicon wafer':[[n_Si, n_Si], [27.7e-6, 34e-6]],
           'Al2O3': [[n_Al2O3, n_Al2O3], [0.63e-3, 0.63e-3]],
           'SiO2': [[1.97, 1.97], [27.7e-6, 34e-6]]}





""" General architecture of BPF (band pass filter) is (HL)^n HH (LH)^n. 
By repeating the alternating HL layer n times we build a reflector. 
Then the cavity is HH gap, which has thickness d = lambda_n / 2 (which supports
multiple reflectons). 
(HL)^n HH (LH)^n
(LH)^n LL (HL)^n
AR, H, L, H, L, L, H, L, H, AR
air - (HL)^p H - (2L)^q - (HL)^(2p+/-1)H - (2L)^q - (HL)^p H - air
air - H - 2L - HLH - 2L - H - air


possibilities:
'fixed', [n, d]
'alternating', ['material1', 'material2'] - varies only d
'generating', ['material1', 'material2'] - generates random n, d
'pick', ['material1', 'material2'] - pick possibles n, d

air - H - kapton - HLH - kapton - H - air
"""
picking_list = ['Mylar', 'Al2O3', 'air']

### method ###

method = []
#air_layer = ['alternating', ['air']]
#alumina_layer = ['fixed', [n_Al2O3, 0.63e-3]]

c = 2
p = 0
q = 3



method.append( ['generating', ['Si']] )
method.append( ['alternating', ['air']] )
method.append( ['generating', ['Si']] )
method.append( ['alternating', ['air']] )
method.append( ['generating', ['Si']] )
method.append( ['alternating', ['air']] )
method.append( ['generating', ['Si']] )
method.append( ['alternating', ['air']] )
method.append( ['generating', ['Si']] )
method.append( ['alternating', ['air']] )
method.append( ['generating', ['Si']] )
method.append( ['alternating', ['air']] )
method.append( ['generating', ['Si']] )
#method.append( ['generating', ['Si', 'air']] )
#method.append( ['generating', ['Si']] )
#method.append( ['generating', ['Si', 'air']] )
#method.append( ['generating', ['Si']] )
#method.append( ['generating', ['Si', 'air']] )
#method.append( ['generating', ['Si']] )
#method.append( ['generating', ['Si', 'air']] )





#num_layers =  len(method) # number of layers on a singe side of a coating
num_layers = 50
system = system(theta_i, freq, polar, num_layers)

### alternating ###
alternating_list = ['air'] * num_layers

### block function ###
#fit_function = 'block'
#cent_freq = 1.9e12 
#allowed_band = 0.001

### AR function ###
#fit_function = 'AR'
weights = [0.5, 0.5]    # statistical weights for fitness (mean, std)

### custom function ###
fit_function = 'custom'
allowed_band = 0.005
restricting_band = 0.01
cent_threshold = 0.95
side_threshold = 0.05

################################# OPTIMIZATION PARAMETERS #####################
num_optimize = 2
pop_num = [2000, 10, 10]         # number of individuals in population 
num_generations = [2, 100, 10000]    # number of generations     
num_parents = [10, 3, 3]     # number of parents 
change_lim = [0.3, 0.01, 0.01]      # the maximum what a n or d can change


picking_only = False
optimization = False
moth_eye     = True
check_design = False
save         = False

########################### DESIGN PICK FROM THE LIBRARY ######################

if picking_only == True:
    params = system.pick_parameters(method, library, pop_num[0])
    genes_pick, index = system.pick_genes(pop_num[0], params, fit_function,
                   cent_freq, allowed_band, restricting_band, 
                   cent_threshold, side_threshold)
#    system.plotter(genes[0][0:num_layers], genes[0][num_layers:num_layers * 2], 
#                   freq_min, freq_max, pop_num, genes[0][-1])
    param = params[index]
    
################################# OPTIMIZATION ################################

if optimization == True:
    param = system.make_parameters(method, library, alternating_list)
    seed = np.zeros((num_layers * 2 + 1))
    for i in range(num_optimize):
        genes = system.make_genes(pop_num[i], param, fit_function,
                   cent_freq, allowed_band, restricting_band, 
                   cent_threshold, side_threshold, weights)
        if i == 0 and picking_only == True:
            genes[0] = genes_pick[0]
           
        if i > 0:
           genes[0:3] = seed
        new_genes = system.optimization(genes, num_generations[i], pop_num[i], num_parents[i],
                     param, change_lim[i], fit_function, cent_freq, 
                     allowed_band, restricting_band, 
                   cent_threshold, side_threshold, weights)
        seed = new_genes[0:3]
        
    system.plotter(new_genes[0][0:num_layers], new_genes[0][num_layers:num_layers * 2],
                   freq_min, freq_max, allowed_band, cent_freq)
    system.step(new_genes[0][0:num_layers], new_genes[0][num_layers:num_layers * 2])

################################# MOTH EYE DESIGN ################################
if moth_eye == True:
    case = 2
    sides = 2
    num_layers = 50
    n_slab = n_Si
    d_slab = 5e-3
    d_layer = [d_slab / 4] * num_layers
    n, d = system.moth_eye(case, n_slab, d_slab, d_layer, num_layers, sides, cent_freq = 1.2 * 1e12)
    
    system.plotter(n, d, freq_min, freq_max, allowed_band, cent_freq)
    system.step(n, d, 'Moth Eye', num_layers)
################################# DESIGN CHECK ################################
substrate = 'BPF_10_s'
n_HTPE = 1.52
n_H = n_Al2O3
n_AR = 1.77

d_H = 0.63e-3
d_L = 0.625e-3
d_AR = 0.375e-3 

d_Si = 1.13e-05
d_air = 4e-5

if check_design == True:
# actual filter
    n = [n_Al2O3, n_air, n_Al2O3, n_air, n_Al2O3, n_air, n_Al2O3]
    d = [0.63e-3, 3.9e-3, 0.63e-3, 0.65e-3, 0.63e-3, 3.9e-3, 0.63e-3]

    
#    n = [n_Al2O3, n_air, n_Al2O3]
#    d = [0.63e-3, 1.1e-3, 0.63e-3]
    R, T, A = system.RTA(n, d)
    system.plotter(n, d, freq_min, freq_max, allowed_band, cent_freq)

######################### SAVE EXPERIMENT PARAMETERS ##########################

### date and time ###
now = datetime.datetime.now()
date = now.strftime("%d%m")
time = now.strftime("%H%M")
if save == True:
    if not os.path.exists("experiment_simulation/day_{}".format(date)):
        os.makedirs("experiment_simulation/day_{}".format(date))
    
    if optimization == True:
        best_design = new_genes[0]
        exper = 'optimized'
    if optimization == False and picking_only == True:
        best_design = genes_pick[0]
        exper = 'picked'
    if optimization == False and picking_only == False:
        best_design = [n, d]
        exper = 'checked'
        
    file = open("experiment_simulation/day_{}/{}_{}.txt".format(date, substrate, time), '+w')
    file.write("Frequancy range : {} - {} Hz \n".format('%.2e' % Decimal(freq_min), 
                                          '%.2e' % Decimal(freq_max)))
    file.write("Angle of incidence : {} rad \n".format(theta_i))
    file.write("Numbers of layers : {}\n".format(num_layers))
    file.write('Merit function : {} \n'.format(fit_function))
    file.write("Allowed band : {} \n".format(allowed_band))
    if fit_function == 'custom':
        file.write('Restrictive band : {} \n'.format(restricting_band))
        file.write('Central threshhold : {} \n'.format(cent_threshold))
        file.write('Side threshhold : {} \n'.format(side_threshold))
    file.write('\n')
    file.write("Picking only? : {} \n".format(picking_only))
    file.write("Picking sample space : {} \n".format(pop_num[0]))
    file.write("Optimization? : {} \n".format(optimization))
    if optimization == True or picking_only == True:
        file.write('Number of optimization loops : {} \n'.format(num_optimize))
        file.write('Population number : {} \n'.format(pop_num))
        file.write('Number of generations : {} \n'.format(num_generations))
        file.write('\n')
        file.write("Method :\n")
        file.write('{} \n'.format(method))
        file.write('\n')
        file.write("Fitness value : {} \n".format(best_design[-1]))
        file.write('\n')
    file.write("Best design refractive index: \n")
    file.write('{} \n'.format(best_design[0:num_layers]))
    file.write('\n')
    file.write("Best design thickness : \n")
    file.write('{} \n'.format(best_design[num_layers : num_layers * 2]))
    file.write('\n')
    file.write('DESCRIPTION : \n')
    file.close()
    
    
    
## add air if silicon is < 10% 
## allow for the thinest layer or 1 lamnda, don't allow 5 or 3 
## programm should allow for initial guess should be user defined 
## plot error 3d graph that would show the degeneracy of the solution
# matric of t1 and t2 as a grid, every new generation log t1 and t2
# showing explaining whats happening 