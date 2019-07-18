#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:55:29 2019

@author: DI
"""
import numpy as np
import random 
from matplotlib import pyplot as plt
from decimal import Decimal
class system():
    """ """
    
    n_air = 1
#    Z_0 = 120 * np.pi
    Z_0 = 299792458 * 4 * np.pi * 1e-7
################################# DIFINE CLASS ################################
    def __init__(self, theta_i, freq, polar, num_layers):
        """
        Args:  
            theta_i (array/lst): Angle od incedence of incoming radiation. 
            freq (array/lst): Frequency range of an incoming radiation.
            
        """
        self.theta_i = theta_i     
        self.freq = freq
        self.lmbd = 299792458 / self.freq # c = 299792458 m/s
        self.num_layers = num_layers
        self.polar = polar
    
################################# REFRACTIVE INDEX ############################   
    
    def refractive_index(self, dielec_const, loss_tan):
        """
        Args:  
            dielec_const (float): Dielectric constant of a material.
            loss_tan (float): Loss tangent of a material.
        Return:
            complex: Refractive index of a medium.
        Note:
            (-) sign instead of (+) in classical.
        """
        return np.sqrt(dielec_const) * np.sqrt(1 - 1j * loss_tan)   

################################# SNELL'S LAW #################################
        
    def Snell(self, th1, n_out):
        """ Snell's Law. 
        Args: 
            th1(list): Incedent angle in the air.
            n_out(complex): Refractive index of transmission medium. 
        Return:
            list: Transmission angle.
        """
        return np.arcsin((self.n_air.real/n_out.real) * np.sin(th1))
    
########################## PROPOGATION CONSTANT ###############################
        
    def propagation_constant(self, th1, n_out):
        """
        Args: 
            th1(list): Incedent angle in the air.
            n_out(complex): Refractive index of transmission medium. 
        Return:
            list: Propagation constant of a medium. 
        """
        return 1j * (2 * np.pi * n_out * np.cos(th1)) / self.lmbd
    
########################## PARALLEL IMPEDANCE #################################
        
    def parallel_load_impedance(self, th1, n_out):
        """
        Args: 
            th1(list): Incedent angle in the air.
            n_out(complex): Refractive index of transmission medium. 
        Return:
            float: Parallel polarized load impedance.
        """
        return self.Z_0 * (n_out**2 - np.sin(th1)**2)**0.5 / n_out**2
    
######################## PERPENDICULAR IMPEDANCE ##############################

    def perpendicular_load_impedance(self, th1, n_out):
        """
        Args: 
            th1(list): Incedent angle in the air.
            n_out(complex): Refractive index of transmission medium. 
        Return:
            float: Perpendicular polarized load impedance.
        """
        return self.Z_0 / (n_out**2 - np.sin(th1)**2)**0.5

############################# MOTH EYE (AR) COATING ###########################
        
    def moth_eye(self, case, n_slab, d_slab, d_layer, num_layers, sides, cent_freq = 1.2 * 1e12):
        """ Calculates effective refractive index of a moth eye coating. 
        Args:
            n_slab(complex): refractive index of a substrate. 
            d_slab(float): thickness of a substrate. 
            d_layer(lst): thicknesses of each layer of a coating
            num_layers(int): number of layers on a single side.
            cent_lmbd(float): central wavelength. Defult value 1.2 THz. 
        Returns:
            n(lst): a list of refractive indeces corresponding to a system of 
            a silicon substrate with layers of coating (one- or both sided). 
            d(lst): a list of thicknesses corresponding to a system of 
            a silicon substrate with layers of coating (one- or both sided).
        """
        cent_lmbd = 299792458 / cent_freq
        eta = np.zeros(num_layers, dtype=complex)
        
        if case == 1:  ## base and frac are fixed
            base_max = cent_lmbd / 4      # diameter of a lower base
            frac = 1 / num_layers         # fixed fraction
            decrement = base_max * frac   # decrement for diameter of consequant layer
            base = base_max               # first layer
            for i in range(num_layers):
                # eta defined for closed pack haxagonal
                eta[i] = base**2 * np.pi / (base_max**2  * 2 * 3**0.5)  
                base -= decrement
                
        elif case == 2: ## first base is fixed, frac is free
            base = []
            base.append(cent_lmbd / 4)
            # create other widths randomly
            new_base = (np.random.uniform(0.1 * cent_lmbd, 
                                          0.6 * cent_lmbd, num_layers - 1))
            for i in range(len(new_base)):
                base.append(new_base[i])
            for i in range(num_layers):
                eta[i] = base[i]**2 * np.pi / (np.max(base)**2  * 2 * 3**0.5)
        
        elif case == 3 or 4: ## all bases are free 
            base = np.random.uniform(0.1 * cent_lmbd, 0.6 * cent_lmbd, num_layers)
            for i in range(num_layers):
                eta[i] = base[i]**2 * np.pi / (np.max(base)**2  * 2 * 3**0.5)
            
        n_eff = eta * n_slab + (1 - eta) * self.n_air
        n = []
        d = []
        # invert the list for the air-si interface 
        n_eff_inv = n_eff[::-1]
        d_layer_inv = d_layer[::-1]
        if sides == 1:
            for i in range(num_layers):
                n.append(n_eff_inv[i])
                d.append(d_layer_inv[i])
            n.append(n_slab)

        if sides == 2:
            for i in range(num_layers):
                n.append(n_eff_inv[i])
                d.append(d_layer_inv[i])
            n.append(n_slab)
            d.append(d_slab)
            for i in range(num_layers):
                n.append(n_eff[i])
                d.append(d_layer[i])
        return n, d

############################ MAKE N,D LIST ####################################

    def nd_list(self, n_slab, d_slab, n_layer, d_layer, num_layers, sides):
        """ When passed a layers stacking only one side  of double 
        """
        n = []
        d = []
        # invert the list for the air-si interface 
        d_layer_inv = d_layer[::-1]
        n_layer_inv = n_layer[::-1]
        if sides == 1:
            for i in range(num_layers):
                n.append(n_layer_inv[i])
                d.append(d_layer_inv[i])
            n.append(n_slab)
        if sides == 2:
            for i in range(num_layers):
                n.append(n_layer_inv[i])
                d.append(d_layer_inv[i])
            n.append(n_slab)
            d.append(d_slab)
            for i in range(num_layers):
                n.append(n_layer[i])
                d.append(d_layer[i])
        return n, d

################################# RTA #########################################
    
    def RTA(self, n, thickness):
        """ 
        Args: 
            n(list): list of refractive indeces of all the layers
            thickness(list): list of thicknesses of all the layers
            polar(str): accepts 'p' for parallel polarization and 's' for perpendicular
            n_load(complex): None by defult for double-sided interface.
                When using one-sided interface, input a refractive index of load 
                material (substrate).
        Return:
            R(matrix): reflectance as a function of incedent angle and frequency
            T(matrix): transmission as a function of incedent angle and frequency
            A(matrix): absorptance as a function of incedent angle and frequency
        """
        # REFRACTION AND TRANSMITTION
        R = np.zeros((len(self.freq), len(self.theta_i)), dtype=complex)
        T = np.zeros((len(self.freq), len(self.theta_i)), dtype=complex)
        
        for i in range(len(self.theta_i)):
        # for each incedence angle 
        # find the components A, B, C and D for LTE matrix  
        # begin with identity matrix
            A = 1 
            B = 0
            C = 0
            D = 1
            # then find layer matrix components 
            for layer in range(len(thickness)):
                d = thickness[layer]
                n_out = n[layer]
                theta_out = self.Snell(self.theta_i[i], n_out)
                y = self.propagation_constant(theta_out, n_out)
                if self.polar == 'p':
                    # TLM impedance
                    Z = self.parallel_load_impedance(self.theta_i[i], n_out)
                    # source impedance 
                    Z_s = self.Z_0 * np.cos(self.theta_i[i])
                    # for 1 sided case load =! sourse 
                    Z_load = self.parallel_load_impedance(self.theta_i[i], n[-1])
                if self.polar == 's':
                    Z = self.perpendicular_load_impedance(self.theta_i[i], n_out)
                    Z_s = self.Z_0 / np.cos(self.theta_i[i])
                    # for 1 sided case load =! sourse. 
                    # Refractive index of substrate is the last element, i.e. n[-1]
                    Z_load = self.perpendicular_load_impedance(self.theta_i[i], n[-1])
                # layer matrix components 
                A_layer = np.cosh(y * d)
                B_layer = Z * np.sinh(y * d)
                C_layer = np.sinh(y * d) / Z
                D_layer = np.cosh(y * d)
                # then temporary values are 
                A_tmp = A * A_layer + B * C_layer
                B_tmp = A * B_layer + B * D_layer
                C_tmp = C * A_layer + D * C_layer 
                D_tmp = C * B_layer + D * D_layer
                # new value for LTE matrix is
                A = A_tmp 
                B = B_tmp
                C = C_tmp
                D = D_tmp
                # compute reflectance and transmittane 
#                if self.sides == 1:
#                    r = (A * Z_load - D * Z_s + B - C * Z_load * Z_s) / \
#                    (A * Z_load + D * Z_s + B + C * Z_load * Z_s)
#                    t = 2 * Z_load / (A * Z_load + D * Z_s + B + C * Z_load * Z_s)
#                if self.sides == 2:
                r = (A * Z_s - D * Z_s + B - C * Z_s * Z_s) / \
                    (A * Z_s + D * Z_s + B + C * Z_s * Z_s)
                t = 2 * Z_s / (A * Z_s + D * Z_s + B + C * Z_s * Z_s)
                R[:, i] = r * np.conjugate(r)
                T[:, i] = t * np.conjugate(t)
        return R, T, 1 - R - T


################################# AR FITNESS ####################################
    
    def AR_fitness(self, R, weights):
        """Calculates fitness value of each individual.
        Args:
            R_generation(lst): list of reflectances for the current generation.
            weights(lst): statistical weights which balance individual 
                contribution of each of the fitness parameters.
        Returns:
            fitness(lst): list of fitness values.
        """
        # mean value of R for each individual
        fit_mean = np.mean(R)  
        # standard deviation of R for each individual
        fit_std = np.std(R)
        # calculate fitness valaue of an individual
        fit = weights[0] * fit_mean + weights[1] * fit_std
        return fit
    
################################# BLOCK FITNESS ####################################
        
    def block_fitness(self, R, cent_freq, allowed_band):  
        change = cent_freq * allowed_band 
        total_sum = 0
        for i in range(len(self.freq)):
            if cent_freq - change < self.freq[i] < cent_freq + change:
                fit = 0
            else:
                fit = 1
            total_sum += (R[i] - fit)**2
        return (total_sum / len(self.freq))**0.5
    
################################# BLOCK FITNESS ####################################
       
        
    def custom_fitness(self, T, cent_freq, allowed_band, restricting_band, 
                       cent_threshold, side_threshold):  
        """ Transmission measure. """
        change1 = cent_freq * allowed_band
        change2 = cent_freq * restricting_band
        total_sum = 0
        for i in range(len(self.freq)):
            if cent_freq - change1 < self.freq[i] < cent_freq + change1:
                if T[i] <= cent_threshold:
                    total_sum = total_sum + (T[i] - cent_threshold)**2
            if self.freq[i] < cent_freq - change2 or self.freq[i] > cent_freq + change2:
                if T[i] >= side_threshold:
                    total_sum = total_sum + (T[i] - side_threshold)**2  
        return (total_sum / len(self.freq))**0.5
        
################################# MAKE PARAMETERS ####################################

    def make_parameters(self, method, library, alternating_list = None):
        """ Makes one set of parameters, which serve as a instruction for 
        generating genes (designs / sets of layers). 
        """
        param = np.zeros((2 * self.num_layers, 2), dtype=complex)
        i = 0
        for layer in range(len(method)):
            if method[layer][0] == 'fixed':
                param[layer] = method[layer][1][0]
                param[layer + self.num_layers] = method[layer][1][1]
            if method[layer][0] == 'alternating':
                material = alternating_list[i]
                min_d = library[material][1][0]
                max_d = library[material][1][1]
                param[layer] = library[material][0][1]
                param[layer + self.num_layers] = [min_d, max_d]
                i += 1
            if method[layer][0] == 'generating':
                material = np.random.choice(method[layer][1])
                min_n = library[material][0][0]
                max_n = library[material][0][1]
                min_d = library[material][1][0]
                max_d = library[material][1][1]
                param[layer] = [min_n, max_n]
                param[layer + self.num_layers] = [min_d, max_d]
            if method[layer][0] == 'pick':
                material = random.choice(method[layer][1])
                min_n = library[material][0][0]
                max_n = library[material][0][1]
                thickness = random.choice(library[material][1])
                min_d = thickness
                max_d = thickness
                param[layer] = [min_n, max_n]
                param[layer + self.num_layers] = [min_d, max_d]
        return param
        
################################# (ONLY) PICK PARAMETERS #######################
    
    def pick_parameters(self, method, library, pop_num):
        """ Makes pop_num sets of parameters by randomly selecting one of predefined 
        materials. Each of them is used individually to produce single gene (design). 
        """
        params = np.zeros((pop_num, 2 * self.num_layers, 2), dtype=complex)
        for param in params:
            for layer in range(len(method)):
                if method[layer][0] == 'pick':
                    material = random.choice(method[layer][1])
                    min_n = library[material][0][0]
                    max_n = library[material][0][1]
                    min_d = library[material][1][0]
                    max_d = library[material][1][-1]
                    param[layer] = [min_n, max_n]
                    param[layer + self.num_layers] = [min_d, max_d]
                else:
                    print('This function only picks predefined materials.')
        return params

    
################################# (ONLY) PICK GENES ############################
    

    def pick_genes(self, pop_num, params, fit_function,
                   cent_freq = None, allowed_band = None, restricting_band = None, 
                   cent_threshold = None, side_threshold= None, weights = None):
        genes = np.zeros((pop_num, 2 * self.num_layers + 1), dtype=complex)
        for i in range(len(params)):
            param = params[i]
            gene = genes[i]
            for layer in range(len(param)):
                if param[layer][0] == param[layer][1]:
                    gene[layer] = param[layer][1]
                else:
                    gene[layer] = np.random.uniform(param[layer][0], param[layer][1])
            R, T, A = self.RTA(gene[0:self.num_layers], gene[self.num_layers:self.num_layers * 2])
            if fit_function == 'block':
#                fitness = self.block_fitness(R, cent_freq, self.freq, allowed_band)
                gene[-1] = self.block_fitness(R, cent_freq, allowed_band)
            if fit_function == 'AR':
                gene[-1] = self.AR_fitness(R, weights) 
            if fit_function == 'custom':
                gene[-1] = self.custom_fitness(T, cent_freq, allowed_band, restricting_band, 
                       cent_threshold, side_threshold)
        # find initial index of the design with the best fitness 
        fitness = []
        for gene in genes:
            fitness.append(gene[-1])
        index = fitness.index(min(fitness))
        genes_sorted = sorted(genes, key = lambda genes: genes[-1])
        return genes_sorted, index
                    

################################# MAKE GENES ####################################
    

    def make_genes(self, pop_num, param, fit_function,
                   cent_freq = None, allowed_band = None, restricting_band = None, 
                   cent_threshold = None, side_threshold= None, weights = None):
        genes = np.zeros((pop_num, 2 * self.num_layers + 1), dtype=complex)
        for gene in genes:
            for layer in range(len(param)):
                if param[layer][0] == param[layer][1]:
                    gene[layer] = param[layer][1]
                else:
                    gene[layer] = np.random.uniform(param[layer][0], param[layer][1])
            R, T, A = self.RTA(gene[0:self.num_layers], gene[self.num_layers:self.num_layers * 2])
            if fit_function == 'block':
                gene[-1] = self.block_fitness(R, cent_freq, allowed_band)
            if fit_function == 'AR':
                gene[-1] = self.AR_fitness(R, weights) 
            if fit_function == 'custom':
                gene[-1] = self.custom_fitness(T, cent_freq, allowed_band, restricting_band, 
                       cent_threshold, side_threshold)
        genes_sorted = sorted(genes, key = lambda genes: genes[-1])
        return genes_sorted
                    

################################# OPTIMIZATION ####################################
    

    def optimization(self, genes_sorted, num_generations, pop_num, num_parents,
                     params, change_lim, fit_function, cent_freq = None, 
                     allowed_band = None, restricting_band = None, 
                   cent_threshold = None, side_threshold= None, weights = None):
       
        fig = plt.figure(num=None, figsize=(10, 4), dpi=100, facecolor='w', edgecolor='k')
#        d_list = np.zeros((2, num_generations), dtype=complex) # !!!!
        for generation in range(num_generations):
            print('GENERATION ', generation)
            # plot best (min) fintess of a generation 
            print('fitness', genes_sorted[0][-1])
            fitness = genes_sorted[0][-1]
            frame1 = fig.add_subplot(111)
            frame1.plot(generation, fitness, '-o')
            frame1.set_title('Best Fitness')
            frame1.set_xlabel('Generation')
            frame1.set_ylabel('Fitness')
            plt.tight_layout()
            parents =  genes_sorted[0:num_parents]
            kids = np.zeros((pop_num - num_parents, 2 * self.num_layers + 1), dtype=complex)
            for kid in kids:
                # choose randomly a parent
                parent = random.sample(list(parents), 1)
                for layer in range(2 * self.num_layers):
                    if params[layer][0]==params[layer][1]:
                        kid[layer]=params[layer][0]
                    else:
                        change = np.random.uniform(1 - change_lim, 1 + change_lim)
                        kid[layer] = parent[0][layer] * change
                        if params[layer][0] >= kid[layer] or kid[layer] >= params[layer][1]:
                           kid[layer]= parent[0][layer]
                R, T, A = self.RTA( kid[0:self.num_layers], 
                                     kid[self.num_layers:self.num_layers * 2])           
                if fit_function == 'block':
                    kid[-1] = self.block_fitness(R, cent_freq, allowed_band)
                if fit_function == 'AR':
                    kid[-1] = self.AR_fitness(R, weights) 
                if fit_function == 'custom':
                    kid[-1] = self.custom_fitness(T, cent_freq, allowed_band, restricting_band, 
                       cent_threshold, side_threshold)
            # initialize new generation
            genes = np.concatenate((kids, parents))
            genes_sorted = sorted(genes, key = lambda genes: genes[-1])
#            d_list[0, generation] = genes_sorted[0][3] # !!!
#            d_list[1, generation] = genes_sorted[0][5] # !!!
        return genes_sorted
    
    
    
    
################################# PLOTTER #####################################

    def plotter(self, n, d, freq_min, freq_max, allowed_band, cent_freq):
        R, T, A = self.RTA(n, d) 
        # for block function
        block = np.zeros((len(self.freq)))
        change = cent_freq * allowed_band 
        for i in range(len(self.freq)):
            if cent_freq - change < self.freq[i] < cent_freq + change:
                block[i] = 0
            else:
                block[i] = 1 
       
        wave = 299792458 / self.freq
        plot = 'freq'
        if plot == 'wave':
            band = wave * 1e3
            cent = 299792458 / cent_freq * 1e3
            band_range = "Wavelength, m"
        elif plot == 'freq':
            band = self.freq
            cent = cent_freq
            band_range = "Frequency, Hz"
        fig = plt.figure(num=None, figsize=(8, 7), dpi=100, facecolor='w', edgecolor='k')
        for i in range(len(self.theta_i)):
            foo = r"$\theta_i$ = {:.2f} $\pi$".format(self.theta_i[i] / np.pi)
            frame1 = fig.add_subplot(311)
            frame1.plot(band, np.abs(R[:, i]), label = foo)
            # plot block function on top 
#            frame1.plot(band, block)
#            frame1.plot((cent, cent), (0, 1), 'k-')
            frame1.set_ylim(0, 1)
            frame1.set_ylabel('Reflectance')
            frame1.set_xlabel('{}'.format(band_range))
            frame1.set_title('Frequency range {} - {}, {} layers'.format('%.2e' % Decimal(freq_min), 
                '%.2e' % Decimal(freq_max), self.num_layers))
            frame2 = fig.add_subplot(312)
            frame2.plot(band, T[:, i])
            frame2.set_ylim(0, 1)
            frame2.set_xlabel('{}'.format(band_range))
            frame2.set_ylabel('Transmittance')
            frame3 = fig.add_subplot(313)
            frame3.plot(band, A[:, i])
            frame3.set_ylim(0,1)
            frame3.set_ylabel('Absorption')
            frame3.set_xlabel('{}'.format(band_range))
        plt.tight_layout()
#        return R, T, A
                
################################# STEP PLOTTER ###############################
    def step(self, n, d, name, num_layers):
        n = list(n)
        new = [0]
        n.insert(0,n[0])
        for i in range(len(d)):
            if i == 0:
                new.append(d[i])
            else:
                val = new[-1] + d[i]
                new.append(val)
        plt.step(new, n)
        plt.xlabel('thickness of a layer')
        plt.ylabel('refractive index')
        plt.title('{}, {} layers'.format(name, num_layers))
         
 
