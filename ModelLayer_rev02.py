# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:33:32 2020

@author: kevin
"""

# Import modules
import os, glob, inspect, copy, pickle
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.gridspec as gs
from matplotlib.ticker import ScalarFormatter
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.stats import norm, multivariate_normal, truncnorm


class Tree(dict):
    """
    Tree class for easier structure naming/organization.
    
    A tree implementation using python's autovivification feature as described here:
    https://stackoverflow.com/questions/2358045/how-can-i-implement-a-tree-in-python/43237270#43237270
    ______________________________________________________________________
    """
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

    # Cast a (nested) dict to a (nested) Tree class
    def __init__(self, data={}):
        for k, data in data.items():
            if isinstance(data, dict):
                self[k] = type(self)(data)
            else:
                self[k] = data
                
                
class ModelLayer:
    """
    Our model layer class. 
      
    Each instance represents a layer (or line) read from an Excel file containing
    layer attributes (e.g. depth, unit weight, sigma_vo_eff, shear_strength, Vs, etc.)
    
    Methods are defined for calibrating GQ/H+MRDF model parameters to reference data
    for those layer instances, for visualizing, and writing summary output, etc.
    
    """
    # ---------------------Define Class Variables------------------------
    g = 9.81    # gravity [m/s**2]

    # GQ/H Upper/Lower bounds and initial condition for curve-fitting 
    theta_LB    = [-10.0, -5.41, 0.2]
    theta_UB    = [ 0.75,  9.05, 0.99]
    theta_guess = [-0.25,  0.32, 0.7]

    # MRFD Upper/Lower bounds and initial condition for curve-fitting
    mrdf_LB     = [0.415,  0.1,   0.5]
    mrdf_UB     = [0.865,  0.95, 50.0]
    mrdf_guess  = [0.865,  0.4,   4.0]
    
    # Note: The lower and upper bound constraints and initial conditions used for 
    # model calibration were selected for consistency with DEEPSOIL bounds based on 
    # fitting behaviors for sands or clays over a range of depths. 
    
    # Controls range and pt. density for sampling Darendeli/Menq reference models
    ref_data_strain_min_perc = 0.00001
    ref_data_strain_max_perc = 10.0
    ref_data_numb_points     = 50
    
    # ---------------------Define Class Methods---------------------------
    def __init__(self,ini_dict):
        """
        Instance initialization. Define attributes/hierarchy here.
        ______________________________________________________________________
        Inputs
        -------
        ini_dict : initialization dictionary for instance attributes
        """
        # Define instances variables
        self.layer_ID         = ini_dict['layer_ID']
        self.depth            = ini_dict['depth']
        self.thickness        = ini_dict['thickness']
        self.unit_weight      = ini_dict['unit_weight']
        self.sigma_vo_eff     = ini_dict['sigma_vo_eff']
        self.shear_strength   = ini_dict['shear_strength']
        self.shear_cov        = ini_dict['shear_cov']
        self.corr             = ini_dict['corr']
        self.shear_str_min    = ini_dict['shear_str_min']
        self.shear_str_max    = ini_dict['shear_str_max']
        self.distribution     = ini_dict['distribution']
        self.Vs               = ini_dict['Vs']
        self.PI               = ini_dict['PI']
        self.OCR              = ini_dict['OCR']
        self.Ko               = ini_dict['Ko']
        self.Cu               = ini_dict['Cu']
        self.d50              = ini_dict['d50']       #[mm]
        self.material         = ini_dict['material']
        self.memo             = str(ini_dict['memo'])
        #self.Gmax             = self.unit_weight/self.g * self.Vs**2
        #self.gamma_ref        = self.shear_strength/self.Gmax
        
        # Initialize fitting parameters
        self.data = Tree()
        self.data['fit']['theta'] = None   # Should be a dict when assigned
        self.data['fit']['mrdf']  = None   # Should be a dict when assigned
        self.data['reference']    = None   # Should be a pandas dataframe when assigned
        
        if np.max(np.abs(self.corr)) > 1:
            print("Correlation coefficient should be between -1 and 1")
        
        return
    
    
    def Gmax(self):
        """
        Returns small strain modulus.
        ______________________________________________________________________

        """
        return self.unit_weight/self.g * self.Vs**2
    
    def gamma_ref(self):
        """
        Reference shear strain (true; non-psuedo).
        ______________________________________________________________________

        """
        return self.shear_strength/self.Gmax()

    
    def assign_reference_data(self, ReferenceData):
        """
        Assign a reference material data for calibration to the layer
        ReferenceData should be a pandas data frame and contain the following
        key/column fields: 'strain(%)','G/Gmax','damping(%)'
        ______________________________________________________________________
        Inputs
        -------
        ReferenceData : dataframe containing 
        """
        if isinstance(ReferenceData, pd.DataFrame):
            if all([field in ReferenceData for field in ('strain(%)','G/Gmax','damping(%)')]):
                self.data['reference']  = ReferenceData
            else:
                print("Reference data does not contain proper column fields: {'strain(%)','G/Gmax','damping(%)'}.")
        else:
            print("Reference data is not a dataframe.")
        return
    
    def calibrate_layer_params(self,min_strain_perc=0.0001,max_strain_perc=0.05,min_strain_perc_mrdf=0.00,echo=True,display=True):
        """
        Fast calibration of GQ/H & MRDF
        Performs GQ/H fitting (for modulus reduction) and MRDF fitting (for damping)
        ______________________________________________________________________
        Inputs
        -------
        min_strain_perc : minimum strain (%) for fitting GQ/H
        max_strain_perc : maximum strain (%) for fitting GQ/H
        min_strain_perc_mrdf : minimum strain (%) for fitting MRDF
        echo : echo calibration results
        display : visualize calibration

        Returns
        -------
        
        """
        if echo:
            print("""Step 1: GQ/H Model Calibration""")
        self.fit_modulus_reduction(min_strain_perc=min_strain_perc,
                                   max_strain_perc=max_strain_perc,
                                   echo=echo,
                                   display=display)
        
        if echo:
            print("""Step 2: MRDF Calibration""")
        self.fit_damping_MRDF(min_strain_perc=min_strain_perc_mrdf,
                              echo=echo,
                              display=display)
        return    

    def fit_modulus_reduction(self,min_strain_perc=0.0001,max_strain_perc=0.05,echo=False,display=True):
        """
        Performs modulus reduction fitting per Groholski et al. (2016).
        Updates the instance theta parameters. 
        ______________________________________________________________________
        Inputs
        -------
        min_strain_perc : minimum strain (%) for fitting
        max_strain_perc : maximum strain (%) for fitting
        echo : if True, echos fit parameters
        display : if True, displays fit parameters

        Returns
        -------
        
        """
        # Layer calcs
        gamma_r     = self.gamma_ref()
        ref_strain  = self.data['reference']['strain(%)']/1e2
        ref_modulus = self.data['reference']['G/Gmax']
        
        # Compute theta_tau associated with reference data
        theta_tau_ref   = self.fcn_theta_tau(ref_strain,gamma_r,ref_modulus)
        
        # Fitting data
        x_data = (ref_strain/gamma_r).to_numpy()
        y_data = theta_tau_ref.to_numpy()

        # Adjusting fititng data for min/max bounds
        cond1 = x_data > min_strain_perc/1e2/gamma_r
        cond2 = x_data < max_strain_perc/1e2/gamma_r
        x_data_final = x_data[cond1*cond2]
        y_data_final = y_data[cond1*cond2]

        # Obj fcn for curve-fitting
        def func(X, theta_1,theta_2,theta_5):
            """
            Per Groholski et al. (2016), when using curve-fitting algorithms, 
            assume (theta_3, theta_4) = 1.0 
            
            Note that the parameters must satisfy the constraint:
            theta_tau <= 1.0
            
            X=gamma/gamma_r
            """
            theta_3 = 1.0
            theta_4 = 1.0
            num = theta_4*(X)**theta_5
            denom = theta_3**theta_5 + theta_4*(X)**theta_5
            theta_tau_fit = theta_1 + theta_2*num/denom
            return theta_tau_fit 

        # Define arbitrary upper/lower bounds and initial guess, then fit function
        lower_bound = self.theta_LB
        upper_bound = self.theta_UB
        ini_guess   = self.theta_guess
        popt, pcov  = curve_fit(func, x_data_final,
                                y_data_final,
                                p0=ini_guess,
                                bounds=(lower_bound,upper_bound),
                                method="trf" ) # Other methods include: {'lm','trf','dogbox'}


        # An adhoc check for the constraint on theta_tau < 1 over 0 to 10% strain;
        # increments theta_2 by *0.99 every iteration until theta_tau < 0.99;
        # needed since sometimes theta_tau can be very slightly greater than 1 
        # after fitting;
        cond = True
        while cond:
            theta_tau_check = func(np.logspace(np.log10(0.0000001),np.log10(0.1)) /gamma_r, popt[0],popt[1],popt[2]) 
            #theta_tau_check = func(np.linspace(0,0.10)/gamma_r, popt[0],popt[1],popt[2]) 
            if (theta_tau_check > 0.99).any():
                popt[1] = popt[1]*0.99
            else:
                cond = False

        # Pack thetas into a dictionary
        thetas = {'1':popt[0],
                  '2':popt[1],
                  '3':1.0,
                  '4':1.0,
                  '5':popt[2]}
        
        # Save to instance
        self.data['fit']['theta'] = thetas

        # Echo the curve-fit parameters
        if echo:
            print("Estimated Curve-Fitting Parameters:")
            for key in thetas.keys():
                print("\tTheta_{}: {:5.4f}".format(key,thetas[key]))
        
        # Display the modulus reduction
        if display:
            self.plot_modulus_fit()
        return  
    
    def fit_damping_MRDF(self,min_strain_perc=0.00,echo=True,display=True):
        """
        Performs modulus reduction fitting per Phillips and Hashash (2009). 
        Updates the instance MRDF parameters.
        ______________________________________________________________________
        Inputs
        -------
        min_strain_perc : minimum strain level (%) for fitting large strain
                          damping behavior
        echo : if True, echoes fit coefficients
        display : if True, displays fit results

        Returns
        -------
        """
        # Layer calcs
        ref_strain = self.data['reference']['strain(%)']/1e2
        gamma_r    = self.gamma_ref()
        thetas     = self.data['fit']['theta']
        tau_max    = self.shear_strength
        Gmax       = self.Gmax()
        
        
        # Define the modulus ratios at reference strain values (X-data)
        G_gamma_max = self.GQH_F_BB(ref_strain,gamma_r,thetas,tau_max)/ref_strain
        Gratio      = G_gamma_max/Gmax
        
        # Calculate the masing damping (set masing=True), and MRDF factors (Y-data)
        damping_perc_masing = self.calculate_damping(masing=True, 
                                                     Np=5000,
                                                     echo=False,
                                                     display=False)
        F_MRDF_ref = self.data['reference']['damping(%)']/damping_perc_masing

        # Apply the minimum strain (%) mask
        mask = ref_strain > min_strain_perc/1e2
        x_data_final = Gratio[mask]
        y_data_final = F_MRDF_ref[mask]

        # Obj fcn for curve-fitting
        def func(G_ratio, p1,p2,p3):
            """
            Groholski et al. (2016), Eq. 23
            """
            return p1 - p2*((1.-G_ratio)**p3)

        
        # Define search bounds, initial guess, and fit the parameters
        lower_bound = self.mrdf_LB
        upper_bound = self.mrdf_UB
        ini_guess   = self.mrdf_guess

        popt, pcov  = curve_fit(func,
                                x_data_final,
                                y_data_final,
                                p0=ini_guess,
                                bounds=(lower_bound,upper_bound),
                                method="trf") # Other methods: {'lm','trf','dogbox'}

        
        # Update intsance data
        self.data['fit']['mrdf'] = {'1':popt[0],
                                    '2':popt[1],
                                    '3':popt[2]}
        
        # My fit MRDF F(gamma_max) factors from fit p-coefficients
        F_MRDF_MyFit = self.MRDF_factor(Gratio,self.data['fit']['mrdf'])
        
        # True GQ/H + MRDF hysteretic damping
        damping_perc_mrdf = self.calculate_damping(masing=False,
                                                   Np=5000,
                                                   echo=False,
                                                   display=False)
        
        if echo:
            print("My Fit - MRDF Coefficients:")
            print("""\tp1 = {}\n\tp2 = {}\n\tp3 = {}
                  """.format(popt[0],popt[1],popt[2]))
        
        if display:
            fig = plt.figure(figsize=(6,4))
            plt.semilogx(ref_strain*1e2,
                         F_MRDF_ref,
                         'ko',fillstyle='none',ms=7.5,
                         label='Reference')
            plt.semilogx(ref_strain*1e2,
                         F_MRDF_MyFit,
                         'b-',fillstyle='none',ms=7.5,
                         label='QG/H+MRDF (My Fit)')
            plt.title('Fit of MRDF Reduction Factors')
            plt.xlabel('Shear strain, $\gamma$ (%)')
            plt.ylabel('Reduction factor for\n hysteretic damping, $F(\gamma)$')
            plt.ylim(top=2)
            plt.legend()
            plt.show()
            
            
            fig = plt.figure(figsize=(6,4))
            plt.semilogx(ref_strain*1e2,
                         self.data['reference']['damping(%)'],
                         'ko',fillstyle='none',ms=7.5,
                         label='Reference')
            plt.semilogx(ref_strain*1e2,
                         damping_perc_masing,
                         'r-',fillstyle='none',ms=7.5,
                         label='GQ/H+Masing')
            plt.semilogx(ref_strain*1e2,
                         F_MRDF_MyFit*damping_perc_masing,
                         'b-',fillstyle='none',ms=7.5,
                         label='GQ/H+MRDF (Fit)')
            plt.title('Comparison of Damping Ratios')
            plt.xlabel('Shear strain, $\gamma$ (%)')
            plt.ylabel(r'Damping ratio, $\xi$ (%)')
            plt.legend()
            plt.show()
        return
    
    def calculate_damping(self,masing=False,Np=5000,echo=True,display=True):
        """
        Calculates the damping ratio values at the reference material strain levels 
        using the GQ/H+MRDF model.
        ______________________________________________________________________
        Inputs
        -------
        masing : if true, uses standard masing unload-reload; else uses instance GQ/H+MRDF 
                 coefficients
        Np : number of points
        echo : if true, echos (strain level, damping ratio)
        display : if true, plots element results

        Returns
        -------
        """
        ref_strain = self.data['reference']['strain(%)']/1e2
        

        # Run element tests and compute damping ratios
        damping_perc = np.zeros(len(ref_strain))
        for i, strain_level in enumerate(ref_strain):
            _, damp, hys = self.single_element_test(masing=masing,
                                                    target_strain_perc=strain_level*1e2,
                                                    Np=Np,
                                                    display=display)
            damping_perc[i] = damp
            
            if echo:
                print("At strain = {:.2e}%, damping = {:4.2f}%".
                      format(strain_level*100,damp))
        return damping_perc
    
    def single_element_test(self,target_strain_perc=0.1,masing=False,Np=5000,display=True):
        """
        Performs a single element test at specified shear strain level
        (ie., gamma_rev or gamma_max)
        ______________________________________________________________________
        Inputs
        -------
        masing : if true, uses standard masing unload-reload; else uses GQ/H+MRDF coefficients
        target_strain_perc : target strain level
        Np : number of points
        display : if true, plots element results

        Returns
        -------
        """
        gamma_r = self.gamma_ref()
        tau_max = self.shear_strength
        thetas  = self.data['fit']['theta']
        p_dict  = self.data['fit']['mrdf']
        Gmax    = self.Gmax()
        
        if masing:
            p_dict = {'1':1,
                      '2':0,
                      '3':0}
            
        # First unloading
        x_strain1 = np.linspace(target_strain_perc/1e2,
                               -target_strain_perc/1e2,
                                Np)
        gamma_max = np.abs(target_strain_perc/1e2)
        G_gamma_max = self.GQH_F_BB(gamma_max,gamma_r,thetas,tau_max)/gamma_max
        gamma_rev = x_strain1[0] # Get last reversal point
        tau_rev = self.GQH_F_BB(gamma_max,gamma_r,thetas,tau_max)

        X = np.abs(x_strain1-gamma_rev)/2
        y_unload = self.GQH_F_URL_MRDF(X,gamma_r,tau_rev,-tau_max,thetas,p_dict,G_gamma_max,Gmax)

        
        # First reloading
        x_strain2 = np.linspace(-target_strain_perc/1e2,
                                 target_strain_perc/1e2,
                                Np)
        gamma_max = np.abs(target_strain_perc/1e2)
        G_gamma_max = self.GQH_F_BB(gamma_max,gamma_r,thetas,tau_max)/gamma_max
        gamma_rev = x_strain2[0]  # Get last reversal point
        tau_rev = y_unload[-1]

        X = np.abs(x_strain2-gamma_rev)/2
        y_reload = self.GQH_F_URL_MRDF(X,gamma_r,tau_rev,tau_max,thetas,p_dict,G_gamma_max,Gmax)

        
        # Get the max strain level and secant shear modulus for
        # elastic energy calculations
        gamma_max = np.amax(np.abs(x_strain2))
        G_gamma_max = self.GQH_F_BB(gamma_max,gamma_r,thetas,tau_max)/gamma_max

        # Calculate damping ratio
        xi_perc = self.damping_ratio(x_strain2,y_reload,y_unload,gamma_max,G_gamma_max)

        # Save hysteretic response
        hys = {'x_unload':x_strain1,
               'x_reload':x_strain2,
               'y_unload':y_unload,
               'y_reload':y_reload,
                }
    
        if display:
            fig = plt.figure(figsize=(6,4))
            plt.plot(np.concatenate((hys['x_unload'],hys['x_reload']))*1e2,
                     np.concatenate((hys['y_unload'],hys['y_reload']))/1e3,
                     'b-',
                     label='Virgin')
            plt.xlabel(r'Shear strain, $\gamma$ (%)')
            plt.ylabel(r'Shear stress, $\tau$ (kPa)')
            plt.title('GQ/H Model + MRDF (Modified Masing Unload-Reload)')
            plt.legend()
            plt.show()    
        return (target_strain_perc,xi_perc,hys)
    
    def plot_damping(self):
        """
        Plot strain-damping curves
        ______________________________________________________________________
        Inputs
        -------
        Returns
        -------
        """
        fig = plt.figure(figsize=(6,4))
        plt.semilogx(self.reference_data['strain(%)'],
                     self.reference_data['damping(%)'],
                     'bo-',
                     label='Reference')
        plt.ylabel('$G/G_{max}$')
        plt.xlabel('Shear Strain $\gamma$ (%)')
        plt.legend()
        plt.show()
        return
    
    def plot_modulus_reduction(self):
        """
        Plot strain-modulus reduction curves
        ______________________________________________________________________
        Inputs
        -------
        Returns
        -------
        """
        fig = plt.figure(figsize=(6,4))
        plt.semilogx(self.data['reference']['strain(%)'],
                     self.data['reference']['G/Gmax'],
                     'bo-',
                     label='Reference')
        plt.ylabel('$G/G_{max}$')
        plt.xlabel('Shear Strain $\gamma$ (%)')
        plt.legend()
        plt.show()
        return
    
    def plot_modulus_fit(self):
        """
        Plot modulus fit
        ______________________________________________________________________
        Inputs
        -------
        Returns
        -------
        """
        gamma_r     = self.gamma_ref()
        ref_strain  = self.data['reference']['strain(%)']/1e2
        ref_modulus = self.data['reference']['G/Gmax']
        thetas      = self.data['fit']['theta']
        tau_max     = self.shear_strength
        Gmax        = self.Gmax()
        
        theta_tau_ref   = self.fcn_theta_tau(ref_strain,gamma_r,ref_modulus)
        
        theta_tau_fit = self.fcn_theta_tau_fitting(ref_strain,gamma_r,thetas)
        shear_ratio_fit = self.fcn_shear_ratio(ref_strain,gamma_r,theta_tau_fit)
        G_ratio_fit = (shear_ratio_fit*tau_max/ref_strain)/Gmax
        
        
        # Marker/line styling
        ms = 10
        marker_style1 = dict(color='tab:blue', linestyle='', marker='o',
                            markersize=ms, fillstyle='full')
        marker_style2 = dict(color='tab:red', linestyle='--', marker='^',
                            markersize=ms, fillstyle='none')
        marker_style3 = dict(color='green', linestyle='--', marker='s',
                            markersize=ms, fillstyle='none')
        marker_style4 = dict(color='black', linestyle='--', marker='',
                            markersize=ms, fillstyle='none')

        # Setup figure/axes
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,4))
        
        # Plot reference data
        ax1.semilogx(ref_strain*1e2,
                     ref_modulus,
                     **marker_style1,
                     label='Reference')

        ax2.semilogx(ref_strain*1e2/gamma_r,
                     theta_tau_ref,
                     **marker_style1,
                     label='Reference')

        ax3.semilogx(ref_strain*1e2,
                     ref_modulus*Gmax*ref_strain/1e3,
                     **marker_style1,
                     label='Reference')
        
        # Plot the fit
        ax1.semilogx(ref_strain*1e2,
                     G_ratio_fit,
                     **marker_style2,
                     label='GQ/H (My Fit)')

        ax2.semilogx(ref_strain*1e2/gamma_r,
                     theta_tau_fit,
                     **marker_style2,
                     label='GQ/H (My Fit)')

        ax3.semilogx(ref_strain*1e2,
                     shear_ratio_fit*tau_max/1e3,
                     **marker_style2,
                     label='GQ/H (My Fit)')
        
        # Max shear strength line
        ax3.semilogx(ax3.get_xlim(),
                     np.ones(2)*tau_max/1e3,
                     **marker_style4,
                     label=r'$\tau_{max}$')

        ax1.set(xlabel=r'Shear strain, $\gamma$ (%)',
                ylabel=r'$G/G_{max}$')
        ax2.set(xlabel=r'Normalized shear strain, $\gamma/\gamma_{r}$',
                ylabel=r'$\theta_{\tau}$')
        ax3.set(xlabel=r'Shear strain, $\gamma$ (%)',
                ylabel=r'$\tau$ (kPa)')
        ax1.grid('on',which='major',color='lightgray')
        ax2.grid('on',which='major',color='lightgray')
        ax3.grid('on',which='major',color='lightgray')
        #ax1.set_ylim(bottom=0,top=1)
        ax3.legend()
        plt.show()

        return
    
    def save_pdf_output(self,path=None,display=True):
        """
        Generates a plot frame for pdf output
        ______________________________________________________________________
        Inputs
        -------
        path : path to save pdf
        display : if True, shows plot in console before saving
        
        Returns
        -------
        """
        
        fname = """Layer_{}_calibration_summary.pdf""".format(
            str(self.layer_ID))
        if path:
            fname = os.path.join(path,fname)
        #fname = os.path.join(os.getcwg(),"Claibration Results",fname)
        
        gamma_r     = self.gamma_ref()
        ref_strain  = self.data['reference']['strain(%)']/1e2
        ref_modulus = self.data['reference']['G/Gmax']
        ref_damping = self.data['reference']['damping(%)']
        thetas      = self.data['fit']['theta']
        tau_max     = self.shear_strength
        Gmax        = self.Gmax()

        theta_tau_ref   = self.fcn_theta_tau(ref_strain,gamma_r,ref_modulus)

        theta_tau_fit = self.fcn_theta_tau_fitting(ref_strain,gamma_r,thetas)
        shear_ratio_fit = self.fcn_shear_ratio(ref_strain,gamma_r,theta_tau_fit)
        G_ratio_fit = (shear_ratio_fit*tau_max/ref_strain)/Gmax
        damping_perc_fit = self.calculate_damping(masing=False,Np=5000,echo=False,Display=False)

        ms = 7.5
        marker_style1 = dict(color='black', linestyle='', marker='o',
                            markersize=ms, fillstyle='none')
        marker_style2 = dict(color='b', linestyle='-', marker='^',
                            markersize=ms, fillstyle='none')
        marker_style3 = dict(color='r', linestyle='--', marker='',
                            markersize=ms, fillstyle='none')
        marker_style4 = dict(color='grey', linestyle='--', marker='',
                            markersize=ms, fillstyle='none')
        marker_style5 = dict(color='b', linestyle='-', marker='',
                            markersize=ms, fillstyle='none')

        
        # Create figure
        hfig = plt.figure(figsize=(10.5,8)) #constrained_layout=True
        
        # Create a gridspec obj
        gs1 = gs.GridSpec(nrows=3,ncols=2)
        
        # Create axes
        ax1 = plt.subplot(gs1[0, 0])
        ax2 = plt.subplot(gs1[1, 0])
        ax3 = plt.subplot(gs1[2, 0])
        ax4 = plt.subplot(gs1[0:2, 1])
        ax5 = plt.subplot(gs1[2, 1])
        
        ax1.semilogx(ref_strain*1e2,
                 ref_modulus,
                **marker_style1,
                label='Reference')
        ax2.semilogx(ref_strain*1e2,
                 ref_damping,
                **marker_style1,
                label='Reference')
        ax3.semilogx(ref_strain*1e2,
                 ref_modulus*Gmax*ref_strain/1e3,
                **marker_style1,
                label='Reference')
        
        ax1.semilogx(ref_strain*1e2,
                     G_ratio_fit,
                     **marker_style2,
                     label='GQ/H + MRDF (Fit)')
        ax2.semilogx(ref_strain*1e2,
                     damping_perc_fit ,
                     **marker_style2,
                     label='GQ/H + MRDF (Fit)')
        ax3.semilogx(ref_strain*1e2,
                     shear_ratio_fit*tau_max/1e3,
                     **marker_style2,
                     label='GQ/H + MRDF (Fit)')
        ax3.semilogx(ax3.get_xlim(),
                     np.ones(2)*tau_max/1e3,
                     **marker_style3,
                     label=r'$\tau_{max}$')
        ax3.legend()
        
        _,_,hyst1 = self.single_element_test(target_strain_perc=0.1,masing=True,Np=5000,display=False)
        _,_,hyst2 = self.single_element_test(target_strain_perc=1.5,masing=True,Np=5000,display=False)
        _,_,hyst3 = self.single_element_test(target_strain_perc=5,masing=True,Np=5000,display=False)
   
        for hys in [hyst1,hyst2,hyst3]:
            ax4.plot(np.concatenate((hys['x_unload'],hys['x_reload']))*1e2,
                         np.concatenate((hys['y_unload'],hys['y_reload']))/1e3,
                         **marker_style4,
                         label='GQ/H + Masing')
            
        _,_,hyst1 = self.single_element_test(target_strain_perc=0.1,masing=False,Np=5000,display=False)
        _,_,hyst2 = self.single_element_test(target_strain_perc=1.5,masing=False,Np=5000,display=False)
        _,_,hyst3 = self.single_element_test(target_strain_perc=5,masing=False,Np=5000,display=False)
     
        for hys in [hyst1,hyst2,hyst3]:
            ax4.plot(np.concatenate((hys['x_unload'],hys['x_reload']))*1e2,
                         np.concatenate((hys['y_unload'],hys['y_reload']))/1e3,
                         **marker_style5,
                         label='GQ/H + MRDF')    
        
        ax5.text(0,1,"Layer Information\n\n" + 
                        """\tLayer ID: {}
                           \tMaterial: {}
                           \tDepth (m): {}
                           \t$\gamma$ ($kN/m^3$): {}
                           \t$V_s$ (m/s): {}
                           \t$G_{{max}}$ (MPa): {}
                           \t$\\tau_{{max}}$ (kPa): {}
                            """.format(self.layer_ID,
                                       self.material,
                                        np.round(self.depth,2),
                                       np.round(self.unit_weight/1e3),
                                       np.round(self.Vs),
                                       np.round(self.Gmax/1e6),
                                       np.round(self.shear_strength/1e3)),
                                horizontalalignment='left',
                                verticalalignment='top')
                 
        ax5.text(0.5,1.0, "Calibration Information\n\n" +
                """\tGQ/H Fit Parameters
                    \t($\\theta_1$,$\\theta_2$,$\\theta_3$,$\\theta_4$,$\\theta_5$):
                    \t({:3.2f},{:3.2f},{:3.2f},{:3.2f},{:3.2f})\n\n
                    \tMRDF Fit Parameters
                    \t($p_1$,$p_2$,$p_3$):
                    \t({:3.2f},{:3.2f},{:3.2f})
                    """.format(self.data['fit']['theta']['1'],
                               self.data['fit']['theta']['2'],
                               self.data['fit']['theta']['3'],
                               self.data['fit']['theta']['4'],
                               self.data['fit']['theta']['5'],
                               self.data['fit']['mrdf']['1'],
                               self.data['fit']['mrdf']['2'],
                               self.data['fit']['mrdf']['3']),
                        horizontalalignment='left',
                        verticalalignment='top')
        
        ax5.set_xticklabels([])
        ax5.set_yticklabels([])
        ax5.axis('off')
        
        ax1.set(ylabel=r'$G/G_{max}$')
        ax2.set(ylabel=r'Damping Ratio, $\xi$ (%)')
        ax3.set(xlabel=r'Shear strain, $\gamma$ (%)',
                ylabel=r'Shear Stress, $\tau$ (kPa)')
        ax4.set(xlabel=r'Shear strain, $\gamma$ (%)',
                ylabel=r'Shear stress, $\tau$ (kPa)')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        if display:
            plt.show()
        
        hfig.savefig(fname, dpi=150, facecolor='w', edgecolor='w',
                orientation='portrait', papertype='letter', format='pdf',
                transparent=False, bbox_inches='tight', pad_inches=0.1,
                metadata=None)
        hfig.clf()
        
        return
    

    # ------------------------------------------------
    def fcn_shear_ratio(self,gamma,gamma_r,theta_tau):
        """
        Groholski et al. (2016), Eq. 10
        Expression for shear stress to max shear stress of BB curve
        ______________________________________________________________________
        Inputs
        -------
        gamma : strain
        gamma_r : reference strain
        thea_tau : as defined by Groholski et al. (2016), Eq. 12
        
        Returns
        -------
        tau_ratio :
        """
        num = 2.*(gamma/gamma_r)
        denom = 1. + (gamma/gamma_r) + np.sqrt((1+(gamma/gamma_r))**2. - 4.*theta_tau*(gamma/gamma_r))
        tau_ratio = num/denom
        return tau_ratio

    def fcn_theta_tau(self,gamma,gamma_r,G_ratio):
        """
        Groholski et al. (2016), Eq. 12
        Expression for theoretical theta_tau based on MRD curves
        ______________________________________________________________________
        Inputs
        -------
        gamma : strain
        gamma_r : reference strain
        G_ratio : G/Gmax
        
        Returns
        -------
        theta_tau : analytical theta_tau
        """
        num = G_ratio + G_ratio*(gamma/gamma_r) - 1
        denom = (gamma/gamma_r) * G_ratio**2 
        theta_tau = num/denom
        return theta_tau

    def fcn_theta_tau_fitting(self,gamma,gamma_r,thetas):
        """
        Groholski et al. (2016), Eq. 13
        Expression for curve-fitting form of theta_tau
        ______________________________________________________________________
        Inputs
        -------
        gamma : strain
        gamma_r : reference strain
        thetas : fitting parameters; a dictionary
        
        Returns
        -------
        theta_tau : empirically fitted theta_tau
        """
        num = thetas['4']*np.power(gamma/gamma_r,thetas['5'])
        denom = np.power(thetas['3'],thetas['5']) + thetas['4']*np.power(gamma/gamma_r,thetas['5'])
        theta_tau_fit = thetas['1'] + thetas['2']*num/denom
        return theta_tau_fit
    
    def GQH_F_BB(self,X,gamma_r,thetas,tau_max):
        """
        Groholski et al. (2016), Eq. 21
        Backbone curve of GQ/H model in terms of shear strength
        ______________________________________________________________________
        Inputs
        -------
        X : strain
        gamma_r : reference strain
        thetas : fitting parameters; a dictionary
        tau_max : shear strength
        
        Returns
        -------
        tau : GQ/H backbone in terms of shear stress
        """
        theta_tau = self.fcn_theta_tau_fitting(X,gamma_r,thetas)

        tau = tau_max*((1./(2.*theta_tau))
                       *(1.+X/gamma_r-np.sqrt(np.power(1+X/gamma_r,2.)
                                              -4.*theta_tau*X/gamma_r)))
        return tau

    def GQH_F_URL_Masing(self,X,gamma_r,tau_rev,tau_max,thetas):
        """
        Groholski et al. (2016), effectively, Eq. 22
        URL with 2nd Masing Rule following Eq. 20
        ______________________________________________________________________
        Inputs
        -------
        X : strain
        gamma_r : reference strain
        ta_rev : last stress reversal point
        tau_max : shear strength
        thetas : fitting parameters; a dictionary
        
        Returns
        -------
        tau : URL following masing rule in terms of shear stress
        """
        Fbb = self.GQH_F_BB(X,gamma_r,thetas,tau_max)
        tau = tau_rev + 2.*Fbb

        return tau

    def MRDF_factor(self,G_ratio,p_dict):
        """
        Groholski et al. (2016), Eq. 23
        MRDF Reduction Factor
        ______________________________________________________________________
        Inputs
        -------
        G_ratio : G/Gmax
        p_dict : fitting parameters; a dictionary
        
        Returns
        -------
        MRDF: empirically-fitted modulus reduction factors
        """
        return p_dict['1'] - p_dict['2']*(1.-G_ratio)**p_dict['3']

    def GQH_F_URL_MRDF(self,X,gamma_r,tau_rev,tau_max,thetas,p_dict,G_gamma_max,Gmax):
        """
        Groholski et al. (2016), Eq. 24
        Using non-masing/modified modulus reduction rules
        ______________________________________________________________________
        Inputs
        -------
        X : strain
        gamma_r : reference strain
        ta_rev : last stress reversal point
        tau_max : shear strength
        thetas : fitting parameters; a dictionary
        G_gamma : shear modulus experienced at gamma_max on backbone curve
        Gmax : small strain shear modulus
        
        Returns
        -------
        tau : URL following non-masing (modified MRDF) rules in terms of shear stress
        """
        theta_tau = self.fcn_theta_tau_fitting(X,gamma_r,thetas)

        G_ratio = G_gamma_max/Gmax
        F = self.MRDF_factor(G_ratio,p_dict)

        rho = np.sign(tau_max)

        tau = rho*F*((rho*tau_max/theta_tau)*(
            1+(X/gamma_r)-np.sqrt(np.power(1+X/gamma_r,2)-4*theta_tau*X/gamma_r))
                -G_gamma_max*X*2)  + rho*G_gamma_max*X*2 + tau_rev
        return tau
    
    def damping_ratio(self,x_strain,y_reload,y_unload,gamma_max,G_gamma_max):
        """
        Calculate damping ratio (e.g. per Kramer (1996), Fig. 5.22)
        
        Calculates the damping ratio from a hysteretic loop defined
        by (x_strain, y_reload, y_unload) and the elastic energy defined
        by (gamma_max, G_gamma_max).
        ______________________________________________________________________
        Inputs
        -------
        x_strain : strain (e.g. -gamma_max to gamma_max)
        y_reload : reloading stress path
        y_unload : unloading stress path from gamma_ax
        gamma_max : max shear strain, or shear strain level for computing damping ratio
        G_gamma_max : max secant shear modulus for strain level
        
        Returns
        -------
        damping_ratio_perc : damping ratio in (%)
        """
        # Sort, x_strain, y_unload, y_reload ascending, monotonic
        # for integration
        x_strain = np.sort(x_strain)
        y_reload = np.sort(y_reload)
        y_unload = np.sort(y_unload)

        # Calculate hysteretic and peak/elastic energies
        A_hysteretic = np.trapz(y_reload-y_unload,x_strain)
        A_elastic = 0.5*G_gamma_max*(gamma_max**2)

        # Calculate damping ratio
        damping_ratio = A_hysteretic/(4*np.pi*A_elastic)
        damping_ratio_perc = damping_ratio*100

        return damping_ratio_perc

    # ---------------------Define Static Methods---------------------------
    def create_reference_data(path,echo=True):
        """
        Creates (compiles) a tree data structure of material reference data.
        
        To maintain consistency for now,
        
        Files should be csv, comma-delimited.
        Files must use the headers: strain(%),G/Gmax,damping(%)
        Files must use 'RefData' tag, and be seperated by three underscores
        as shown, e.g.
                RefData_Clay_VD91_PI15.csv
                RefData_Sand_EPRI93_Depth006m.csv
        ______________________________________________________________________
        Inputs
        -------
        path : reference material data folder path
        
        Returns
        -------
        ref_data : reference data
        """
        # Grab reference data input files from directory
        if os.path.isdir(path):
            fnames = os.listdir(path)
            
            fnames = [name for name in fnames if '.csv' in name]
            
            grabbed_files = []
            tree_constructor= []
            for file in fnames: #glob.glob('*.csv'):
                if 'RefData' in file:
                    branch = file.strip('RefData_').strip('.csv').split('_')
                    tree_constructor.append(branch)
                    grabbed_files.append(file)
    
            # Create my reference data construct (tips of tress are pandas dataframes)
            ref_data = Tree()
            for file, branch in zip(grabbed_files,tree_constructor):
                full_file_path = os.path.join(path,file)
                df = pd.read_csv(full_file_path)
                ref_data[branch[0]][branch[1]][branch[2]] = df
    
            # Echo
            if echo:
                for file in grabbed_files:
                    print(file)
                for branch in tree_constructor:
                    print(branch)
            print("Created reference database.")
        return ref_data
    
    

class ModelProfile:
    """
    Our profile layer class.
    
    Basically represents our borehole data/characterization, and stores our layer instances.
    Methods are defined for calibrating, and viewing results relative to the scope of the profile.
    """
    
    # ---------------------Define Class Methods---------------------------
    def __init__(self, df, echo=True):
        """
        Instance initialization. Define attributes/hierarchy here.
        ______________________________________________________________________
        Inputs
        -------
        df : dataframe containing layer metadata
        """
        
        # Basically creating a list of layer objects and storing it in our profile instance here
        numb_layers = df.shape[0]
        if echo:
            print("""Number of layers is: {}\n""".format(numb_layers))

        layers = [] # initialize
        for idx in range(0,numb_layers):

            # Pack the excel line into a ini dict for layer instantiation
            ini_dict = {'layer_ID'      : df['Layer No'][idx],
                        'depth'         : df['Depth to Middle of Layer (m)'][idx],
                        'thickness'     : df['Thickness (m)'][idx],
                        'unit_weight'   : df['Unit Weight (kN/m3)'][idx]*1e3,
                        'sigma_vo_eff'  : df["Sig' V [kPa]"][idx]*1e3,
                        'shear_strength': df['Shear Strength [kPa]'][idx]*1e3,         # used for normal
                        'shear_cov'     : df['Shear Strength COV [%]'][idx]*1e-2,      # used for normal
                        'corr'          : df['rho (Vs, Shear Str)'][idx],              # used for normal and uniform
                        'shear_str_min' : df['Shear Strength - Min [kPa]'][idx]*1e3,   # used for uniform
                        'shear_str_max' : df['Shear Strength - Max [kPa]'][idx]*1e3,   # used for uniform
                        'distribution'  : df['Distribution'][idx],                     # e.g. normal, uniform
                        'Vs'            : df['Vs (m/s)'][idx],
                        'PI'            : df['PI'][idx],
                        'OCR'           : df['OCR'][idx],
                        'Ko'            : df['K_OCR'][idx],
                        'Cu'            : df['Cu'][idx],
                        'd50'           : df['d50 (mm)'][idx],              # in [mm]
                        'material'      : df['Material'][idx],
                        'memo'          : df['Modulus/Damping Curve'][idx],} # I'm using the memo field to store what
                                                                             # modulus/damping curve to use

            # instantiate layer
            layer = ModelLayer(ini_dict)
            #print(layer.__dict__)

            # append to our layer list
            layers.append(layer)

            if echo:
                print("""Layer {} attributes:\n{}""".format(idx, layers[idx].__dict__))
        print("""Instantiated a profile with {} layers.""".format(numb_layers))
    
        # Define instance variables
        self.layers = layers
        
        # Maybe retrieve/create reference database here?
        
        return
    
    
    def Vs(self):
        """
        Returns profile Vs.
        ______________________________________________________________________
        Outputs
        -------
        Vs : array of profile Vs corresponding to layer instances
        """
        Vs = np.array([layer.Vs for layer in self.layers])
        return Vs
    
    
    def shear_strength(self):
        """
        Returns profile shear strength.
        ______________________________________________________________________
        Outputs
        -------
        shear_strength : max shear strength corresponding to layer instances
        """
        shear_strength = np.array([layer.shear_strength for layer in self.layers])
        return shear_strength

    
    def OCR(self):
        """
        Returns profile OCR.
        ______________________________________________________________________
        Outputs
        -------
        OCR : over-consolidation ratio corresponding to layer instances
        """
        OCR = np.array([layer.OCR for layer in self.layers])
        return OCR    

    
    def Ko(self):
        """
        Returns profile Ko.
        ______________________________________________________________________
        Outputs
        -------
        Ko : coefficient of lateral earth pressure coefficient corresponding to layer instances
        """
        Ko = np.array([layer.Ko for layer in self.layers])
        return Ko    
    
    
    def depth_top(self):
        """
        Returns depth to top of layer based on layer instance values
        ______________________________________________________________________
        Outputs
        -------
        depth_top : depth to top of layer
        """
        # Get layer thicknesses
        thickness = [layer.thickness for layer in self.layers]
        depth_top = np.concatenate((np.array([0]),np.cumsum(thickness[:-1])))
        return depth_top
    
    
    def depth_bott(self):
        """
        Returns depth to bottom of layer based on layer instance values
        ______________________________________________________________________
        Outputs
        -------
        depth_bott : depth to bottom of layer
        """
        thickness = [layer.thickness for layer in self.layers]
        thickness[-1] = 0.0001 # Make half-space finite sized, but small for plotting
        depth_bott = np.cumsum(thickness)
        return depth_bott
    
    
    def thickness(self):
        """
        Returns thickness of layer based on layer instance values
        ______________________________________________________________________
        Outputs
        -------
        thickness : thickness of layer
        """
        thickness = [layer.thickness for layer in self.layers]
        return thickness
    
    
    def Tn(self):
        """
        Returns estimate of fundamental period of column, Tn.
        ______________________________________________________________________
        Outputs
        -------
        Tn : fundamental period (s)
        """
        H = np.sum(self.thickness()[:-1])
        Vs_avg = H/np.sum(self.thickness()[:-1]/self.Vs()[:-1])
        Tn = 4*H/Vs_avg
        return Tn
    
    
    def rayleigh_damping(self,f1_ratio=0.2,f2_ratio=5.0,psi=0.03):
        """
        Returns f1,f2 corner frequencies and alpha/beta damping records 
        for Rayleigh frequency-dependent damping.
        ______________________________________________________________________
        Inputs
        -------
        psi : damping ratio
        f1_ratio : f1/fn ( = 1/5 default)
        f2_ratio : f2/fn ( = 5 default)
        -------
        f1 : first corner freq
        f2 : second corner feq (> f1)
        """
        f1 = f1_ratio/self.Tn()
        f2 = f2_ratio/self.Tn()
        
        beta = 2.0*psi/(2.0*np.pi*(f1+f2))
        alpha = beta*(2.0*np.pi*f1)*(2.0*np.pi*f2)
        
        return f1, f2, alpha, beta
    
    
    def assign_reference_models(self,reference_database,echo=True):
        """
        Assigns layer reference data/models (i.e. G/Gmax, damping) to use for
        calibration of the GQ/H + MRDF model.
        ______________________________________________________________________
        Inputs
        -------
        reference_database : reference database containing discrete data
        """

        success_count = 0
        for layer in self.layers:

            # Vucetic and Dobry (19991); clays/cohesive materials; model is PI dependent
            if "Vucetic and Dobry (1991)" in layer.memo:
                if layer.PI <= 15:
                    layer.assign_reference_data(reference_database['Clay']['VD91']['PI15'])
                elif layer.PI <= 30:
                    layer.assign_reference_data(reference_database['Clay']['VD91']['PI30'])
                elif layer.PI <= 50:
                    layer.assign_reference_data(reference_database['Clay']['VD91']['PI50'])
                elif layer.PI <= 100:
                    layer.assign_reference_data(reference_database['Clay']['VD91']['PI100'])
                else:
                    layer.assign_reference_data(reference_database['Clay']['VD91']['PI200'])


            # EPRI (1993); granular materials, model is depth-dependent
            elif "EPRI (1993)" in layer.memo:
                if layer.depth <= 6:
                    layer.assign_reference_data(reference_database['Sand']['EPRI93']['Depth006m'])
                elif layer.depth <= 15:
                    layer.assign_reference_data(reference_database['Sand']['EPRI93']['Depth015m'])
                elif layer.depth <= 36:
                    layer.assign_reference_data(reference_database['Sand']['EPRI93']['Depth036m'])
                elif layer.depth <= 76:
                    layer.assign_reference_data(reference_database['Sand']['EPRI93']['Depth076m'])
                elif layer.depth <= 152:
                    layer.assign_reference_data(reference_database['Sand']['EPRI93']['Depth152m'])
                else:
                    layer.assign_reference_data(reference_database['Sand']['EPRI93']['Depth305m'])


            # Seed and Idriss (1970); sands
            elif "Seed and Idriss (1970)" in layer.memo:
                if "Average" in layer.memo or "Mean" in layer.memo:
                    layer.assign_reference_data(reference_database['Sand']['SI70']['Mean'])
                elif "Lower" in layer.memo:
                    layer.assign_reference_data(reference_database['Sand']['SI70']['Lower'])
                elif "Upper" in layer.memo:
                    layer.assign_reference_data(reference_database['Sand']['SI70']['Upper'])
                    
            # Sy et al. (1991); sand or silt
            elif "Sy et al. (1991)" in layer.memo:
                if 'Sand' in layer.material or 'Sand' in layer.memo:
                    layer.assign_reference_data(reference_database['Sand']['SY91']['Mean'])
                elif 'Silt' in layer.material or 'Silt' in layer.memo:
                    layer.assign_reference_data(reference_database['Silt']['SY91']['Mean'])


            # Darendeli (2001); generic (sand, silt clay) with d50 < 30 mm; depends on PI, OCR, sigma_eff_vo, freq, N
            elif "Darendeli (2001)" in layer.memo:

                    gamma = np.logspace(np.log10(ModelLayer.ref_data_strain_min_perc), # sample space
                                        np.log10(ModelLayer.ref_data_strain_max_perc),
                                        ModelLayer.ref_data_numb_points) 

                    MR, Dtotal, Dmin = mrd_darendeli(gamma,
                                                    np.array([layer.PI]),
                                                    np.array([layer.OCR]),
                                                    np.array([layer.sigma_vo_eff])/1e3, # stress in kPa
                                                    Ko =np.array([1.0]) if not layer.Ko else np.array([layer.Ko]) # assign Ko=1.0 if none given
                                                    ) 
                    d = {'strain(%)': gamma,
                         'G/Gmax': MR,
                         'damping(%)': Dtotal-Dmin} # Subtract ss damping as only want hyst. comp. for MRDF fitting 
                    layer.assign_reference_data(pd.DataFrame(data=d))


            # Menq (2003); granular materials with d50 > 30 mm
            elif "Menq (2003)" in layer.memo:


                    gamma = np.logspace(np.log10(ModelLayer.ref_data_strain_min_perc), # sample space
                                        np.log10(ModelLayer.ref_data_strain_max_perc),
                                        ModelLayer.ref_data_numb_points) 

                    MR, Dtotal, Dmin = mrd_menq(gamma,
                                               np.array([layer.sigma_vo_eff])/1e3,      # stress in kPa
                                               Ko =np.array([1.0]) if not layer.Ko else np.array([layer.Ko]), # assign Ko=1.0 if none given
                                               Cu =np.array([1.2]) if not layer.Cu else np.array([layer.Cu]),   # assign as medium sand if no ..
                                               d50=np.array([1.0]) if not layer.d50 else np.array([layer.d50])  # d50 or Cu given
                                               )
                    d = {'strain(%)': gamma,
                         'G/Gmax': MR,
                         'damping(%)': Dtotal-Dmin} # Subtract ss damping as only want hyst. comp. for MRDF fitting  
                    layer.assign_reference_data(pd.DataFrame(data=d))

            else:
                if echo:
                    print("""No applicable key found for material '{}' in layer: {}""".format(layer.material,layer.layer_ID))
                success_count -= 1

            success_count += 1
        if echo:
            print("""Assigned reference data to {} layers.""".format(success_count))
        return


    def calibrate_profile(self,min_strain_perc_gqh=0.0001,max_strain_perc_gqh=0.5,min_strain_perc_mrdf=0.005,
                          start_depth =-np.Inf,end_depth=np.Inf,echo=True,view_calib_output=False, 
                          save_calib_output_as_pdf=False,output_folder='Calibration Results'):
        """
        Calibrate model profile layers.
        ______________________________________________________________________
        Inputs
        -------
        layers : list containing sequence of all layer instances from dataframe
        min_strain_perc_gqh : minimum strain for fitting GQH model (0.0001 default)
        max_strain_perc_gqh : maximum strain for fitting GQH model (0.5 default)
        min_strain_perc_mrdf : minimum strain for fitting MRDF model (0.005 default)
        
        start_depth: starting depth of layers (-Inf default)
        end_depth :  ending depth of layers (Inf default)
        
        echo : if True, echos successful calibration for each layer
        view_calib_output : if True, plots individual layer calibration in console (slows)
        save_calib_output_as_pdf : if True, saves individual layer calibration pdf plots (slow)
        output_folder : output folder name to save pdf outputs (saves as root sub-directory)

        """
        # Set output location for writing
        if save_calib_output_as_pdf:
            output_path_pdf = os.path.join(os. getcwd(), output_folder) 
        
        
        # They layers we want to calibrate after applying depth filters
        keep_layers = [layer for layer in self.layers if layer.depth >= start_depth and layer.depth <= end_depth]
        
        
        # Loop over and calibrate each soil layer
        exclude = 'Half Space'
        success_count = 0
        for layer in keep_layers:

            # Calibrate laters
            if exclude not in layer.material:
                if echo:
                    print("""Layer ID: {}""".format(layer.layer_ID))
                layer.calibrate_layer_params(min_strain_perc=min_strain_perc_gqh,
                                             max_strain_perc=max_strain_perc_gqh,
                                             min_strain_perc_mrdf=min_strain_perc_mrdf,
                                             echo=echo,display=False)
                success_count += 1

                # Save pdf summary
                if save_calib_output_as_pdf:
                    layer.save_pdf_output(path=output_path,
                                          display=view_calib_output)

            else: # Exclude all materials from allowable materials list
                if echo:
                    print("""Layer ID: {} excluded from calibration.
                    Layer Material: {}""".format(layer.layer_ID,layer.material))
        print("""Profile calibration complete. Calibrated {} layers.""".format(success_count))
        return

    
    def preview_GQH_calibration(self,nrows=5,ncols=6):
        """
        Generates a matrix plot preview of calibrated modulus reduction curves.
        ______________________________________________________________________
        Inputs
        -------
        nrows : number of rows per plate
        ncols : number of columns per plate
        """
        color = 'mediumblue'
        numb_plate = int(np.ceil(len(self.layers)/(nrows*ncols)))
        for idx_plate in range(0,numb_plate):

            # Parse the layer start/stop numbers onto plates
            layer_start = idx_plate*nrows*ncols
            layer_end   = (idx_plate+1)*nrows*ncols
            if layer_end > len(self.layers): # less than a full plate condition
                layer_end = len(self.layers)

            fig = plt.figure(figsize=(11,8.5))
            fig.suptitle('GQ/H-Calibrated Modulus Reduction $G/G_{max}$ Curves',x=0.5,y=0.92)
            ax = [plt.subplot(nrows,ncols,i+1) for i in range(nrows*ncols)]
            plt.subplots_adjust(wspace=0, hspace=0)

            for idx, layer in enumerate(self.layers[layer_start:layer_end]):

                if layer.data['fit']['theta']: # Gets around undefined rock layer

                    #strain      = np.linspace(1e-8,1e-1,50000)
                    strain      = np.logspace(np.log10(0.0000001),np.log10(0.1)) 
                    gamma_r     = layer.shear_strength/layer.Gmax()
                    thetas      = layer.data['fit']['theta']
                    tau_max     = layer.shear_strength
                    Gmax        = layer.Gmax()

                    theta_tau_fit = layer.fcn_theta_tau_fitting(strain,gamma_r,thetas)
                    shear_ratio_fit = layer.fcn_shear_ratio(strain,gamma_r,theta_tau_fit)
                    G_ratio_fit = (shear_ratio_fit*tau_max/strain)/Gmax

                    ax[idx].semilogx(layer.data['reference']['strain(%)'],
                                     layer.data['reference']['G/Gmax'],
                                    'ko',ms=5.5,fillstyle='none')

                    ax[idx].semilogx(strain*1e2,
                                     G_ratio_fit,
                                    '-',color=color) 
                    ax[idx].text(0.15,0.1,"""layer {}""".format(layer.layer_ID),
                                 verticalalignment='bottom',horizontalalignment='left',
                                 transform=ax[idx].transAxes)

                # Remove labels/ticks
                for a in ax:
                    a.set_xticklabels([])
                    a.set_yticklabels([])
        return


    def preview_MRDF_calibration(self,nrows=5,ncols=6):
        """
        Generate a matrix plot preview of calibrated damping curves for all layers
        ______________________________________________________________________
        Inputs
        -------
        nrows : number of rows per plate
        ncols : number of columns per plate
        """

        color = 'mediumblue'
        numb_plate = int(np.ceil(len(self.layers)/(nrows*ncols)))
        for idx_plate in range(0,numb_plate):

            # Parse the layer start/stop numbers onto plates
            layer_start = idx_plate*nrows*ncols
            layer_end   = (idx_plate+1)*nrows*ncols
            if layer_end > len(self.layers): # less than a full plate condition
                layer_end = len(self.layers)

            fig = plt.figure(figsize=(11,8.5))
            fig.suptitle('MRDF-Calibrated Damping $\\xi$ Curves',x=0.5,y=0.92)
            ax = [plt.subplot(nrows,ncols,i+1) for i in range(nrows*ncols)]
            plt.subplots_adjust(wspace=0, hspace=0)

            for idx, layer in enumerate(self.layers[layer_start:layer_end]):

                if layer.data['fit']['theta']: # Gets around undefined rock layer

                    damping_perc = layer.calculate_damping(masing=False,Np=5000,echo=False,display=False)

                    ax[idx].semilogx(layer.data['reference']['strain(%)'],
                                     layer.data['reference']['damping(%)'],
                                    'ko',ms=5.5,fillstyle='none')

                    ax[idx].semilogx(layer.data['reference']['strain(%)'],
                                     damping_perc,
                                    '-',color=color) 
                    ax[idx].text(0.15,0.80,"""layer {}""".format(layer.layer_ID),
                                 verticalalignment='bottom',horizontalalignment='left',
                                 transform=ax[idx].transAxes)

                    # Remove labels/ticks
                    for a in ax:
                        a.set_xticklabels([])
                        a.set_yticklabels([])
        return


    def preview_hysteretic_calibration(self,nrows=5,ncols=6,strain_levels_perc=(0.1,1.5,5.0)):
        """
        Generate a matrix plot preview of calibrated hysteretic behavior for all layers
        ______________________________________________________________________
        Inputs
        -------
        nrows : number of rows per plate
        ncols : number of columns per plate
        strain_levels_perc = strain levels to draw hysteretic curves up to
        """

        color='mediumblue'
        numb_plate = int(np.ceil(len(self.layers)/(nrows*ncols)))
        for idx_plate in range(0,numb_plate):

            # Parse the layer start/stop numbers onto plates
            layer_start = idx_plate*nrows*ncols
            layer_end   = (idx_plate+1)*nrows*ncols
            if layer_end > len(self.layers): # less than a full plate condition
                layer_end = len(self.layers)

            fig = plt.figure(figsize=(11,8.5))
            fig.suptitle('GQ/H+MRDF-Calibrated Hysteretic Behavior | Strain levels: '+str(strain_levels_perc)+'%',x=0.5,y=0.92)
            ax = [plt.subplot(nrows,ncols,i+1) for i in range(nrows*ncols)]
            plt.subplots_adjust(wspace=0, hspace=0)

            for idx, layer in enumerate(self.layers[layer_start:layer_end]):

                if layer.data['fit']['theta']: # Gets around undefined rock layer


                    hysts =  [layer.single_element_test(target_strain_perc=strain,
                                                        Np=5000,display=False) for strain in strain_levels_perc]


                    [ax[idx].plot(np.concatenate((hys[2]['x_unload'],hys[2]['x_reload']))*1e2,
                                  np.concatenate((hys[2]['y_unload'],hys[2]['y_reload']))/1e3,
                                  '-',color=color,label='GQ/H + MRDF') for hys in hysts   ]

                    ax[idx].text(0.15,0.80,"""layer {}""".format(layer.layer_ID),
                                 verticalalignment='bottom',horizontalalignment='left',
                                 transform=ax[idx].transAxes)

                    # Remove labels/ticks
                    for a in ax:
                        a.set_xticklabels([])
                        a.set_yticklabels([])
        return
    

    def save_calibration_results(self,output_file_name=None,output_sheet_name="CalibratedProfile",ss_damp_perc=2):
        """
        Export calibration results to an output excel file
        ______________________________________________________________________
        Inputs
        -------
        fname : output filename or full path
        output_sheet_name: output sheetname
        ss_damp_perc = small strain damping in perc (assumed for whole profile)
        """
        out_dict = {'Layer Number'              :[layer.layer_ID for layer in self.layers],
                    'Layer_Name'                :[layer.material for layer in self.layers],
                    'Thickness (m)'             :[layer.thickness for layer in self.layers],
                    'Unit Weight (KN/m^3)'      :[layer.unit_weight/1e3 for layer in self.layers],
                    'Shear Wave Velocity (m/s)' :[layer.Vs for layer in self.layers],
                    'Shear Strength (kPa)'      :[layer.shear_strength/1e3 for layer in self.layers],
                    'Dmin (%)'                  :list(np.ones(len(self.layers))*ss_damp_perc),
                    'theta1':[],'theta2':[],
                    'theta3':[],'theta4':[],
                    'theta5':[],
                    'p1':[],'p2':[],'p3':[]}
        for layer in self.layers:
            if layer.data['fit']['theta']:
                out_dict['theta1'].append( layer.data['fit']['theta']['1'] )
                out_dict['theta2'].append( layer.data['fit']['theta']['2'] )
                out_dict['theta3'].append( layer.data['fit']['theta']['3'] )
                out_dict['theta4'].append( layer.data['fit']['theta']['4'] )
                out_dict['theta5'].append( layer.data['fit']['theta']['5'] )
                out_dict['p1'].append( layer.data['fit']['mrdf']['1'] )
                out_dict['p2'].append( layer.data['fit']['mrdf']['2'] )
                out_dict['p3'].append( layer.data['fit']['mrdf']['3'] )
            else:
                out_dict['theta1'].append( None )
                out_dict['theta2'].append( None )
                out_dict['theta3'].append( None )
                out_dict['theta4'].append( None )
                out_dict['theta5'].append( None )
                out_dict['p1'].append( None )
                out_dict['p2'].append( None )
                out_dict['p3'].append( None )

        #Create the dataframe
        df_out = pd.DataFrame(out_dict)

        # Export the same format as DEEPSOIL headers/input file.
        if output_file_name:
            df_out.to_excel(output_file_name,
                            sheet_name=output_sheet_name)
            print("""Profile calibration saved: {}""".format(output_file_name))
        return df_out
    
    
    def create_deepsoil_profile(self,input_full_fname=None,DS_output_fname=None,DS_output_path=None,
                                record_depths_m=[0],GWT_depth=0,effective_SSR=0.65,damping="freq-ind",
                                f1=0.2,f2=5.0,echo=False):
        """
        Creates a DEEPSOIL (.dp) input profile using the profile layer attributes and calibrated fields.
        ______________________________________________________________________
        Inputs
        -------
        input_full_fname : input filename full path
        DS_output_fname : DEEPSOIL output filename
        DS_output_path : DEEPSOIL output filename path
        record_depths_m : a list of depth in [m] to output; matches based on closest top of layer
                          NOTE: DEEPSOIL doesn't currently actually use this field so always
                          defaults to surface motion
        GWT_depth : groundwater table depth [m] (default 0m), assigned to to closest layer
        effective_SSR : effective shear stress ratio (65% typical)
        damping : damping option, either 'freq-ind' (frequency independent), the default; or 'rayleigh' (frequency dependent)
        f1_ratio : f1/fn when damping = 'rayleigh'  
        f2_ratio : f2/fn when damping = 'rayleigh'
        echo : if True, echos text file to console
        """
        
        # Get dataframe for constructing deepsoil input; if no input path specified, 
        # constructs the dataframe from profile instance
        if input_full_fname:
            df = pd.read_excel(input_full_fname)
        else:
            df = self.save_calibration_results()
        
        # Initalize all dataframe fields to no record
        df['Output'] = 'FALSE'

        # Divide the data frame into two seperate frames for soil/half space
        #df_soil = df[df['Layer_Name'] != 'Half Space']
        #df_half = df[df['Layer_Name'] == 'Half Space']
        df_soil = df[~df['Layer_Name'].str.contains('Half Space')]
        df_half = df[ df['Layer_Name'].str.contains('Half Space')]
        numb_soil_layers = len(df_soil['Layer Number'])
        
        # Assign GWT to closest element
        depth_top = np.append(0,df_soil['Thickness (m)'].iloc[:-1].cumsum().to_numpy())
        closest_depth_idx = np.nanargmin(np.abs(depth_top-GWT_depth))
        GWT_layer = closest_depth_idx + 1 # Add 1 since layers start from 1

        # Assign output tags to closest top of layer
        for depth_m in record_depths_m:
            closest_depth_idx = np.nanargmin(np.abs(depth_top-depth_m))
            df.loc[closest_depth_idx,'Output'] = 'TRUE'

            
        # Write the analysis settings
        if damping == "rayleigh":
            block1 = inspect.cleandoc("""
                        [FILE_VERSION]:[1]
                        [ANALYSIS_DOMAIN]:[TIME+FREQUENCY]
                            [MAX_ITERATIONS]:[15] [COMPLEX_MOD]:[SHAKE_FI] [EFFECTIVE_SSR]:[{effective_SSR}]
                        [ANALYSIS_TYPE]:[NONLINEAR]
                        [SHEAR_TYPE]:[VELOCITY]
                        [MAX_ITERATIONS]:[5]
                        [ERROR_TOL]:[1E-05]
                        [STEP_CONTROL]:[FLEXIBLE] [MAX_STRAIN_INC]:[0] [INTERPOLATION]:[LINEAR]
                        [VISCOUS_DAMPING]:[RAYLEIGH_2] [RAYLEIGH_TYPE]:[FREQUENCIES] [f1] [f2]
                        [DAMPING_UPDATE]:[FALSE]
                        [NUM_LAYERS]:[{numb_soil_layers}]
                        [WATER_TABLE]:[{GWT_layer}]""".format(effective_SSR    = effective_SSR,
                                                              f1               = f1_ratio/self.Tn(),
                                                              f2               = f2_ratio/self.Tn(),
                                                              numb_soil_layers = numb_soil_layers,
                                                              GWT_layer        = GWT_layer
                                                             ))
        else:
            block1 = inspect.cleandoc("""
                        [FILE_VERSION]:[1]
                        [ANALYSIS_DOMAIN]:[TIME+FREQUENCY]
                            [MAX_ITERATIONS]:[15] [COMPLEX_MOD]:[SHAKE_FI] [EFFECTIVE_SSR]:[{effective_SSR}]
                        [ANALYSIS_TYPE]:[NONLINEAR]
                        [SHEAR_TYPE]:[VELOCITY]
                        [MAX_ITERATIONS]:[5]
                        [ERROR_TOL]:[1E-05]
                        [STEP_CONTROL]:[FLEXIBLE] [MAX_STRAIN_INC]:[0] [INTERPOLATION]:[LINEAR]
                        [VISCOUS_DAMPING]:[FREQ_IND]
                        [DAMPING_UPDATE]:[FALSE]
                        [NUM_LAYERS]:[{numb_soil_layers}]
                        [WATER_TABLE]:[{GWT_layer}]""".format(effective_SSR    = effective_SSR,
                                                              numb_soil_layers = numb_soil_layers,
                                                              GWT_layer        = GWT_layer
                                                             ))
            

        # Write the layer properties
        block2 = ""
        for idx in range(0,numb_soil_layers):

            text_chunk = '\n'+inspect.cleandoc("""
                [LAYER]:[{layer_number}]
                    [THICKNESS]:[{thickness}] [WEIGHT]:[{weight}] [SHEAR]:[{shear}] [SS_DAMP]:[{ss_damp}]
                    [MODEL]:[GQ] [STRENGTH]:[{strength}] [THETA1]:[{theta1}] [THETA2]:[{theta2}] [THETA3]:[{theta3}] [THETA4]:[{theta4}] [THETA5]:[{theta5}] [A]:[1]
                    [MRDF]:[UIUC] [P1]:[{p1}] [P2]:[{p2}] [P3]:[{p3}]
                    [OUTPUT]:[{output}]""".format(layer_number = idx+1,
                                              thickness    = df_soil['Thickness (m)'][idx],
                                              weight       = df_soil['Unit Weight (KN/m^3)'][idx],
                                              shear        = int(round(df_soil['Shear Wave Velocity (m/s)'][idx])),
                                              ss_damp      = round(df_soil['Dmin (%)'][idx]/1e2,2),
                                              strength     = int(round(df_soil['Shear Strength (kPa)'][idx])),
                                              theta1       = round(df_soil['theta1'][idx],2),
                                              theta2       = round(df_soil['theta2'][idx],14),
                                              theta3       = int(df_soil['theta3'][idx]),
                                              theta4       = int(df_soil['theta4'][idx]),
                                              theta5       = round(df_soil['theta5'][idx],2),
                                              p1           = round(df_soil['p1'][idx],3),
                                              p2           = round(df_soil['p2'][idx],2),
                                              p3           = round(df_soil['p3'][idx],1),
                                              output       = df_soil['Output'][idx],
                                             ))
            block2 += text_chunk

        # Write the half space properties
        block3 = '\n'+inspect.cleandoc("""
                        [LAYER]:[TOP_OF_ROCK]
                            [OUTPUT]:[FALSE]
                        [HALFSPACE]:[ELASTIC] [UNIT_WEIGHT]:[{unit_weight}] [SHEAR]:[{shear}] [DAMPING]:[{ss_damp}]
                        [RS_TYPE]:[FREQUENCY] [RS_DAMPING]:[0.05]
                        [ACCELERATION_HISTORY]:[EXTERNAL] [DEEPSOILACCELINPUT.TXT]
                        [UNITS]:[METRIC]
                        [LAYER_NAMES]:[{numb_soil_layers}]\n""".format(unit_weight      = df_half['Unit Weight (KN/m^3)'].iloc[-1],
                                                       shear            = int(round(df_half['Shear Wave Velocity (m/s)'].iloc[-1])),
                                                       ss_damp          = round(df_half['Dmin (%)'].iloc[-1]/1e2,2),
                                                       numb_soil_layers = numb_soil_layers
                                                      ))
        
        # Write the layer names
        block4 = ""
        for idx in range(0,numb_soil_layers):
            
            # Need to replace spaces with '?'
            layer_name = str(df_soil['Layer_Name'][idx])
            layer_name = layer_name.replace(" ","?")
              
            text_chunk = """[LAYER_NAMES]:[{layer_id}][{layer_name}]\n\t""".format(layer_id   = idx+1,
                                                                                   layer_name = layer_name)
            block4 += text_chunk

            
        # Combine all the fields 
        all_text = block1 + block2 + block3 + block4
        
        # Check if a user path was provided, else use root
        if DS_output_path:
            dirname = DS_output_path
        else:
            dirname = os.getcwd()

        # Make a subfolder if it doesn't exist in path
        if not os.path.exists('DEEPSOIL Input Profiles'):
            os.makedirs('DEEPSOIL Input Profiles')
           
        if DS_output_fname:
            filename = os.path.join(dirname, 'DEEPSOIL Input Profiles',DS_output_fname.strip(".dp") + ".dp")   
        else:
            filename = os.path.join(dirname, 'DEEPSOIL Input Profiles',"DeepSoil_Input.dp") 

        # Save to to a .dp file...
        file = open(filename,"w+")
        file.write(all_text)
        file.close()

        if echo:
            print(all_text)
        print("""DEEPSOIL Input profile created: {}""".format(filename))
        return
    
    

# _____________________________________________________________________________________________   
#
# Helper functions for layer randomization or describing other reference material models
#
# _____________________________________________________________________________________________    
    
def mrd_darendeli(gamma, PI, OCR, sigma_eff_vo, Ko=0.5, freq=1, N=10):
    """
    Computes Darendeli (2001) Modulus reduction and damping curves. 
    Returns a tuple containing modulus reduction, damping, and small strain 
    damping.
    
    Generic silty, sand, clays (110 samples from 20 sites)
    Applicable for 0.0001-0.5%
    _________________________________________________________
    Inputs
    -------
    gamma : strain (%)
    PI : plasticity index (%)
    OCR : over-consolidation ratio
    sigma_eff_vo : effective vertical overburden (kPa)
    K_o : effective coefficient of earth pressure at rest (=sigma_eff_vo/sigma_eff_ho)
    freq : loading frequency (Hz)
    N : number of cycles
    Returns
    -------
    MR : Modulus reduction curve
    Dfinal : Total damping curve (includes small strain damping) (%)
    Dmin : Small strain damping (%)

    """
    
    phi_1, phi_2, phi_3, phi_4, phi_5 = (0.0352, 0.0010, 0.3246, 0.3483, 0.9190)
    phi_6, phi_7, phi_8, phi_9, phi_10, phi_11, phi_12 = (0.8005, 0.0129, -0.1069, -0.2889, 0.2919, 0.6329, -0.0057)
    
    
    # Effective confining pressure
    sigma_eff_o = sigma_eff_vo*(1.+Ko)/3.
    
    
    # Modulus reduction
    gamma_r =  ((phi_1 + phi_2*PI*OCR**phi_3) * ((sigma_eff_o/101.3)**phi_4 ))
    a = phi_5
    
    MR = 1./(1.+(gamma/gamma_r)**a)   
    
    
    # Damping curve
    Dmin = (phi_6 + phi_7*PI*OCR**phi_8) * (sigma_eff_o/101.3)**phi_9 * (1+phi_10*np.log(freq)) 
    b = phi_11 + phi_12*np.log(N)
    
    c1 = -1.1143*a**2 + 1.8618*a + 0.2523
    c2 =  0.0805*a**2 - 0.0710*a - 0.0095
    c3 = -0.0005*a**2 + 0.0002*a + 0.0003
    Dmasing_a1 = 100/np.pi * (4*(gamma - gamma_r*np.log((gamma + gamma_r)/gamma_r))/(gamma**2/(gamma+gamma_r)) - 2)
    Dmasing = c1*Dmasing_a1 + c2*Dmasing_a1**2 + c3*Dmasing_a1**3

    F = b*(MR**0.1) 
    
    Dfinal = F * Dmasing + Dmin
    
    return MR, Dfinal, Dmin


def randomize_mrd_darendeli(df, n=100, sigma1=1.0, sigma2=1.0, rho_DNG=-0.5, MR_min=0.0001, MR_max=1.0, damp_min=0.1, damp_max=50, use_truncnorm=False, truncnorm_params={'n_sigma':1}):
    """
    Creates randomized profiles given a baseline modulus reduction and
    damping curve per Darendeli (2001). Assumes normally distributed by
    strain, where eps1 and eps2 are uncorrelated variables. 
    _________________________________________________________
    Inputs
    -------
    df : dataframe containing {'strain(%)','G/Gmax','damping(%)}
    n : number of realizations
    rho_DNG : strain-dependent correlation coefficient between G/Gmax and damping (=-0.5 typ.)
    sigma1 : std. deviation on eps1
    sigma2 : std. deviation on eps2
    MR_min : limits minimum G/Gmax to MR_min (0.0001 default)
    MR_max : limits maximum G/Gmax to MR_max (1.0 default)
    damp_min : limits minimum damping to damp_min (0.1 default)
    damp_max : limits maximum damping to damp_max (50 default)
    use_truncnorm : if True, uses a truncated standard normal variate for MR sampling
    truncnorm_params : dictionary of truncnorm params, e.g. n_sigma (=1 default)
    
    Returns
    -------
    dict_out : a dictionary of 'n' realizations to return for given initial reference material data
               (key corresponds to index number of realization)
    """
    strain = df['strain(%)'].to_numpy()
    MR     = df['G/Gmax'].to_numpy()
    damp   = df['damping(%)'].to_numpy()
    
    out_dict = {}
    for i in range(0,n):

        # uncorrelated random variables with zero mean, unit std. dev.
        # used for randomizing G/Gmax and Damping
        eps1 = np.random.normal(loc=0.0,scale=sigma1)
        eps2 = np.random.normal(loc=0.0,scale=sigma2)
        
        # If truncated normal variate specified, resample
        if use_truncnorm:
            n_sigma = truncnorm_params['n_sigma']
            eps1 = truncnorm.rvs(-n_sigma,n_sigma,loc=0.0, scale=sigma1)
        

        # modulus ratio std. dev.
        sigma_NG = 0.015 + 0.16*np.sqrt(0.25-(MR-0.5)**2)

        # Randomized modulus reduction
        MR_rand = MR + eps1*sigma_NG


        # damping std. dev.
        sigma_lnD = 0.0067 + 0.78*np.sqrt(damp) #(%)

        # Randomized damping
        sqrt_term = 1. - eps2 * sigma_lnD * rho_DNG**2. 
        damp_rand = damp + rho_DNG*eps1*sigma_lnD + np.sign(sqrt_term)*np.sqrt(np.abs(sqrt_term))
        #damp_rand = damp + rho_DNG*eps1*sigma_lnD + np.sqrt( 1. - eps2 * sigma_lnD * rho_DNG**2. )


        # Enforce bounds (per Rathje et al. 2010)
        MR_rand[MR_rand < MR_min] = MR_min  # Must be a small finite so that theta_tau stays finite
        MR_rand[MR_rand > MR_max] = MR_max
        damp_rand[damp_rand < damp_min] = damp_min
        damp_rand[damp_rand > damp_max] = damp_max

        df_out = pd.DataFrame(data={'strain(%)':strain,'G/Gmax':MR_rand,'damping(%)':damp_rand})
        
        out_dict[i] = df_out

    return out_dict


def mrd_menq(gamma, sigma_eff_vo, Ko=0.5, N=10, Cu=1.2, d50=1):
    """
    Computes Menq (2003) Modulus reduction and damping curves. 
    Returns a tuple containing modulus reduction, damping, and small strain 
    damping.
    
    Granular materials with d50 > ~ 0.3 mmm (59 reconstituted specimens)
    Applicable for 0.0001-0.6% strain
    _________________________________________________________
    Inputs
    -------
    gamma : strain (%)
    sigma_eff_vo : effective overburden (kPa)
    N : number of cycles
    Cu : Coefficient of uniformity = d10/d60   (Cu & d50 set to medium sand)
    d50 : Mean grain size (mm)
    
    Returns
    -------
    MR : Modulus reduction curve
    Dfinal : Total damping curve (includes small strain damping) (%)
    Dmin : Small strain damping (%)
    """
    
    phi_1, phi_2 = (-0.6, 0.382)
    phi_3, phi_4, phi_5, phi_6, phi_11, phi_12 = (0.55, 0.1, -0.3, -0.08, 0.6329, -0.0057)
    
    
    # Effective confining pressure
    sigma_eff_o = sigma_eff_vo*(1.+Ko)/3.
    
    
    # Modulus reduction
    gamma_r =  0.12 * (Cu**phi_1) * ((sigma_eff_o/101.3)**phi_2 )
    a = 0.86 + 0.1*np.log10(sigma_eff_vo/101.3)
    
    MR = 1./(1.+(gamma/gamma_r)**a)   
    
    
    # Damping curve
    Dmin = phi_3 * (Cu**phi_4) *(d50**phi_5) * (sigma_eff_o/101.3)**phi_6
    b = phi_11 + phi_12*np.log(N)
    
    c1 = -1.1143*a**2 + 1.8618*a + 0.2523
    c2 =  0.0805*a**2 - 0.0710*a - 0.0095
    c3 = -0.0005*a**2 + 0.0002*a + 0.0003
    Dmasing_a1 = 100/np.pi * (4*(gamma - gamma_r*np.log((gamma + gamma_r)/gamma_r))/(gamma**2/(gamma+gamma_r)) - 2)
    Dmasing = c1*Dmasing_a1 + c2*Dmasing_a1**2 + c3*Dmasing_a1**3

    F = b*(MR**0.1) 
    
    Dfinal = F * Dmasing + Dmin
    
    return MR, Dfinal, Dmin


class struct:
    """temporay class for storing data"""
    pass


def randomize_Vs_profile(t0,Vs0,n=100,sigma_ln_Vs=0.15,rho_o=0.97,delta=3.8,
                        rho_200=1.0,zo=0.0,b=0.293,use_site_specific_sigma=False,
                        use_thickness_randomization=False,sigma_ln_thickness=0.10,
                        use_simple_rho1L=False,rho1L_params={'case':'exp','yo':0.5,'b':3}):
    """
    Creates a randomized shear wave velocity (Vs) profile.
    ______________________________________________________________________
    Inputs
    -------
    to : layer thicknesses
    Vs0 : Baseline shear wave velocity profile
    n : number of realizations
    sigma_ln_Vs : std. dev. of Vs
    rho_o : parameter
    delta : parameter
    rho_200 : parameter
    zo : parameter
    b : parameter
    use_site_specific_sigma : if True, uses the recommended depth-dependent sigma_ln_Vs
    use_thickness_randomization : if True, randomizes layer thicknesess
    sigma_ln_thickness : std. dev. of layer thickness
    use_simple_rho1L : if True, uses a simple depth-dependent 2-parameter rho_1L fcn
    rho1L_params : rho_1L parameters, dict containing params for 'yo'-initial rho_1L, and 'b'-curvature
        
    Returns
    -------
    dat : class data struture with n Vs randomizations

    """
    def simple_rho1L(z,case='constant',yo=0.5,b=3):
        """
        Simple rho1L definition. Either assumes constant, or uses a
        2-parameter single exponential rise fcn for rho_1L
        ____________________________________________________________
        case : if 'constant', assumes constant, else uses a 2-parameter fcn
        z_norm : normalized depth = z/zmax
        yo : initial rho_1L (or constant value if case='constant')
        b : curvature parameter
        """
        if case == 'constant':
            rho_1L = yo
        else:
            a = 1-yo
            rho_1L = yo + a*(1-np.exp(-b*z))
        return rho_1L
    
    
    # Initialize variables
    numb_soil_layers = len(Vs0)

    
    dat = struct()
    dat.t0      = t0
    dat.Vs0     = Vs0
    dat.z_bot0  = np.cumsum(dat.t0)
    dat.z_top0  = np.concatenate((np.zeros(1), dat.z_bot0[:-1]))
    dat.z0      = 0.5*(dat.z_top0 + dat.z_bot0)
    
    dat.t       = np.array([dat.t0,]*n)
    dat.z_top   = np.array([dat.z_top0,]*n)
    dat.z_bot   = np.array([dat.z_bot0,]*n)
    dat.z       = np.array([dat.z0,]*n)
    
    dat.epsilon = np.zeros((n,numb_soil_layers))
    dat.rho_t   = np.zeros((n,numb_soil_layers))
    dat.rho_z   = np.zeros((n,numb_soil_layers))
    dat.rho_1L  = np.zeros((n,numb_soil_layers))
    dat.Z       = np.zeros((n,numb_soil_layers))
    dat.Vs      = np.zeros((n,numb_soil_layers))
    
    # Set standard deviation profile
    dat.sigma_ln_Vs = np.zeros(numb_soil_layers)
    if use_site_specific_sigma: # Site-specific (depth-dependent)
        dat.sigma_ln_Vs[dat.z <= 50] = 0.15
        dat.sigma_ln_Vs[dat.z > 50] = 0.22
    else: # Constant
        dat.sigma_ln_Vs[:] = sigma_ln_Vs
    
    for k in range(0,n): # Loop on realizations
        
        # Layer randomization, if specified
        if use_thickness_randomization: 
            # Sample normal random variable for depths
            epsilon_thick = np.random.normal(loc=0.0, scale=sigma_ln_thickness, size=np.shape(dat.t)[1])
    
            # Adjust layer thicknesses
            dat.t[k,:] = np.exp(np.log(dat.t0) + epsilon_thick*sigma_ln_thickness)
    
            # Update depths for variable thickness
            dat.z_bot[k,:]   = np.cumsum(dat.t[k,:])
            dat.z_top[k,:]   = np.concatenate((np.zeros(1), dat.z_bot[k,:-1]))
            dat.z[k,:]       = 0.5*(dat.z_top[k,:] + dat.z_bot[k,:])
        
        # Vs randomization
        # Note: This calcuation can be vectorized by returning an array of epsilons
        # and sub-dividing the calculation for the first element from the 2nd to last.
        for i in range(0,numb_soil_layers): # Loop on layers
    
            # Sample normal random variable with zero mean and unit std. dev.
            dat.epsilon[k,i] = np.random.normal(loc=0.0,scale=1.0)
    
            # Interlayer correlation coefficient (rho_1L)
            if use_simple_rho1L: # Use simple expression for rho_1L (with depth)
                
                dat.rho_1L[k,i] = simple_rho1L(dat.z[k,i]/dat.z[k,:].max(),
                                               case=rho1L_params['case'],
                                               yo=rho1L_params['yo'],
                                               b=rho1L_params['b'])
                
            else: # Use standard expression for rho_1L (with thickness/depth terms)
                
                # Thickness correlation
                dat.rho_t[k,i] = rho_o*np.exp(-dat.t[k,i]/delta)
    
                # Depth correlation
                if dat.z[k,i] <= 200: #[m]
                    dat.rho_z[k,i] = rho_200 * np.power((dat.z[k,i]+zo)/(200+zo),b)
                else:
                    dat.rho_z[k,i] = rho_200
                
                dat.rho_1L[k,i] = (1-dat.rho_z[k,i])*dat.rho_t[k,i] + dat.rho_z[k,i]
    
            # Random standard normal variable
            if i == 0: # for surface layer
                dat.Z[k,i] = dat.epsilon[k,i]
            else: # for all other layers
                dat.Z[k,i] = dat.rho_1L[k,i]*dat.Z[k,i-1] + dat.epsilon[k,i]*np.sqrt(1-dat.rho_1L[k,i]**2)
    
            # Shear wave velocity of layer i for given realization
            dat.Vs[k,i] = np.exp(np.log(dat.Vs0[i]) + dat.Z[k,i]*dat.sigma_ln_Vs[i])
    return dat


def estimate_params(Vs30):
    """
    Returns Toro model parameters by Vs30.
    After Rathje et al. (2010)

    Returns
    -------
    params : dictionary with model parameters based on Vs30

    """
    if Vs30 < 180:
        sigma_ln_Vs, rho_200, zo, b, rho_o, delta = 0.37, 0.50, 0.0, 0.744, 0.00, 5.0
    elif Vs30 < 360:
        sigma_ln_Vs, rho_200, zo, b, rho_o, delta = 0.31, 0.98, 0.0, 0.344, 0.99, 3.9
    elif Vs30 < 750:
        sigma_ln_Vs, rho_200, zo, b, rho_o, delta = 0.27, 1.00, 0.0, 0.293, 0.97, 3.8
    else:
        sigma_ln_Vs, rho_200, zo, b, rho_o, delta = 0.36, 0.42, 0.0, 0.063, 0.95, 3.4
    
    params = {'sigma_ln_Vs': sigma_ln_Vs,
              'rho_o': rho_o,
              'delta': delta,
              'rho_200': rho_200, 
              'zo': zo,
              'b': b}
    return params


def randomize_Su_profile(Su0,dat,seed=100005,cov=0.10,rho=0.75,Su_min=5):
    """
    **SUPERCEDED by randomize_from_seed_profile**
    Creates a randomized undrained shear strength (Su) profile, based on a 
    joint distribution with Vs. 
    ______________________________________________________________________
    Inputs
    -------
    Su0 : baseline or mean Su profile
    dat : data containining Vs randomization data
    n : number of realizations
    seed : seed for random number generator
    cov : Su coefficient of variation; can be updated to vary with depth
    rho : correlation coefficient rho(ln(Vs),rho); can be updated to vary with depth
    Su_min : minimum cap for Su [kPa] (5kPa default)
        
    Returns
    -------
    dat : class data struture with n Vs randomizations
    """

    # Fixing random state for reproducibility
    np.random.seed(seed)
    
    # Add the baseline (mean) Su, sigma to dat
    dat.Su0 = Su0[:-1].to_numpy()
    dat.sigma_Su = dat.Su0*cov

    # Sample normal variates for Su realizations based on joint bivariate distribution with correlation coefficient
    dat.epsilon2 = np.random.normal(loc=0.0,scale=1.0,size=dat.epsilon.shape)
    dat.rho = np.ones(dat.Su0.shape)*rho
    dat.epsilon_su = rho*dat.Z + np.sqrt(1-dat.rho**2)*dat.epsilon2

    # Compute and add the Su realization data to dat
    Su = np.repeat([dat.Su0],repeats=1000,axis=0)
    dat.Su = Su + dat.epsilon_su*(Su*cov)
    
    # Apply minimum Su cap; mostly just to ensure that Su won't be zero
    dat.Su[dat.Su < Su_min] = Su_min
    
    return dat


def randomize_from_seed_profile(baseline_profile,N_realiz=10,seed=1001,toro_sigma_ln_Vs=0.05,toro_sigma_ln_thickness=0.0,
                                toro_use_thickness_randomization=True,toro_use_site_specific_sigma=False,
                                use_simple_rho1L=False,rho1L_params={'case':'expo','yo':0.5,'b':3},
                                mrd_sigma1=1.0,mrd_sigma2=1.0,mrd_rho_DNG=-0.5,
                                MR_min=0.0001,MR_max=1.0, damp_min=0.1,damp_max=50,
                                use_truncnorm=False,truncnorm_params={'n_sigma':1},
                                min_shear_cap=5):
    """
    Randomizes the seed/baseline profile for thickness, Vs, shear strength
    and layer modulus reduction and damping curves.
    
    # NOTE: Should move this and associated fcns to the profile class later.
    _________________________________________________________
    Inputs
    -------
    baseline_profile : baseline/seed profile
    N_realize : number of realizations to generate
    seed : seed for random number generator for predictability
    
    TORO PARAMETERS (primary)
    toro_sigma_ln_Vs : std. deviation for layer Vs (= 0.15 default)
    toro_sigma_ln_thickness : std. deviation for layer thickness (= 0.3 default)
    toro_use_site_specific_sigma : if True, uses the recommended depth-dependent sigma_ln_Vs
                                   (e.g. Stewart and Hashash 2014);vdoes not use toro_sigma_ln_Vs 
    toro_use_thickness_randomization : if True, randomizes layer thicknesess

    
    DARENDELI PARAMETERS
    mrd_sigma1 : darendeli MR sigma (= 1.0 default)
    mrd_sigma2 : darendeli damping sigma (= 1.0 default)
    mrd_rho_DNG : darendeli correlation coefficient between MR and damping (= -0.5 default)
    
    Returns
    -------
    realizations : list of N_realize realizations
    dat_Vs_rand :  data structure, fields are MxN, where M corresponds to numb layers, N to numb realizations
    dat_MRD_rand : nested dict of dataframes, dat_MRD_rand[i][j] refers to ith layer, jth realization
    dat_shear_rand : fields are MxN, where M corresponds to numb layers, N to numb realizations
    """
    # Set our seed for predictability
    np.random.seed(seed)


    # Sample N realizations...
    # ..For layer thickness and Vs..
    dat_Vs_rand = randomize_Vs_profile([layer.thickness for layer in baseline_profile.layers][:-1],
                                       [layer.Vs for layer in baseline_profile.layers][:-1],
                                       n=N_realiz,
                                       sigma_ln_Vs=toro_sigma_ln_Vs,
                                       sigma_ln_thickness=toro_sigma_ln_thickness,
                                       use_site_specific_sigma=toro_use_site_specific_sigma,
                                       use_thickness_randomization=toro_use_thickness_randomization,
                                       use_simple_rho1L=use_simple_rho1L,
                                       rho1L_params=rho1L_params,
                                       )    
    
    
    #     # ..For shear strength.. (append to same data struct as Vs randomization) - NORMAL DISTRIBUTION
    #     dat_Vs_rand.Su0      = [layer.shear_strength for layer in baseline_profile.layers][:-1]
    #     dat_Vs_rand.sigma_Su = [layer.shear_strength*layer.shear_cov for layer in baseline_profile.layers][:-1]
    #     dat_Vs_rand.corr     = [layer.corr for layer in baseline_profile.layers][:-1]

    #     dat_Vs_rand.Su       = np.ones(dat_Vs_rand.Vs.shape) # Initialize
    #     for idx, layer in enumerate(baseline_profile.layers[:-1]):
    #         epsilon1 = dat_Vs_rand.Z[:,idx]                                      # Get the ln(Vs) random normal variate for all realizations..
    #         epsilon2 = np.random.normal(loc=0.0,scale=1.0,size=epsilon1.shape)   # Sample for Su (normal dist.)
    #         epsilon_su = layer.corr*epsilon1 + np.sqrt(1-layer.corr**2)*epsilon2 # Assume joint bivariate-normal distribution with correlation coeff.
    #         sigma_su = layer.shear_strength*layer.shear_cov                      # Compute sigma
    #         dat_Vs_rand.Su[:,idx] = np.maximum(min_shear_cap, layer.shear_strength + epsilon_su*sigma_su)


    # Would like to set this up later on for uniform, normal, or log-normal distribution

    # ..For shear strength.. (append to same data struct as Vs randomization) - UNIFORM DISTRIBUTION
    dat_Vs_rand.Su0      = [0.5*(layer.shear_str_min+layer.shear_str_max) for layer in baseline_profile.layers][:-1]
    dat_Vs_rand.sigma_Su = [(layer.shear_str_max-layer.shear_str_min)/(2*np.sqrt(3)) for layer in baseline_profile.layers][:-1]
    dat_Vs_rand.corr     = [layer.corr for layer in baseline_profile.layers][:-1]

    dat_Vs_rand.Su       = np.ones(dat_Vs_rand.Vs.shape) # Initialize
    for idx, layer in enumerate(baseline_profile.layers[:-1]):
        # Get the ln(Vs) random normal variate for all realizations..
        epsilon1 = dat_Vs_rand.Z[:,idx]                                      

        #  Compute mean and std dev for uniform dist..
        mean_su =  0.5*(layer.shear_str_min+layer.shear_str_max)
        sigma_su = (layer.shear_str_max-layer.shear_str_min)/(2*np.sqrt(3))

        # Back-calculate standard uniform variate knowing mean and sigma..
        epsilon2 = (np.random.uniform(layer.shear_str_min,layer.shear_str_max,size=epsilon1.shape)-mean_su)/sigma_su

        # Compute epsilon_su assuming joint bivariate-normal/uniform distribution with correlation coeff.
        epsilon_su = layer.corr*epsilon1 + np.sqrt(1-layer.corr**2)*epsilon2 

        # Compute Su
        dat_Vs_rand.Su[:,idx] = np.maximum(min_shear_cap, mean_su + epsilon_su*sigma_su)
        #dat_Vs_rand.Su[:,idx] = np.maximum(min_shear_cap, np.random.uniform(layer.shear_str_min,layer.shear_str_max,size=epsilon1.shape)) # check for uniform
        
    
    # ..For modulus reduction and damping..
    dat_MRD_rand = {}
    for idx, layer in enumerate(baseline_profile.layers[:-1]):
        dat_MRD_rand[idx] = randomize_mrd_darendeli(layer.data['reference'],
                                                    n=N_realiz,
                                                    sigma1=mrd_sigma1,
                                                    sigma2=mrd_sigma2,
                                                    rho_DNG=mrd_rho_DNG,
                                                    MR_min=MR_min,
                                                    MR_max=MR_max,
                                                    damp_min=damp_min,
                                                    damp_max=damp_max,
                                                    use_truncnorm=use_truncnorm,
                                                    truncnorm_params=truncnorm_params)

        
        
    # Create a list of N different realizations/profile classes; each profile is assigned different layer
    # Vs, thickness, strength, and G/Gmax properties for a given realization, calibrated, and saved
    realizations = []
    for i in range(0,N_realiz):

        # Make a copy of our baseline profile
        profile_i = copy.deepcopy(baseline_profile)

        # Thickness and Vs randomization
        # Note: Gmax re-calculated by layer class during calib.
        for j, (Vs, thickness) in enumerate(zip(dat_Vs_rand.Vs[i],dat_Vs_rand.t[i])):
            profile_i.layers[j].Vs = Vs
            profile_i.layers[j].thickness = thickness

        # Soil property G/Gmax and damping randomization
        for j in range(0,len(baseline_profile.layers)-1):
            profile_i.layers[j].data['reference'] = dat_MRD_rand[j][i]

        # Shear strength randomization
        for j, shear_strength in enumerate(dat_Vs_rand.Su[i]):
            profile_i.layers[j].shear_strength = shear_strength

        realizations.append(profile_i)
    
    return realizations, dat_Vs_rand, dat_MRD_rand



def randomize_from_seed_profile2(baseline_profile,reference_database,N_realiz=10,seed=1001,toro_sigma_ln_Vs=0.05,toro_sigma_ln_thickness=0.0,
                                toro_use_thickness_randomization=True,toro_use_site_specific_sigma=False,
                                use_simple_rho1L=False,rho1L_params={'case':'exp','yo':0.5,'b':3},
                                mrd_sigma1=1.0,mrd_sigma2=1.0,mrd_rho_DNG=-0.5,
                                MR_min=0.0001,MR_max=1.0, damp_min=0.1,damp_max=50,
                                use_truncnorm=False,truncnorm_params={'n_sigma':1},
                                min_shear_cap=5,recalc_OCR_Ko=False,Snc=0.22,m=0.8,Su_fac=1.0):
    """
    Randomizes the seed/baseline profile for thickness, Vs, shear strength
    and layer modulus reduction and damping curves.
    
    # NOTE: Should move this and associated fcns to the profile class later.
    _________________________________________________________
    Inputs
    -------
    baseline_profile : baseline/seed profile
    N_realize : number of realizations to generate
    seed : seed for random number generator for predictability
    
    TORO PARAMETERS (primary)
    toro_sigma_ln_Vs : std. deviation for layer Vs (= 0.15 default)
    toro_sigma_ln_thickness : std. deviation for layer thickness (= 0.3 default)
    toro_use_site_specific_sigma : if True, uses the recommended depth-dependent sigma_ln_Vs
                                   (e.g. Stewart and Hashash 2014);vdoes not use toro_sigma_ln_Vs 
    toro_use_thickness_randomization : if True, randomizes layer thicknesess

    STRENGTH PARAMETERS
    Note: randomization fields are specified in the spreadsheet (e.g. COV, correlation coeffient, Min./Max.)
    min_shear_cap : minimum shearing resistance cap [kPa]
    recalc_OCR_Ko : if True, re-calculates Ko, and OCR based on the new randomized Su
    Snc : normally-consolidated undrained strength ratio (Su/sigma'vo)NC (0.22 default for simple-shear)
    m : OCR exponent (0.8 default for simple shear)
    Su_fac : scaling factor to multiply Su by before re-calculating OCR (1.0 default)
             e.g., if input shear strength uses an implied 20% overstrength for eqk rate effects, use an Su_fac = 0.8 to calculate static OCR, Ko
    
    DARENDELI PARAMETERS
    mrd_sigma1 : darendeli MR sigma (= 1.0 default)
    mrd_sigma2 : darendeli damping sigma (= 1.0 default)
    mrd_rho_DNG : darendeli correlation coefficient between MR and damping (= -0.5 default)
    MR_min : minimum MR limit (0.0001 default)
    MR_max : maximum MR limit (1.0 default)
    damp_min : minimum damp limit (0.1 default)
    damp_max : maximum damp limit (50 default)
    use_truncnorm : if True, uses a truncated normal variate for sampling MR with n_sigma defined in truncnorm_params
    truncnorm_params : dict of params for truncnorm, e.g. {'n_sigma':1}, n_sigma denotes the truncation +/-bounds
    
    Returns
    -------
    realizations : list of N_realize realizations
    dat_Vs_rand :  data structure, fields are MxN, where M corresponds to numb layers, N to numb realizations
    dat_MRD_rand : nested dict of dataframes, dat_MRD_rand[i][j] refers to ith layer, jth realization
    dat_shear_rand : fields are MxN, where M corresponds to numb layers, N to numb realizations
    """
    # Create Nrealizations/profiles
    realizations = [copy.deepcopy(baseline_profile) for i in range(0,N_realiz)]
    
    # Set our seed for predictability
    np.random.seed(seed)
    
    # Sample N realizations...
    # ..for layer thickness and Vs..
    dat_Vs_rand = randomize_Vs_profile([layer.thickness for layer in baseline_profile.layers][:-1],
                                       [layer.Vs for layer in baseline_profile.layers][:-1],
                                       n=N_realiz,
                                       sigma_ln_Vs=toro_sigma_ln_Vs,
                                       sigma_ln_thickness=toro_sigma_ln_thickness,
                                       use_site_specific_sigma=toro_use_site_specific_sigma,
                                       use_thickness_randomization=toro_use_thickness_randomization,
                                       use_simple_rho1L=use_simple_rho1L,
                                       rho1L_params=rho1L_params,
                                       )    

    # ..for shear strength.. (append to same data struct as Vs randomization) - UNIFORM DISTRIBUTION
    dat_Vs_rand.Su0      = [0.5*(layer.shear_str_min+layer.shear_str_max) for layer in baseline_profile.layers][:-1]
    dat_Vs_rand.sigma_Su = [(layer.shear_str_max-layer.shear_str_min)/(2*np.sqrt(3)) for layer in baseline_profile.layers][:-1]
    dat_Vs_rand.corr     = [layer.corr for layer in baseline_profile.layers][:-1]

    dat_Vs_rand.Su       = np.ones(dat_Vs_rand.Vs.shape) # Initialize
    for idx, layer in enumerate(baseline_profile.layers[:-1]):
        # Get the ln(Vs) random normal variate for all realizations..
        epsilon1 = dat_Vs_rand.Z[:,idx]                                      

        #  Compute mean and std dev for uniform dist..
        mean_su =  0.5*(layer.shear_str_min+layer.shear_str_max)
        sigma_su = (layer.shear_str_max-layer.shear_str_min)/(2*np.sqrt(3))

        # Back-calculate standard uniform variate knowing mean and sigma..
        epsilon2 = (np.random.uniform(layer.shear_str_min,layer.shear_str_max,size=epsilon1.shape)-mean_su)/sigma_su

        # Compute epsilon_su assuming joint bivariate-normal/uniform distribution with correlation coeff.
        epsilon_su = layer.corr*epsilon1 + np.sqrt(1-layer.corr**2)*epsilon2 

        # Compute Su
        dat_Vs_rand.Su[:,idx] = np.maximum(min_shear_cap, mean_su + epsilon_su*sigma_su)
        
    
    # Assign Vs, thickness, and Su
    for i, profile_i in enumerate(realizations):

        # Thickness and Vs randomization
        # Note: Gmax re-calculated by layer class during calib.
        for j, (Vs, thickness) in enumerate(zip(dat_Vs_rand.Vs[i],dat_Vs_rand.t[i])):
            profile_i.layers[j].Vs = Vs
            profile_i.layers[j].thickness = thickness

        # Shear strength randomization
        for j, shear_strength in enumerate(dat_Vs_rand.Su[i]):
            profile_i.layers[j].shear_strength = shear_strength
            
        # Update OCR and Ko
        # Note: assume Snc, m = 0.22, 0.8 default for simple-shear; use Ladd et al. for estimating Ko,oc
        if recalc_OCR_Ko:
            for j in range(0,len(dat_Vs_rand.Su[i])):
                profile_i.layers[j].OCR = ((profile_i.layers[j].shear_strength*Su_fac/profile_i.layers[j].sigma_vo_eff)/Snc)**(1/m)
                profile_i.layers[j].Ko = (0.0054*profile_i.layers[j].PI+0.4207)*profile_i.layers[j].OCR**0.42
            
        # Update MRD reference data
        profile_i.assign_reference_models(reference_database,echo=False)
        

    # ..Now randomize for modulus reduction and damping..
    # Note: since Ko and OCR can potentially vary with each profile realization, we now need to sample each
    # grid point seperately..
    dat_MRD_rand = {} # initialize; this will end up being a nested dict of dataframes, i.e. dat_MRD_rand[layer key][realization key]
    for i in range(0,len(baseline_profile.layers[:-1])):
        MRD_realization = {} # initialize
        for j, profile_i in zip(range(0,N_realiz),realizations):
            # Recall randomize_mrd_darendeli returns a dict of N realizations
            MRD_realization[j] = randomize_mrd_darendeli(profile_i.layers[i].data['reference'], # Append to dict
                                                        n=1,
                                                        sigma1=mrd_sigma1,
                                                        sigma2=mrd_sigma2,
                                                        rho_DNG=mrd_rho_DNG,
                                                        MR_min=MR_min,
                                                        MR_max=MR_max,
                                                        damp_min=damp_min,
                                                        damp_max=damp_max,
                                                        use_truncnorm=use_truncnorm,
                                                        truncnorm_params=truncnorm_params)[0] # Zero index to extract df
        # Append the realizations to the layer
        dat_MRD_rand[i] = MRD_realization

            
    # Assign MRD randomizations
    for profile_i in realizations:
        
        # Soil property G/Gmax and damping randomization
        for j in range(0,len(profile_i.layers)-1):
            profile_i.layers[j].data['reference'] = dat_MRD_rand[j][i]

    return realizations, dat_Vs_rand, dat_MRD_rand


def plot_detail_randomization(baseline_profile,realizations,figsize=(14,7),fs=10,grid=True,save_fig=False,ext=['.pdf','.png']):
    """
    Plots detailed profile randomizations for Vs, Su, OCR, and Ko
    _________________________________________________________
    Inputs
    -------
    baseline_profile : baseline/seed profile
    realizations : nested dictionary of all realizations i,j corresponding to layer,realiz
    """
    # Set up axes
    fig, ax = plt.subplots(1,4, figsize=figsize)
    N_realiz = len(realizations)
    colors = [cm.tab20c(x) for x in np.linspace(0,1,N_realiz)]
    
    # Vs
    for real,color in zip(realizations,colors):
        ax[0].plot(np.repeat(real.Vs(),2),
                 np.dstack((real.depth_top(),real.depth_bott())).flatten(),
                '-',color=color)
    ax[0].plot(np.repeat(baseline_profile.Vs(),2),
                 np.dstack((baseline_profile.depth_top(),baseline_profile.depth_bott())).flatten(),
           'k',label='Baseline')
    
    # Shear strength
    for real,color in zip(realizations,colors):
        ax[1].plot(np.repeat(real.shear_strength()/1e3,2),
                 np.dstack((real.depth_top(),real.depth_bott())).flatten(),
                '-',color=color)
    ax[1].plot(np.repeat(baseline_profile.shear_strength()/1e3,2),
                 np.dstack((baseline_profile.depth_top(),baseline_profile.depth_bott())).flatten(),
           'k',label='Baseline')

    # OCR
    for real,color in zip(realizations,colors):
        ax[2].semilogx(np.repeat(real.OCR(),2),
                 np.dstack((real.depth_top(),real.depth_bott())).flatten(),
                '-',color=color)
    ax[2].semilogx(np.repeat(baseline_profile.OCR(),2),
                 np.dstack((baseline_profile.depth_top(),baseline_profile.depth_bott())).flatten(),
           'k',label='Baseline')
    
    # Ko
    for real,color in zip(realizations,colors):
        ax[3].plot(np.repeat(real.Ko(),2),
                 np.dstack((real.depth_top(),real.depth_bott())).flatten(),
                '-',color=color)
    ax[3].plot(np.repeat(baseline_profile.Ko(),2),
                 np.dstack((baseline_profile.depth_top(),baseline_profile.depth_bott())).flatten(),
           'k',label='Baseline')
    
    # Formatting
    if grid:
        [axi.grid(alpha=0.2) for axi in ax]
    [axi.invert_yaxis() for axi in ax]
    
    ax[0].legend()
    
    ax[0].set_ylabel(r'$Depth$ $(m)$')
    [axi.set_xlabel(xlab,fontsize=fs) for axi, xlab in zip(ax,[r'$V_s$ $(m/s)$',r'$\tau_{max}$ $(kPa)$',r'$OCR$',r'$K_O}$'])]
    
    [axi.set_ylim(top=0.0) for axi in ax]
    ax[2].set_xlim(left=1.0)
    ax[3].set_xlim(left=0.0)
    
    ax[2].xaxis.set_major_formatter(ScalarFormatter())
    ax[2].set_xticks(np.array([1,10,100]))
    
    [axi.xaxis.tick_top() for axi in ax]
    [axi.xaxis.set_label_position('top') for axi in ax]
    
    if save_fig:
        [fig.savefig('Fig_Randomizations'+ext,bbox_inches='tight',dpi=400) for ext in ext]
    plt.show()
    return


def plot_Vs_randomization(baseline_profile,realizations):
    """
    Plots Vs randomizations
    _________________________________________________________
    Inputs
    -------
    baseline_profile : baseline/seed profile
    realizations : nested dictionary of all realizations i,j corresponding to layer,realiz
    """
    N_realiz = len(realizations)
    
    # Plot Vs-thickness realizations
    fig, ax = plt.subplots(figsize=(4,6))
    colors = [cm.tab20c(x) for x in np.linspace(0,1,N_realiz)]
    for real,color in zip(realizations,colors):
        ax.plot(np.repeat(real.Vs(),2),
                 np.dstack((real.depth_top(),real.depth_bott())).flatten(),
                '-',color=color)
    ax.plot(np.repeat(baseline_profile.Vs(),2),
                 np.dstack((baseline_profile.depth_top(),baseline_profile.depth_bott())).flatten(),
           'k',label='Baseline')
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel(r'$V_s$ $(m/s)$')
    plt.ylabel(r'$Depth$ $(m)$')
    ax.set_ylim(top=0.0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.show()
    
    return


def plot_shear_randomization(baseline_profile,realizations):
    """
    Plots shear strength randomizations
    _________________________________________________________
    Inputs
    -------
    baseline_profile : baseline/seed profile
    realizations : nested dictionary of all realizations i,j corresponding to layer,realiz
    """
    N_realiz = len(realizations)
    
    # Plot Shear strength variation
    fig, ax = plt.subplots(figsize=(4,6))
    colors = [cm.tab20c(x) for x in np.linspace(0,1,N_realiz)]
    for real,color in zip(realizations,colors):
        ax.plot(np.repeat(real.shear_strength()/1e3,2),
                 np.dstack((real.depth_top(),real.depth_bott())).flatten(),
                '-',color=color)
    ax.plot(np.repeat(baseline_profile.shear_strength()/1e3,2),
                 np.dstack((baseline_profile.depth_top(),baseline_profile.depth_bott())).flatten(),
           'k',label='Baseline')
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel(r'$\tau_{max}$ $(kPa)$')
    plt.ylabel(r'$Depth$ $(m)$')
    ax.set_ylim(top=0.0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.show()
    return


def preview_modulus_randomization(baseline_profile,realizations,dat2,nrows=5,ncols=6,plot_mean=False,plot_1sigma=False):
    """
    Generate a matrix plot preview of randomized modulus reduction curves. 
    ______________________________________________________________________
    Inputs
    -------
    baseline_profile : baseline/seed profile
    realizations : nested dictionary of all realizations i,j corresponding to layer,realiz
    nrows : number of rows per plate
    ncols : number of columns per plate
    plot_mean : if True, plots mean reference curve
    plot_1sigma : if True, plots mean reference curve +/- 1sigma
    """
    N_realiz = len(realizations)
    colors = [cm.tab20c(x) for x in np.linspace(0,1,N_realiz)]
    
    numb_plate = int(np.ceil(len(baseline_profile.layers)/(nrows*ncols)))
    for idx_plate in range(0,numb_plate):

        # Parse the layer start/stop numbers onto plates
        layer_start = idx_plate*nrows*ncols
        layer_end   = (idx_plate+1)*nrows*ncols
        if layer_end > len(baseline_profile.layers): # less than a full plate condition
            layer_end = len(baseline_profile.layers)

        fig = plt.figure(figsize=(11,8.5))
        fig.suptitle('G/Gmax Curves for all Realizations',x=0.5,y=0.92)
        ax = [plt.subplot(nrows,ncols,i+1) for i in range(nrows*ncols)]
        plt.subplots_adjust(wspace=0, hspace=0)

        for idx, layer in enumerate(baseline_profile.layers[layer_start:layer_end]):

            if layer.data['fit']['theta']: # Gets around undefined rock layer

                ax[idx].text(0.15,0.1,"""layer {}""".format(layer.layer_ID),
                             verticalalignment='bottom',horizontalalignment='left',
                             transform=ax[idx].transAxes)
                # Remove labels/ticks
                for a in ax:
                    a.set_xticklabels([])
                    a.set_yticklabels([])

                for key, color in zip(dat2[idx].keys(),colors):

                    ax[idx].semilogx(dat2[idx][key]['strain(%)'],
                                     dat2[idx][key]['G/Gmax'],
                                    '-',color=color)
                    
                # Plot the mean and sigma curves
                if plot_mean:
                    ax[idx].semilogx(layer.data['reference']['strain(%)'],
                                     layer.data['reference']['G/Gmax'],
                                    '-',color='k')
                if plot_1sigma:
                    MR = layer.data['reference']['G/Gmax']
                    sigma_NG = 0.015 + 0.16*np.sqrt(0.25-(MR-0.5)**2)
                    ax[idx].semilogx(layer.data['reference']['strain(%)'],
                                     layer.data['reference']['G/Gmax']+sigma_NG,
                                    '--',color='k') 
                    ax[idx].semilogx(layer.data['reference']['strain(%)'],
                                     layer.data['reference']['G/Gmax']-sigma_NG,
                                    '--',color='k') 
    return




def preview_damping_randomization(baseline_profile,realizations,dat2,nrows=5,ncols=6,plot_mean=False,plot_1sigma=False):
    """
    Generate a matrix plot preview of randomized modulus reduction curves. 
    ______________________________________________________________________
    Inputs
    -------
    baseline_profile : baseline/seed profile
    realizations : nested dictionary of all realizations i,j corresponding to layer,realiz
    nrows : number of rows per plate
    ncols : number of columns per plate
    plot_mean : if True, plots mean reference curve
    plot_1sigma : if True, plots mean reference curve +/- 1sigma
    """
    N_realiz = len(realizations)
    colors = [cm.tab20c(x) for x in np.linspace(0,1,N_realiz)]
    
    numb_plate = int(np.ceil(len(baseline_profile.layers)/(nrows*ncols)))
    for idx_plate in range(0,numb_plate):

        # Parse the layer start/stop numbers onto plates
        layer_start = idx_plate*nrows*ncols
        layer_end   = (idx_plate+1)*nrows*ncols
        if layer_end > len(baseline_profile.layers): # less than a full plate condition
            layer_end = len(baseline_profile.layers)

        fig = plt.figure(figsize=(11,8.5))
        fig.suptitle('Damping Curves for all Realizations',x=0.5,y=0.92)
        ax = [plt.subplot(nrows,ncols,i+1) for i in range(nrows*ncols)]
        plt.subplots_adjust(wspace=0, hspace=0)

        for idx, layer in enumerate(baseline_profile.layers[layer_start:layer_end]):

            if layer.data['fit']['theta']: # Gets around undefined rock layer

                ax[idx].text(0.15,0.80,"""layer {}""".format(layer.layer_ID),
                             verticalalignment='bottom',horizontalalignment='left',
                             transform=ax[idx].transAxes)
                # Remove labels/ticks
                for a in ax:
                    a.set_xticklabels([])
                    a.set_yticklabels([])

                for key, color in zip(dat2[idx].keys(),colors):

                    ax[idx].semilogx(dat2[idx][key]['strain(%)'],
                                     dat2[idx][key]['damping(%)'],
                                    '-',color=color)
                
                # Plot the mean and sigma curves
                if plot_mean:
                    ax[idx].semilogx(layer.data['reference']['strain(%)'],
                                     layer.data['reference']['damping(%)'],
                                    '-',color='k')
                if plot_1sigma:
                    damp = layer.data['reference']['damping(%)']
                    sigma_lnD = 0.0067 + 0.78*np.sqrt(damp) #(%)
                    ax[idx].semilogx(layer.data['reference']['strain(%)'],
                                     layer.data['reference']['damping(%)']+sigma_lnD,
                                    '--',color='k') 
                    ax[idx].semilogx(layer.data['reference']['strain(%)'],
                                     layer.data['reference']['damping(%)']-sigma_lnD,
                                    '--',color='k')     
    return



def preview_mrd_randomization_detailed(plot_layers,baseline_profile,realizations,dat2,nrows=5,ncols=6,plot_mean=False,plot_1sigma=False,figsize=(11,8.5),save_fig=False,
                                       title='Modulus Reduction and Damping Curves',title_xy=(0.5,0.92)):
    """
    Generate a detailed preview of mrd randomizations
    ______________________________________________________________________
    Inputs
    -------
    baseline_profile : baseline/seed profile
    realizations : nested dictionary of all realizations i,j corresponding to layer,realiz
    nrows : number of rows per plate
    ncols : number of columns per plate
    plot_mean : if True, plots mean reference curve
    plot_1sigma : if True, plots mean reference curve +/- 1sigma
    """
    N_realiz = len(realizations)
    colors = [cm.tab20c(x) for x in np.linspace(0,1,N_realiz)]
    
    fig, axes = plt.subplots(len(plot_layers),2,figsize=figsize)
    axes = axes.flatten()
    fig.suptitle(title,x=title_xy[0],y=title_xy[1],fontstyle='italic')
    
    for idx, layer in zip(np.arange(0,len(plot_layers)*2,2),[baseline_profile.layers[i] for i in plot_layers]):

        if layer.data['fit']['theta']: # Gets around undefined rock layer

            # Layer tag
            axes[idx].text(0.75,0.85,"""Layer {}""".format(layer.layer_ID),fontsize=12,fontstyle='italic',
                         verticalalignment='bottom',horizontalalignment='left',
                         transform=axes[idx].transAxes)

            
            # MR curves...
            for key, color in zip(dat2[idx].keys(),colors):

                axes[idx].semilogx(dat2[idx][key]['strain(%)'],
                                 dat2[idx][key]['G/Gmax'],
                                '-',color=color)
            # Plot the mean and sigma curves
            if plot_mean:
                axes[idx].semilogx(layer.data['reference']['strain(%)'],
                                 layer.data['reference']['G/Gmax'],
                                '-',color='k')
            if plot_1sigma:
                MR = layer.data['reference']['G/Gmax']
                sigma_NG = 0.015 + 0.16*np.sqrt(0.25-(MR-0.5)**2)
                axes[idx].semilogx(layer.data['reference']['strain(%)'],
                                 layer.data['reference']['G/Gmax']+sigma_NG,
                                '--',color='k') 
                axes[idx].semilogx(layer.data['reference']['strain(%)'],
                                 layer.data['reference']['G/Gmax']-sigma_NG,
                                '--',color='k') 

                
            # Damping curves...
            for key, color in zip(dat2[idx].keys(),colors):

                axes[idx+1].semilogx(dat2[idx][key]['strain(%)'],
                                 dat2[idx][key]['damping(%)'],
                                '-',color=color)
            # Plot the mean and sigma curves
            if plot_mean:
                axes[idx+1].semilogx(layer.data['reference']['strain(%)'],
                                 layer.data['reference']['damping(%)'],
                                '-',color='k',label='Target $\mu$')
            if plot_1sigma:
                damp = layer.data['reference']['damping(%)']
                sigma_lnD = 0.0067 + 0.78*np.sqrt(damp) #(%)
                axes[idx+1].semilogx(layer.data['reference']['strain(%)'],
                                 layer.data['reference']['damping(%)']+sigma_lnD,
                                '--',color='k',label='Target $\mu$+1$\sigma$') 
                axes[idx+1].semilogx(layer.data['reference']['strain(%)'],
                                 layer.data['reference']['damping(%)']-sigma_lnD,
                                    '--',color='k')
    #ax.legend()
    [ax.set_ylabel('$G/G_{max}$') for ax in axes[np.arange(0,len(plot_layers)*2,2)]]
    [ax.set_ylabel('Damping (%)') for ax in axes[np.arange(1,len(plot_layers)*2,2)]]
    [ax.set_xlabel('Shear Strain, $\gamma$ (%)') for ax in axes[-2:]]
    [ax.set_ylim(0,1.2) for ax in axes[np.arange(0,len(plot_layers)*2,2)]]
    [ax.set_ylim(0,30) for ax in axes[np.arange(1,len(plot_layers)*2,2)]]
    [ax.set_xlim(1e-4,1) for ax in axes]
    [ax.grid(alpha=0.2) for ax in axes]
    
    # Add legend to first damping plot
    axes[1].legend(loc='upper left')
    
    if save_fig:
        [fig.savefig('Fig_MRD_Randomizations_by_Layer'+ext,bbox_inches='tight',dpi=400) for ext in ['pdf','png']]
    plt.show()
    return


def plot_mrd_dist(layer_idx,dat_MRD_rand,plot='MR',target_strain=0.01,figsize=None):
    """
    Plots slice of the Darendeli distribution
    _________________________________________________________
    Inputs
    -------
    layer_idx : layer index to plot
    dat_MRD_rand : list (layer level) of dicts (realization level) containing dataframes (MRD data)
    plot: if 'MR', plots MR distribution, else plots damping distribution
    target_strain: desired strain level of slice
    figsize : figure size, tuple
    
    Returns
    -------
    """
    
    slice_mr = []
    slice_damp = []
    for realiz_idx in dat_MRD_rand[layer_idx].keys(): #Loop on realizations for given layer
        idx = np.argmin(np.abs(dat_MRD_rand[layer_idx][realiz_idx]['strain(%)'] - target_strain))
        slice_mr.append(dat_MRD_rand[layer_idx][realiz_idx]['G/Gmax'][idx])
        slice_damp.append(dat_MRD_rand[layer_idx][realiz_idx]['damping(%)'][idx])

    if plot == 'MR':
        data = slice_mr
    else:
        data = slice_damp

    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)
    #m = truncnorm.fit(data)

    
    if figsize:
        plt.figure(figsize=figsize)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    title = "$Z_i$ (Strain %.4f%%): $\mu_{ln}$ = %.2f,  $\sigma_{ln}$ = %.2f" % (target_strain, mu, std)
    plt.title(title)

    plt.gca().get_yaxis().set_visible(False)

    plt.show()
    return


def plot_lnVs_distribution(dat,plot_nlayer=None,ncol=4,figsize=None,show_legend=True,fs=10):
    """
    Plot the input versus model output ln(Vs) distributions for each layer
    dat : output data
    plot_nlayer : number of layers to plot
    """

    # Get the model input data
    ln_Vs_baseline = np.log(dat.Vs0) # Baseline median Vs (mean ln(Vs)) profile
    sigmalnVs = dat.sigma_ln_Vs      # lnVs std deviation data

    # Get the model output data; reorganize Vs-realiz profiles into dataframe, rows=layers,cols=realiz
    dat_ln_Vs = np.log(pd.DataFrame(data=dat.Vs.transpose()))

    # Setup plot indices
    Nlayer = dat_ln_Vs.shape[0]
    if plot_nlayer:
        plot_indices = range(0,Nlayer,int(np.ceil(Nlayer/plot_nlayer)))
    else:
        plot_indices = range(0,Nlayer)

    # Set up axes
    nrow = int(np.ceil(len(plot_indices)/ncol))
    if figsize:
        fig, ax = plt.subplots(nrow,ncol,figsize=figsize)
    else:
        fig, ax = plt.subplots(nrow,ncol)
    ax = ax.flatten()

    for idx, axi in zip(plot_indices,ax): # Loop on layers
        # Get data
        data = dat_ln_Vs.iloc[idx,:]

        # Fit a normal distribution to the data:
        mu, std = norm.fit(data)

        # Plot the histogram.
        axi.hist(data, bins=25, density=True, alpha=0.6, color='g')

        # Plot the PDFS
        xmin, xmax = axi.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        # ..model fit
        p = norm.pdf(x, mu, std)
        axi.plot(x, p, '-', color='k', linewidth=2,
                label='Model Fit: '+'$\mu_{ln}$=%.2f,  $\sigma_{ln}$=%.2f'%(mu, std))

        # ..model input
        p2 = norm.pdf(x, ln_Vs_baseline[idx], sigmalnVs[idx])
        axi.plot(x, p2, '--', color='tab:red', linewidth=2,
                label='Model Input: '+'$\mu_{ln}$=%.2f,  $\sigma_{ln}$=%.2f'%(ln_Vs_baseline[idx], sigmalnVs[idx]))
        
        if show_legend:
            title = 'Layer ' + str(idx)
            axi.legend(title=title,fontsize=fs,loc='upper left')
            ymin, ymax = axi.get_ylim()
            axi.set_ylim(bottom=ymin,top=ymax+0.55*(ymax-ymin))
        axi.get_yaxis().set_visible(False)
        
    #plt.tight_layout()
    plt.show()
    return


def plot_Z_dist(dat,idx=0,figsize=None):
    """
    Plot distribution of realizations of standard normal distribution for layer idx
    Should have a (mu_ln, sigma_ln) = (0,1)
    """
    dat_epsilon = pd.DataFrame(data=dat.Z.transpose())


    data = dat_epsilon.iloc[idx,:]

    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)
    
    if figsize:
        plt.figure(figsize=figsize)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    
    title = "$Z_i$ (Layer %i): $\mu_{ln}$ = %.2f,  $\sigma_{ln}$ = %.2f" % (idx, mu, std)
    plt.title(title)
    
    plt.gca().get_yaxis().set_visible(False)
    
    plt.show()
    return


def plot_model_rho(thicknesses,dat,notes):
    """
    Plots the model depth, thickness, and interlayer coefficient profiles
    """
    # These should be the same regardless of realization (when thickness is constant); Zi will be affected by rho_1L between realizations though
    dat_Z = pd.DataFrame(data=dat.Z.transpose()) 
    dat_epsilon = pd.DataFrame(data=dat.epsilon.transpose()) 


    fig, ax = plt.subplots(1,4,figsize=(12,6))
    ax[0].plot(dat.rho_t[1,:],np.cumsum(thicknesses),'s-',markeredgecolor='k',
               label=r'$\rho_t(t)$')
    ax[1].plot(dat.rho_z[1,:],np.cumsum(thicknesses),'^-',markeredgecolor='k',
               label=r'$\rho_z(z)$')
    ax[2].plot(dat.rho_1L[1,:],np.cumsum(thicknesses),'o-',markeredgecolor='k',
               label=r'$\rho_{1L}$')

    ax[3].plot(dat_Z,np.cumsum(thicknesses),'-',label=r'$\rho_{1L}$',color='lightgrey',alpha=0.2)
    ax[3].plot(dat_Z.mean(axis=1),np.cumsum(thicknesses),'-',label=r'$\rho_{1L}$',color='k')

    ax[0].set_ylabel('Depth (m)')
    ax[0].set_xlabel(r'$\rho_t(t)$',fontsize=12)
    ax[1].set_xlabel(r'$\rho_z(z)$',fontsize=12)
    ax[2].set_xlabel(r'$\rho_{1L}$',fontsize=12)
    ax[3].set_xlabel(r'$Z_{i}$',fontsize=12)

    #ax[0].legend(loc='lower left')
    [axi.invert_yaxis() for axi in ax]
    [axi.set_ylim(top=0) for axi in ax]
    [axi.set_xlim(left=0,right=1) for axi in ax[0:3]]
    [axi.grid() for axi in ax]
    fig.suptitle(notes,x=0.5,y=0.94)
    plt.show()
    return


def plot_Su_random(dat,plot_realiz_i=[],n_sigma=1,ylim=(0,100),figsize=(6,8)):
    """
    Plot ranomization profile
    plot_realiz_i : list containing indices of realizations to plot | list
    n_sigma : number of sigma bounds to plot
    -------
    None.

    """
    n_realization = dat.Su.shape[0]
    
    # Plot results
    fig, ax = plt.subplots(figsize=figsize)
    plt.xlabel('Undrained Shear Strength, $S_U$ (kPa)')
    plt.ylabel('Depth (m)')
    
    # Mean
    plt.plot(np.repeat(dat.Su0,2)/1e3,
             np.dstack((dat.z_top0,dat.z_bot0)).flatten(),
             label='Baseline',
             zorder=n_realization+1)
    
    for k in range(0,n_realization):
        plt.plot(np.repeat(dat.Su[k,:]/1e3,2),
                 np.dstack((dat.z_top[k,:],dat.z_bot[k,:])).flatten(),
                 color=(0.85,0.85,0.85),
                 label=( 'Realization' if k == 0 else None))
        
    # +/- sigma bounds
    sigma_su = pd.DataFrame(data=dat.Su.transpose()).std(axis=1)
    
    plt.plot(np.repeat(dat.Su0 + n_sigma*sigma_su,2)/1e3,
             np.dstack((dat.z_top0,dat.z_bot0)).flatten(),
             '--', color='salmon',
             label='$\mu_{S_U}$ $\pm$ $\sigma_{S_U,realiz}$',
             zorder=n_realization+1)
    plt.plot(np.repeat(dat.Su0 - n_sigma*sigma_su,2)/1e3,
             np.dstack((dat.z_top0,dat.z_bot0)).flatten(),
             '--', color='salmon',
             label=None,
             zorder=n_realization+1)
        
    if plot_realiz_i:
        for k in plot_realiz_i:
            plt.plot(np.repeat(dat.Su[plot_realiz_i,:],2)/1e3,
                     np.dstack((dat.z_top[plot_realiz_i,:],dat.z_bot[plot_realiz_i,:])).flatten(),
                     color='tab:red',lw=1.5,
                     label='')
    
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_ylim(bottom=ylim[1], top=ylim[0])
    plt.legend()
    plt.show()
    
    return None


def plot_Su_distribution(dat,plot_nlayer=None,ncol=4,figsize=None,show_legend=True,fs=10):
    """
    Plot the input versus model output Su distributions for each layer
    dat : output data
    plot_nlayer : number of layers to plot
    """

    # Get the model input data
    mean_Su  = np.array(dat.Su0)/1e3        # Baseline mean Su profile [kPa]
    sigma_Su = np.array(dat.sigma_Su)/1e3   # Su std deviation [kPa]

    # Get the model output data; reorganize Su-realiz profiles into dataframe, rows=layers,cols=realiz
    dat_Su = pd.DataFrame(data=dat.Su.transpose())

    # Setup plot indices
    Nlayer = dat_Su.shape[0]
    if plot_nlayer:
        plot_indices = range(0,Nlayer,int(Nlayer/plot_nlayer))
    else:
        plot_indices = range(0,Nlayer)

    # Set up axes
    nrow = int(np.ceil(len(plot_indices)/ncol))
    if figsize:
        fig, ax = plt.subplots(nrow,ncol,figsize=figsize)
    else:
        fig, ax = plt.subplots(nrow,ncol)
    ax = ax.flatten()

    for idx, axi in zip(plot_indices,ax): # Loop on layers
        # Get data
        data = dat_Su.iloc[idx,:]/1e3 #[kPa]

        # Fit a normal distribution to the data:
        mu, std = norm.fit(data)

        # Plot the histogram.
        axi.hist(data, bins=25, density=True, alpha=0.6, color='g')

        # Plot the PDFS
        xmin, xmax = axi.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        # ..model fit
        p = norm.pdf(x, mu, std)
        axi.plot(x, p, '-', color='k', linewidth=2,
                label='Model Fit: '+'$\mu_{ln}$=%.2f,  $\sigma_{ln}$=%.2f'%(mu, std))

        # ..model input
        p2 = norm.pdf(x, mean_Su[idx], sigma_Su[idx])
        axi.plot(x, p2, '--', color='tab:red', linewidth=2,
                label='Model Input: '+'$\mu_{ln}$=%.2f,  $\sigma_{ln}$=%.2f'%(mean_Su[idx], sigma_Su[idx]))
        
        if show_legend:
            title = 'Layer ' + str(idx)
            axi.legend(title=title,fontsize=fs,loc='upper left')
            ymin, ymax = axi.get_ylim()
            axi.set_ylim(bottom=ymin,top=ymax+0.55*(ymax-ymin))
        axi.get_yaxis().set_visible(False)
        
    #plt.tight_layout()
    plt.show()
    return


def plot_joint_lnVs_shear_dist(dat,idx=1):
    """
    Plots the joint distribution between lnVs and shear strength.
    """
    
    # Re-organize Su and lnVs realization data into dataframe (rows=layers, columsn=realiz)
    dat_Su    = pd.DataFrame(data=dat.Su.transpose())
    dat_ln_Vs = pd.DataFrame(data=np.log(dat.Vs.transpose()))

    # Get the input mean and std dev.
    rho = dat.corr[idx]
    mu_yy_inp, std_yy_inp = dat.Su0[idx]/1e3, dat.sigma_Su[idx]/1e3 #[kPa]
    mu_xx_inp, std_xx_inp = np.log(dat.Vs0[idx]), dat.sigma_ln_Vs[idx]

    # Ge the x and y data for computing mean and std. dev.
    yy = dat_Su.iloc[idx,:]/1e3 #[kPa]
    xx = dat_ln_Vs.iloc[idx,:]

    # Assemble covariance matrix
    corr = rho
    means = [mu_xx_inp, mu_yy_inp]  
    stds = [std_xx_inp, std_yy_inp]
    covs = [[stds[0]**2          , stds[0]*stds[1]*corr], 
            [stds[0]*stds[1]*corr,           stds[1]**2]] 

    #print(means)
    #print(covs)
    
    # Get data limits
    xmin = xx.min()
    xmax = xx.max()
    ymin = yy.min()
    ymax = yy.max()


    # Definitions for the axes
    # ----------------------------------------------------------
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    hist_height = 0.15
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, hist_height]
    rect_histy = [left + width + spacing, bottom, hist_height, height]

    # Start with a rectangular Figure
    plt.figure(figsize=(6, 6))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # Plot joint-distribution of x-var and y-var
    # ----------------------------------------------------------
    # ..Scatter plot
    ax_scatter.scatter(xx, yy)

    # ..Contour plot
    n = 500
    X,Y = np.meshgrid(np.linspace(xmin,xmax,n),
                      np.linspace(ymin,ymax,n))
    pos = np.array([X.flatten(),Y.flatten()]).T
    rv = multivariate_normal(means, covs, allow_singular=True)
    ax_scatter.contour(X,Y,rv.pdf(pos).reshape(n,n),linewidths=2.5)


    # Plot y-var distribution
    # ----------------------------------------------------------
    # ..Plot the histogram
    ax_histy.hist(yy, bins=25, density=True, alpha=0.8,orientation='horizontal')

    # ..Plot model output PDF
    x = np.linspace(ax_histy.get_ylim()[0], ax_histy.get_ylim()[1], 100)
    mu, std = norm.fit(yy)
    p = norm.pdf(x, mu, std)
    ax_histy.plot(p,x, 'k', linewidth=1.5)

    # ..Plot model input PDF
    p2 = norm.pdf(x, mu_yy_inp, std_yy_inp)
    ax_histy.plot(p2,x, '--', color='tab:red', linewidth=1.5)


    # Plot the x-var distribution
    # ----------------------------------------------------------
    # ..Plot the histogram
    ax_histx.hist(xx, bins=25, density=True, alpha=0.8)

    # ..Plot the model output PDF
    x = np.linspace(ax_histx.get_xlim()[0], ax_histx.get_xlim()[1], 100)
    mu, std = norm.fit(xx)
    p = norm.pdf(x, mu, std)
    ax_histx.plot(x,p, 'k', linewidth=1.5)

    # ..Plot the model input PDF
    p2 = norm.pdf(x, mu_xx_inp, std_xx_inp)
    ax_histx.plot(x,p2, '--', color='tab:red', linewidth=1.5)


    # ----------------------------------------------------------
    # Limits
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histx.axes.get_xaxis().set_ticks([])
    ax_histx.axes.get_yaxis().set_ticks([])
    ax_histy.axes.get_xaxis().set_ticks([])
    ax_histy.axes.get_yaxis().set_ticks([])

    # Labels
    fs = 11.5
    ax_scatter.set_xlabel('$ln(V_S)$',fontsize=fs)
    ax_scatter.set_ylabel('$S_U$ (kPa)',fontsize=fs)

    # Grid
    ax_scatter.grid(alpha=0.2)

    # Text 
    text = 'Layer ' + str(idx) + r',  $\rho$='+str(corr)
    text = text + '\n' + r'($\mu$, $\sigma$$)_{S_U}$=' + str(np.round(mu_yy_inp,2))+', '+str(np.round(std_yy_inp,2))
    text = text + '\n' + r'($\mu$, $\sigma$$)_{lnV_S}$=' + str(np.round(mu_xx_inp,2))+', '+str(np.round(std_xx_inp,2))
    ax_scatter.text(0.02, 0.88, text, horizontalalignment='left',
                  verticalalignment='center', transform=ax_scatter.transAxes,fontsize=10)
    plt.show()
    return



def savedat(dat,fname='data.pkl'):
    """Saves dat realiz output as a pickle file"""
    with open(fname,'wb') as f:
        temp = dat.__dict__
        keys = [key for key in temp.keys() if '__' not in key]
        out_dict = {}
        for key in keys:
            out_dict[key] = temp[key]
        pickle.dump(out_dict,f,protocol=pickle.HIGHEST_PROTOCOL)
    return 


def loaddat(fname='data.pkl'):
    with open('data.pkl','rb') as f:
        readin = pickle.load(f)
    return readin