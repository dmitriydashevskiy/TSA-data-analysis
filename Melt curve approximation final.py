#!/usr/bin/env python
# coding: utf-8

# In[153]:


# import necessary lybraries and functions
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import numpy as np
from matplotlib import pyplot
import pandas as pd
import os
import seaborn as sns
from IPython.display import display
sns.set()


# In[ ]:


"""
For the program to work, the data must be pre-organized as follows.
A directory with .xls files with arbitrary names must be created. Each file contains one sheet.
The first column represents cells in which the values of temperatures in degrees Celsius,
at which measurements were made, are entered (the enumeration starts from the second cell,
the first cell may contain an arbitrary symbol). 
The first row (except for the first cell) represents the cells in which the ligand concentrations
at which the measurements were made are written.
The ligand concentration should be written as follows:
numerical value of concentration - space - order designator
( M - moles, mM - millimoles, uM - micromoles , nM - nanomoles ).
In the square bounded by the first row and the first column, the fluorescence values of the sample
at a given temperature and ligand concentration are written in each cell.
As an example of data organization for processing, you can use the folder 'Example for processing'.
"""


# In[154]:


"""
The function 'data_search' is made to finde directory and memorize the names of files to be processed.

For the work of the function it is necessary to specify the full path to the folder where the
files are located using the variable 'directory_name'.
The folder should contain only files that contain data to be processed in the form specified above.
If there is a need to store in the folder other files that do not contain data for processing,
you can pass them to the function as a list of the names of these files in a variable 'extra_files'.

The result of the function work is a list of the names of the files being processed,
as well as printing these names.
"""
def data_search(directory_name, extra_files = ['.DS_Store'] ):
    data_name = []
    for file_name in os.listdir(directory_name):
        if file_name not in extra_files: 
            data_name.append(file_name)
    data_name = sorted(data_name)
    print('Files to be processed: ',data_name)
    return data_name
    


# In[155]:


"""
The function 'data_search' is made to calculate the fluorescence value of the sample
at given curve parameters and fixed temperature.

For the work of the function, the next values must be passed into it :
current temperature in degrees Celsius - T,
enthalpy change – H,
melting point in degrees Celsius – Tm, 
fluorescence of the sample in the native state – Fn, 
fluorescence of the sample in the unfolded state – Fu, 
slope values at the beginning and at the end of the curve – a,b.

The result of the function work is a the fluorescence value of the sample.
"""
def meltcurve_computation(T,H,Tm,Fn,Fu,a,b):
    T_ref = 298 
    T = T + 273
    Tm = Tm + 273
    G = H - T * (H/Tm)
    R = 8.31
    return (Fn+(T-T_ref)*a + (Fu+(T-T_ref)*b)*np.exp(-G/(R*T)))/(1 + np.exp(-G/(R*T)))


# In[156]:


"""
The function 'data_read' is made to get and save data from the files.

For the work of the function it is necessary to pass the list of file names in 'data_name' 
and pass to the directory in 'directory_name' into function.

The result of the function work is a dictionary 'experiment_dict' 
in which under the name of each files is a dictionary 'curve_dict' that includes:
concentration values in an easy-to-read form  – 'concentration_names' ,
numerical values of concentrations in moles – 'concentrations' , 
list of temperatures – 'x_data',
list of experimental fluorescence values – 'y_data'. 
"""

def data_read(data_name, directory_name):
    experiment_dict = dict()
    for k in range(len(data_name)):
        name = data_name[k]
        excel_data = pd.read_excel(directory_name +'/'+data_name[k])
        data = pd.DataFrame(excel_data)
        x_data = []
        y_data = []
        concentrations = []
        concentration_names = data.columns[1:]
        
        for i in concentration_names:
            c = i.split()
            if c[1] == 'uM':
                l = float(c[0])*10**(-6)
            elif c[1] == 'nM':
                l = float(c[0])*10**(-9)
            elif c[1] == 'mM':
                l = float(c[0])*10**(-3)
            elif c[1] == 'M':
                l = float(c[0])
            else:
                print('error: name ', i, ' is invalid')
       
            concentrations.append(l)
            x = np.array(data[data.columns[0]])
            y = np.array(data[i])
            x_data.append(x)
            y_data.append(y)
            
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        concentrations = np.array(concentrations)
        curve_dict = {'x_data': x_data, 'y_data': y_data, 'concentrations' : concentrations,
                      'concentration_names': concentration_names }
        experiment_dict[name]= curve_dict
    return experiment_dict
    


# In[157]:


"""
The function 'curve_approximation' is made to find parameters that best approximate
the selected function to the experimental data.

For the work of the function, the next values must be passed into it :
the dictionary generated by 'data_read' function' - 'experiment_dict',
a function used for approximation - 'approximation_function',
the list of file names - 'data_name'.

Additionally, if you want to set starting values and boundaries for parameter search manually,
you can do it by giving the function the lists 'params_start', 'params_min' and 'params_max'.

You can also give the function the values of temperatures for which you want to calculate
the dissociation constant using the isothermal method, as a list in the variable 'T_isotherm'.

The result of the function work is a modification of dictionary 'experiment_dict'. 
The next values are added to each 'curve_dict':
dictionary with the values of chosen parameters 
and also with the value change of Gibbs free energy change 'G' and change of entropy 'S' – 'parameters',
temperatures chosen for isothermal analysis: 'T_isotherm',
matrix containing fration of unfolded protein for temperatures from 'T_isotherm'.

"""
def curve_approximation(experiment_dict, approximation_function, data_name,
                        params_start = None, params_min = None, params_max = None,
                        T_isotherm = [40,42,44,46,48,50,52,54,56,58,60]):
    H_0 = 10000
    Tm_0 = 45
    a_0 = 0
    b_0 = 0
    
    if params_start == None:
        flag_1 = True
    if params_min == None:
        flag_2 = True
    if params_max == None:
        flag_3 = True        
    
    for k in range(len(data_name)):
        name = data_name[k]
        concentrations = experiment_dict[name]['concentrations']
        x_data = experiment_dict[name]['x_data']
        y_data = experiment_dict[name]['y_data']
        n_u = np.zeros((len(T_isotherm), len(concentrations)))
            
        parameters_names = ['H','Tm','Fn','Fu','a','b', 'G','S']
        parameters = dict()
        for m in parameters_names:
            parameters[m]=[]
        
            
        for i in range(len(x_data)):
            Fn_0 = min(y_data[i])
            Fu_0 = max(y_data[i])

            H_min,H_max = [1000,800000]
            Tm_min,Tm_max = [30,75]
            Fn_min,Fn_max = [0.8*Fn_0,1.1*Fn_0]
            Fu_min,Fu_max = [0.9*Fu_0,1.5*Fu_0] 
            a_min,a_max = [-1,1]
            b_min,b_max = [-0.5,0.5]
            
            if flag_1:
                params_start = [H_0, Tm_0, Fn_0, Fu_0, a_0, b_0]
            if flag_2:
                params_min = [H_min, Tm_min, Fn_min, Fu_min, a_min, b_min]
            if flag_3:
                params_max = [H_max, Tm_max, Fn_max, Fu_max, a_max, b_max]

            param, addition = curve_fit(approximation_function, x_data[i], y_data[i], 
                                        p0 = params_start, bounds = (params_min, params_max))
                
            H, Tm, Fn, Fu, a, b = list(map(float,param))
            S = H/(Tm+273)
            G = H - 298*S
                
            param_list = [H, Tm, Fn, Fu, a, b, G, S]
                
            for n in range(len(param_list)):
                parameters[parameters_names[n]].append(param_list[n])
                
            for j in range(len(T_isotherm)):
                n_u[j][i] = (1/(1+np.exp((H/8.31)*(1/(T_isotherm[j]+273) - 1/(Tm+273)))))
                
        experiment_dict[name]['parameters'] = parameters
        experiment_dict[name]['T_isotherm'] = T_isotherm
        experiment_dict[name]['n_u'] = n_u
    

    return experiment_dict  


# In[158]:


"""
The function 'visualization' is made to visualize the result of function 'curve_approximation' work.

For the work of the function, the next values must be passed into it:
the dictionary generated by 'data_read' function' 
and modified by 'curve_approximation' function – 'experiment_dict',
the list of file names - 'data_name'.

The result of the function work is a displaying a series of tables and graphes for each file:
the name of the processed file,
table of chosen parameters,
graph showing the experimental points of melting curve and their approximation,
graph showing Gibbs free energy change dependence on ligand concentration,
graph showing melting point 'Tm' dependence on ligand concentration,
graph showing the fraction of unfolded protein dependence on ligand concentration
for temperatures from 'T_isotherm' – 'Isotherm'.
"""

def visualization(experiment_dict, data_name):
    for k in range(len(data_name)):
        name = data_name[k]
        concentration_names = experiment_dict[name]['concentration_names']
        concentrations = experiment_dict[name]['concentrations']
        x_data = experiment_dict[name]['x_data']
        y_data = experiment_dict[name]['y_data']
        parameters = experiment_dict[name]['parameters']
        T_isotherm = experiment_dict[name]['T_isotherm']
        n_u = experiment_dict[name]['n_u']
        print('The file being processed:',name)
        
        parameters_frame = pd.DataFrame.from_dict(parameters, orient='index', columns=concentration_names)
        display(parameters_frame)
        
        plt.figure(figsize =(15,10))
        for i in range(len(x_data)):
            plt.scatter(x_data[i], y_data[i], color = 'C'+str(i))
            f = meltcurve_computation(x_data[i],*np.array(parameters_frame[concentration_names[i]][:6]))
            plt.plot(x_data[i],f,label= concentration_names[i],color = 'C'+str(i))
            plt.xlabel('T,°C')
            plt.ylabel('Flourescence CPM')
        
        
        plt.legend(loc = 2, prop={'size': 10})
        plt.title('Approximation of melting curves')
        plt.show()
        
        plt.figure(figsize =(15,10))
        plt.scatter(concentrations, *np.array(parameters_frame.loc[['G']]))
        plt.ylabel('delta G')
        plt.xlabel( 'Ligand concentration, nM')
        pyplot.xscale('log')
        plt.title('delta G dependence')
        plt.show()
        
        plt.figure(figsize =(15,10))
        plt.scatter(concentrations, *np.array(parameters_frame.loc[['Tm']]))
        plt.ylabel('Tm')
        plt.xlabel( 'Ligand concentration, nM')
        pyplot.xscale('log')
        plt.title('Tm dependence')
        plt.show()
        
        plt.figure(figsize =(15,10))  
        for i in range(len(n_u)):
            plt.plot(concentrations, n_u[i], label = ' %2.0f C°'% T_isotherm[i])
        plt.ylabel('Fraction of unfolded protein')
        plt.xlabel( 'Ligand concentration, nM')
        pyplot.xscale('log')
        plt.legend(loc = 2, prop={'size': 10})
        plt.title('Isotherm')
        plt.show()


# In[159]:


"""
The function 'data_save' is made to save data gained by approximation.

For the work of the function, the next values must be passed into it:
the dictionary generated by 'data_read' function' 
and modified by 'curve_approximation' function – 'experiment_dict',
the list of file names - 'data_name',
the pass to the directory – 'directory_name'.

The result of the function work is a generation of next csv files in the directory 'directory_name':
the file with the table containing Gibbs free energy change dependence on ligand concentration
for each file – 'result_G.csv',
the file with the table containing melting point 'Tm' dependence on ligand concentration
for each file – 'result_G.csv',
the files containing matrix for the fraction of unfolded protein dependence on ligand concentration
for temperatures from 'T_melt' for each file from analysis – name of file + '_result_isothermal.csv'.

"""
def data_save(experiment_dict, data_name, directory_name):
    for k in range(len(data_name)):
        name = data_name[k]
        concentration_names = experiment_dict[name]['concentration_names']
        concentrations = experiment_dict[name]['concentrations']
        x_data = experiment_dict[name]['x_data']
        y_data = experiment_dict[name]['y_data']
        parameters = experiment_dict[name]['parameters']
        T_isotherm = experiment_dict[name]['T_isotherm']
        n_u = experiment_dict[name]['n_u']
        
        if k == 0:
            df = pd.DataFrame(parameters['G'], concentrations,[name])
            dfT = pd.DataFrame(parameters['Tm'], concentrations,[name])
        else:
            df2 = pd.DataFrame(parameters['G'], concentrations,[name])
            df = pd.concat([df,df2], axis = 1, sort=False )
            
            dfT2 = pd.DataFrame(parameters['Tm'], concentrations,[name])
            dfT = pd.concat([dfT,dfT2], axis = 1, sort=False )
        
        data_nu = pd.DataFrame(n_u.T, concentrations , T_isotherm)
        data_nu.to_csv(directory_name +'/' +data_name[k][:-4] + '_result_isothermal.csv')
            
    df.to_csv(directory_name+'/result_G.csv')
    dfT.to_csv(directory_name+'/result_T.csv')   
    
    


# In[161]:


"""
The following sequence of commands implements the full cycle of data processing, namely,
searching and saving files for analysis – 'data_search',
saving data from files – 'data_read',
approximating data from the file according to the selected model – 'curve_approximation',
visualizing the selected parameters and the values calculated from them – 'visualization',
saving the calculated values – 'data_save'.

In the variable 'directory_name' you should specify the full path to the directory
where the files to be processed are located.

To get the best result, we recommend to first try different initial values, as well as constraints
for the parameters in the function 'curve_approximation' and visually evaluate the result obtained with
the help of the function 'visualization', and only when the best approximation to the experimental data
is achieved, proceed to converting and saving the data with the help of the function 'data_save'.  

"""
directory_name = '/Users/dmitrij/Desktop/Лаба/CPM project/Статья/Example for processing'
data_name = data_search(directory_name)
experiment = data_read(data_name, directory_name)
experiment = curve_approximation(experiment, meltcurve_computation, data_name)
visualization(experiment,data_name)
data_save(experiment,data_name, directory_name )


# In[139]:


directoryname = '/Users/dmitrij/Desktop/Лаба/CPM project/Статья/Example'
dataname = []
for filename in os.listdir(directoryname):
    if filename != '.DS_Store' and filename != 'result.csv' and filename != 'result_G_from_T.csv' and filename != 'result_T.csv':
        dataname.append(filename)
    dataname = sorted(dataname)
print(dataname)
def meltcurve_T(x,H,Tm,Fn, a, Fu, b):
    Tref = 273 
    T = x + 273
    Tm = Tm + 273
    G = H - T * (H/Tm)
    R = 8.31
    
    
    return (Fn+(T-Tref)*a + (Fu+(T-Tref)*b)*np.exp(-G/(R*T)))/(1 + np.exp(-G/(R*T)))

def obrabotka_isotermal(dataname):
    for k in range(len(dataname)):
        print(dataname[k])
        excel_data = pd.read_excel(directoryname +'/'+dataname[k])
        data = pd.DataFrame(excel_data)
        xdata = []
        ydata = []
        conc = []
        
        for i in data.columns[1:]:
            c = i.split()
            if c[1] == 'uM':
                l = float(c[0])*10**(-6)
            elif c[1] == 'nM':
                l = float(c[0])*10**(-9)
            elif c[1] == 'mM':
                l = float(c[0])*10**(-3)
            else:
                print('ошибка, название', i)
       
            conc.append(l)
    
            x = np.array(data['X'])
            y = np.array(data[i])
            xdata.append(x)
            ydata.append(y)
            
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        conc = np.array(conc)
        poptall = []
        
        dG1 = []
        dG2 = []
        Tmelt1 = []
        Tmelt2 = []
        N = []
        Temp_interesting = [40,42,44,46,48,50,52,54,56,58,60]
        Frac_all = np.zeros((len(Temp_interesting), len(conc)))
        

        H0 = 10000
        Tm10 = 45
        
        a0 = 0
        b0 = 0
       
        for i in range(len(xdata)):
            H1min,H1max = [1000,800000]
            Tm1min,Tm1max = [30,75]
           
          
            Fnmin,Fnmax = [0.8*min(ydata[i]),1.1*min(ydata[i])]
            amin,amax = [-1,1]
            Fumin,Fumax = [1*max(ydata[i]),1.5*max(ydata[i])] 
            bmin,bmax = [-0.5,0.5]
            
            method1 = 'trf'
            method2 = 'dogbox'

            popt,pcov = curve_fit(meltcurve_T, xdata[i], ydata[i], p0 = [H0,Tm10,min(ydata[i]),a0,max(ydata[i]),b0], bounds=([H1min, Tm1min, Fnmin, amin, Fumin, bmin],[H1max, Tm1max, Fnmax, amax, Fumax, bmax]))
            p = list(map(float,popt))
            poptall.append(p)
            dG1.append(p[0]-298*p[0]/(p[1]+273))
            Tmelt1.append(p[1])
            for j in range(len(Temp_interesting)):
                Frac_all[j][i] = (1/(1+np.exp((p[0]/8.31)*(1/(Temp_interesting[j]+273) - 1/(p[1]+273)))))
            

            perr = np.sqrt(np.diag(pcov))
            
        
        
        Tmelt1 = np.array(Tmelt1)
        dG1 = np.array(dG1)
        Frac_all = np.array(Frac_all)
        plt.figure(figsize =(15,10))
        

        for i in range(len(xdata)):
            plt.scatter(xdata[i], ydata[i], color = 'C'+str(i))
            f = meltcurve_T(xdata[i],*poptall[i])
            plt.plot(xdata[i],f,label='H1=%1.2e, Tm1=%3.0f , Fn=%5.1f, a=%1.5f, Fu=%5.1f, b=%5.3f'% tuple(poptall[i]),color = 'C'+str(i))
            plt.xlabel('T,°C')
            plt.ylabel('Flourescence CPM')

        plt.legend(loc = 2, prop={'size': 10})
        plt.show()
        
        plt.figure(figsize =(15,10))
        plt.scatter(conc, dG1, label = 'G1')
        plt.ylabel('G')
        pyplot.xscale('log')
        plt.legend(loc = 2, prop={'size': 10})
        plt.show()
        
        plt.figure(figsize =(15,10))
        plt.scatter(conc, Tmelt1, label = 'Tmelt1')
        plt.ylabel('Tm')
        pyplot.xscale('log')
        plt.legend(loc = 2, prop={'size': 10})
        plt.show()
        
        
        
        plt.figure(figsize =(15,10))  
        for i in range(len(Frac_all)):
            plt.plot(conc, Frac_all[i], label = 'Temperature %2.0f '% Temp_interesting[i])
        plt.ylabel('Fraction')
        pyplot.xscale('log')
        plt.legend(loc = 2, prop={'size': 10})
        plt.show()
        
        Tmelt1 = np.reshape(Tmelt1,(len(Tmelt1),1))
        dG1 = np.reshape(dG1,(len(dG1),1))
        index1 = conc
        column = [dataname[k]]

        if k == 0:
            df = pd.DataFrame(dG1, index1,column)
            dfT = pd.DataFrame(Tmelt1, index1,column)
        else:
            df2 = pd.DataFrame(dG1, index1,column) 
            df = pd.concat([df,df2], axis = 1, sort=False )
            
            dfT2 = pd.DataFrame(Tmelt1, index1,column) 
            dfT = pd.concat([dfT,dfT2], axis = 1, sort=False )
        
        Frac_data = pd.DataFrame(Frac_all.T,conc , Temp_interesting)
        Frac_data.to_csv(directoryname +'/' +dataname[k][:-4] + '_result_isothermal.csv')
    #df.to_csv(directoryname+'/result_G_from_T.csv')
    #dfT.to_csv(directoryname+'/result_T.csv')
    
    


# In[140]:


obrabotka_isotermal(dataname)


# In[ ]:




