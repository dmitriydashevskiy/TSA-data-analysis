import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from scipy.optimize import curve_fit
from IPython.display import display

sns.set()


"""
For the program to work, the data must be pre-organized as follows.
A directory with .xls files with arbitrary names must be created.
Each file contains one sheet. The first column represents cells in
which the values of temperatures in degrees Celsius, at which
measurements were made, are entered (the enumeration starts from the
second cell, the first cell may contain an arbitrary symbol).
The first row (except for the first cell) represents the cells in which
the ligand concentrations at which the measurements were made are written.
The ligand concentration should be written as follows:
numerical value of concentration - space - order designator
(M - moles, mM - millimoles, uM - micromoles, nM - nanomoles).
In the square bounded by the first row and the first column, the
fluorescence values of the sample at a given temperature and ligand
concentration are written in each cell. As an example of data organization
for processing, you can use the folder 'Example for processing'.
"""

class DataReader:
    """Handles reading and processing of data files"""
    
    @staticmethod
    def search_data(directory_name, extra_files=['.DS_Store']):
        """Find and return list of files to be processed"""
        data_name = []
        for file_name in os.listdir(directory_name):
            if file_name not in extra_files:
                data_name.append(file_name)
        data_name = sorted(data_name)
        print("Files to be processed: ", data_name)
        return data_name

    @staticmethod
    def read_data(data_name, directory_name):
        """Read and process data from files"""
        experiment_dict = dict()
        for k in range(len(data_name)):
            name = data_name[k]
            excel_data = pd.read_excel(directory_name + "/" + data_name[k])
            data = pd.DataFrame(excel_data)
            x_data = []
            y_data = []
            concentrations = []
            concentration_names = data.columns[1:]

            for i in concentration_names:
                c = i.split()
                if c[1] == "uM":
                    conc = float(c[0]) * 10 ** (-6)
                elif c[1] == "nM":
                    conc = float(c[0]) * 10 ** (-9)
                elif c[1] == "mM":
                    conc = float(c[0]) * 10 ** (-3)
                elif c[1] == "M":
                    conc = float(c[0])
                else:
                    print("error: name ", i, " is invalid")

                concentrations.append(conc)
                x = np.array(data[data.columns[0]])
                y = np.array(data[i])
                x_data.append(x)
                y_data.append(y)

            x_data = np.array(x_data)
            y_data = np.array(y_data)
            concentrations = np.array(concentrations)
            curve_dict = {
                "x_data": x_data,
                "y_data": y_data,
                "concentrations": concentrations,
                "concentration_names": concentration_names,
            }
            experiment_dict[name] = curve_dict
        return experiment_dict

class CurveApproximator:
    """Handles curve approximation and parameter fitting"""
    
    @staticmethod
    def meltcurve_computation(T, H, Tm, Fn, Fu, a, b):
        """Compute melt curve values"""
        T_ref = 298
        T = T + 273
        Tm = Tm + 273
        G = H - T * (H / Tm)
        R = 8.31
        return (Fn + (T - T_ref) * a + (Fu + (T - T_ref) * b) *
                np.exp(-G / (R * T))) /\
               (1 + np.exp(-G / (R * T)))

    @staticmethod
    def approximate_curves(experiment_dict, data_name, T_isotherm, params_start=None, params_min=None, params_max=None):
        """Approximate curves and calculate parameters"""
        H_0 = 10000
        Tm_0 = 45
        a_0 = 0
        b_0 = 0

        flag_1 = params_start is None
        flag_2 = params_min is None
        flag_3 = params_max is None

        for k in range(len(data_name)):
            name = data_name[k]
            concentrations = experiment_dict[name]["concentrations"]
            x_data = experiment_dict[name]["x_data"]
            y_data = experiment_dict[name]["y_data"]
            
            # Initialize n_u with correct dimensions based on actual data
            n_u = np.zeros((len(x_data[0]), len(concentrations)))
            print('File in process: ', name)

            parameters_names = ["H", "Tm", "Fn", "Fu", "a", "b", "G", "S"]
            parameters = dict()
            for m in parameters_names:
                parameters[m] = []

            R_square_list = []

            for i in range(len(x_data)):
                F = y_data[i]
                Fn_0 = min(F)
                Fu_0 = max(F)

                H_min, H_max = [1000, 800000]
                Tm_min, Tm_max = [30, 75]
                Fn_min, Fn_max = [0.8 * Fn_0, 1.1 * Fn_0]
                Fu_min, Fu_max = [0.9 * Fu_0, 1.5 * Fu_0]
                a_min, a_max = [-1, 1]
                b_min, b_max = [-0.5, 0.5]

                if flag_1:
                    params_start = [H_0, Tm_0, Fn_0, Fu_0, a_0, b_0]
                if flag_2:
                    params_min = [H_min, Tm_min, Fn_min, Fu_min, a_min, b_min]
                if flag_3:
                    params_max = [H_max, Tm_max, Fn_max, Fu_max, a_max, b_max]

                param, addition = curve_fit(
                    CurveApproximator.meltcurve_computation,
                    x_data[i],
                    y_data[i],
                    p0=params_start,
                    bounds=(params_min, params_max),
                )

                H, Tm, Fn, Fu, a, b = list(map(float, param))
                S = H / (Tm + 273)
                G = H - 298 * S

                param_list = [H, Tm, Fn, Fu, a, b, G, S]
                prediction = CurveApproximator.meltcurve_computation(x_data[i], H, Tm, Fn, Fu, a, b)
                corr_matrix = np.corrcoef(y_data[i], prediction)
                R_square_list.append(corr_matrix[0,1]**2)

                for n in range(len(param_list)):
                    parameters[parameters_names[n]].append(param_list[n])

                for j in range(len(x_data[i])):
                    T = x_data[i][j]
                    n_u[j][i] = (F[j]-Fn-a*(T-25))/(Fu-Fn+(b-a)*(T-25))

            experiment_dict[name]["parameters"] = parameters
            experiment_dict[name]["n_u"] = n_u
            experiment_dict[name]['R^2'] = R_square_list

        return experiment_dict

class Visualizer:
    """Handles visualization of results"""
    
    @staticmethod
    def visualize_results(experiment_dict, data_name):
        """Visualize analysis results"""
        for k in range(len(data_name)):
            name = data_name[k]
            concentration_names = experiment_dict[name]["concentration_names"]
            concentrations = experiment_dict[name]["concentrations"]
            x_data = experiment_dict[name]["x_data"]
            y_data = experiment_dict[name]["y_data"]
            parameters = experiment_dict[name]["parameters"]
            n_u = experiment_dict[name]["n_u"]
            R_square = experiment_dict[name]['R^2']
            print("File in visualization:", name)

            parameters_frame = pd.DataFrame.from_dict(
                parameters, orient="index", columns=concentration_names
            )
            parameters_frame.loc[len(parameters_frame.index)] = R_square
            parameters_frame.rename(index={parameters_frame.index[-1]: 'R^2'}, inplace=True)    
            display(parameters_frame)

            # Generate colors for each concentration
            colors = [plt.cm.nipy_spectral(i/float(len(x_data)-1)) for i in range(len(x_data))]

            plt.figure(figsize=(15, 10))
            for i in range(len(x_data)):
                # Use the same color for both scatter points and line
                color = colors[i]
                plt.scatter(x_data[i], y_data[i], color=color, label=concentration_names[i])
                f = CurveApproximator.meltcurve_computation(
                    x_data[i],
                    *np.array(parameters_frame[concentration_names[i]][:6])
                )
                plt.plot(
                    x_data[i],
                    f,
                    color=color,
                    linestyle='-'
                )
                plt.xlabel("T,°C")
                plt.ylabel("Flourescence CPM")

            plt.legend(loc=2, prop={"size": 10})
            plt.title("Approximation of melting curves")
            plt.show()

            plt.figure(figsize=(15, 10))
            plt.scatter(concentrations, *np.array(parameters_frame.loc[["G"]]))
            plt.ylabel("delta G")
            plt.xlabel("Ligand concentration, nM")
            pyplot.xscale("log")
            plt.title("delta G dependence")
            plt.show()

            plt.figure(figsize=(15, 10))
            plt.scatter(concentrations, *np.array(parameters_frame.loc[["Tm"]]))
            plt.ylabel("Tm")
            plt.xlabel("Ligand concentration, nM")
            pyplot.xscale("log")
            plt.title("Tm dependence")
            plt.show()

            # Generate colors for isotherm plot based on the number of temperature points
            isotherm_colors = [plt.cm.nipy_spectral(i/float(len(n_u)-1)) for i in range(len(n_u))]

            plt.figure(figsize=(15,10))
            for i in range(len(n_u)):
                plt.plot(concentrations, n_u[i], label=' %2.0f C°'% x_data[0][i], color=isotherm_colors[i])
            plt.ylabel('Fraction of unfolded protein')
            plt.xlabel('Ligand concentration, nM')
            pyplot.xscale('log')
            plt.legend(loc=2, prop={'size': 10})
            plt.title('Isotherm')
            plt.show()

class DataSaver:
    """Handles saving of analysis results"""
    
    @staticmethod
    def save_data(experiment_dict, data_name, directory_name, T_isotherm):
        """Save analysis results to files"""
        for k in range(len(data_name)):
            name = data_name[k]
            concentration_names = experiment_dict[name]['concentration_names']
            concentrations = experiment_dict[name]['concentrations']
            x_data = experiment_dict[name]['x_data']
            y_data = experiment_dict[name]['y_data']
            parameters = experiment_dict[name]['parameters']
            n_u = experiment_dict[name]['n_u']
            R_square = experiment_dict[name]['R^2']
            
            if k == 0:
                df = pd.DataFrame(parameters["G"], index=concentrations, columns=[name])
                dfT = pd.DataFrame(parameters["Tm"], index=concentrations, columns=[name])
                dfR = pd.DataFrame(experiment_dict[name]['R^2'], index=concentrations, columns=[name])
            else:
                df2 = pd.DataFrame(parameters["G"], index=concentrations, columns=[name])
                df = pd.concat([df, df2], axis=1, sort=False)

                dfT2 = pd.DataFrame(parameters["Tm"], index=concentrations, columns=[name])
                dfT = pd.concat([dfT, dfT2], axis=1, sort=False)

                dfR2 = pd.DataFrame(experiment_dict[name]['R^2'], index=concentrations, columns=[name])
                dfR = pd.concat([dfR, dfR2], axis=1, sort=False)

            # Create DataFrame for n_u with proper dimensions
            data_nu = pd.DataFrame(n_u.T, index=concentrations, columns=x_data[0])
            data_nu.to_csv(
                directory_name + "/" + data_name[k][:-4] + "_result_isothermal.csv"
            )

        df.to_csv(directory_name + "/result_G.csv")
        dfT.to_csv(directory_name + "/result_T.csv")
        dfR.to_csv(directory_name+'/result_R^2.csv')
        print("Data is saved")

class MeltCurveAnalyzer:
    """Main class for melt curve analysis"""
    
    def __init__(self, directory_name, T_isotherm=None):
        self.directory_name = directory_name
        self.data_name = None
        self.experiment = None
        self.T_isotherm = T_isotherm if T_isotherm is not None else np.arange(25, 95, 5)  # Default temperature range

    def run_analysis(self):
        """Run the complete analysis workflow"""
        self.data_name = DataReader.search_data(self.directory_name)
        self.experiment = DataReader.read_data(self.data_name, self.directory_name)
        self.experiment = CurveApproximator.approximate_curves(
            self.experiment, 
            self.data_name,
            self.T_isotherm
        )
        Visualizer.visualize_results(self.experiment, self.data_name)
        DataSaver.save_data(self.experiment, self.data_name, self.directory_name, self.T_isotherm)

# Example usage
if __name__ == "__main__":
    directory_name = "Example for processing"
    T_isotherm = np.arange(25, 95, 5)  # Temperature range from 25°C to 90°C in steps of 5°C
    analyzer = MeltCurveAnalyzer(directory_name, T_isotherm)
    analyzer.run_analysis()

