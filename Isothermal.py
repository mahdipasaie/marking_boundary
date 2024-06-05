############################## Import Needed Dependencies ################################
import dolfin as df
import fenics as fe
from dolfin import (
    NonlinearProblem, UserExpression, MeshFunction, FunctionSpace, Function, MixedElement,
    TestFunctions, TrialFunction, split, derivative, NonlinearVariationalProblem,
    NonlinearVariationalSolver, cells, grad, project, refine, Point, RectangleMesh,
    as_vector, XDMFFile, DOLFIN_EPS, sqrt, conditional, Constant, inner, Dx, lt,
    set_log_level, LogLevel, MPI, UserExpression, LagrangeInterpolator
)

import numpy as np
import matplotlib.pyplot as plt
import time
from modadisoterm import refine_mesh
from tqdm import tqdm
import random
from scipy.ndimage import binary_dilation, binary_erosion




set_log_level(LogLevel.ERROR)

#################### Define Function For Lcocal Refine #################


def refine_mesh_local( mesh , y_solid , Max_level  ): 
    global dy
    mesh_itr = mesh
    for i in range(Max_level):
        mf = MeshFunction("bool", mesh_itr, mesh_itr.topology().dim() , False )
        cells_mesh = cells( mesh_itr )
        index = 0 
        for cell in cells_mesh :

            if  cell.midpoint()[1]    <   y_solid+2*dy : 
                mf.array()[ index ] = True
            index = index + 1 
        mesh_r = refine( mesh_itr, mf )
        # Update for next loop
        mesh_itr = mesh_r
    return mesh_itr 


#############################  END  ################################


#################### Define Parallel Variables ####################

# Get the global communicator
comm = MPI.comm_world 

# Get the rank of the process
rank = MPI.rank(comm)

# Get the size of the communicator (total number of processes)
size = MPI.size(comm)

#############################  END  ################################


#################### Define Constants  #############################
T = 0

parameters = {
    'dt': 0.13,#0.13
    "dy": 0.8,#0.8     
    "Nx_aprox": 500,
    "Ny_aprox": 4000,
    "max_level": 5, # Maximum level of Coarsening
    "y_solid": 20,
    'w0': 1,
    "W_scale": 1E-8,#m
    "Tau_scale":  2.30808E-8,#seconds 
    'Tau_0': 1,
    "G": 1E7 , # k/m # it should be non-dimensionlized or y should be in meter in equation
    "V": 3E-2 , # m/s # do not scale the time cause the time is scaled in eqution 
    "m_l": 10.5,#-10.5, # K%-1 #negative sign is important?!
    "c_0": 5,# %
    'at': lambda: 1 / (2 * fe.sqrt(2.0)),
    'ep_4': 0.03,
    'k_eq': 0.48,
    'lamda': 1.377,  #1.104875
    'opk': lambda k_eq: 1 + k_eq,
    'omk': lambda k_eq: 1 - k_eq,
    'a1': 0.8839,
    'a2': 0.6267,
    'd_l': 0.6267* 1.377 ,
    "d_s":2.877E-4 ,
    'd0':8E-9,#meter
    "abs_tol": 1E-4, #1e-6
    "rel_tol": 1E-3, #1e-5
    #####################################
    'nonlinear_solver_pf': 'snes',     # "newton" , 'snes'
    'linear_solver_pf': 'gmres',       # "mumps" , "superlu_dist", 'cg', 'gmres', 'bicgstab'
    "preconditioner_pf": 'hypre_amg',       # 'hypre_amg', 'ilu', 'jacobi'
    'maximum_iterations_pf': 100,
}

# To access and compute values with dependencies, call the lambda functions with necessary arguments
parameters['at'] = parameters['at']()  # No dependencies
parameters['opk'] = parameters['opk'](parameters['k_eq'])
parameters['omk'] = parameters['omk'](parameters['k_eq'])
# parameters['d'] = parameters['d'](parameters['a2'], parameters['lamda'])


# Retrieve parameters individually from the dictionary
w0 = parameters['w0']
Tau_0 = parameters['Tau_0']
at = parameters['at']
ep_4 = parameters['ep_4']
k_eq = parameters['k_eq']
lamda = parameters['lamda']
opk = parameters['opk']
omk = parameters['omk']
a1 = parameters['a1']
a2 = parameters['a2']
d0 = parameters['d0']  # Assuming 'd0' is calculated as a1 / lamda
rel_tol = parameters['rel_tol']
abs_tol = parameters['abs_tol']
dt = parameters['dt']
Nx_aprox = parameters['Nx_aprox']
Ny_aprox = parameters['Ny_aprox']
dy = parameters['dy']
max_level = parameters['max_level']
y_solid = parameters['y_solid'] 

#############################  END  ################################


#################### Define Mesh Domain Parameters ############################

dy_coarse = 2**( max_level ) * dy
dy_coarse_init = 2**( 4 ) * dy

nx = (int)(Nx_aprox / dy_coarse ) + 1
ny = (int)(Ny_aprox / dy_coarse ) + 1

nx = nx + 1
ny = ny + 1 

Nx = nx * dy_coarse
Ny = ny * dy_coarse

nx = (int)(Nx / dy_coarse )
ny = (int)(Ny / dy_coarse )

nx_init = (int)(Nx / dy_coarse_init )
ny_init = (int)(Ny / dy_coarse_init )


parameters['Nx'] = Nx
parameters["Ny"] = Ny

#############################  END  ################################

########################## Define Mesh  ##################################

coarse_mesh = fe.RectangleMesh( Point(0, 0), Point(Nx, Ny), nx, ny)

coarse_mesh_init = fe.RectangleMesh( Point(0, 0), Point(Nx, Ny), nx_init, ny_init)


mesh = refine_mesh_local( coarse_mesh_init , y_solid + 500*dy , 4  )

# Printing Initial Mesh Informations 

# Calculate the number of cells in each mesh across all processes
number_of_coarse_mesh_cells = df.MPI.sum(comm, coarse_mesh.num_cells())
number_of_small_mesh_cells = df.MPI.sum(comm, mesh.num_cells())

if rank == 0:
    # Calculate and print details about the mesh sizes and number of cells
    min_dx_coarse = coarse_mesh.hmin() / df.sqrt(2)
    min_dx_small = mesh.hmin() / df.sqrt(2)

    print(f"Minimum Δx of Coarse Mesh = {min_dx_coarse}")
    print(f"Number Of Coarse Mesh Cells: {number_of_coarse_mesh_cells}")
    print(f"Minimum Δx of Small Mesh = {min_dx_small}")
    print(f"Number Of Small Mesh Cells: {number_of_small_mesh_cells}")


#############################  END  ####################################

#################### Define Initial Condition  ####################

# # Initial Condition :
# class InitialConditions(UserExpression):

#     def eval(self, values, x):
#         global omega,dy,y_solid
#         xp = x[0]
#         yp = x[1]
#         # # Sinusoidal perturbation with an amplitude of 5 * dy
#         perturbation_amplitude = 1*dy
#         perturbation_wavelength = 2*dy  # Wavelength remains as dy
#         perturbation =   perturbation_amplitude * np.sin(2 * np.pi * xp / perturbation_wavelength)

#         if yp < y_solid - perturbation_amplitude :  # solid
#             values[0] = 1
#             values[1] = -1
#         elif y_solid - perturbation_amplitude  <= yp <= y_solid + perturbation_amplitude:  # interface with perturbation
#             values[0] = perturbation #random.uniform(-1, 1)
#             values[1] = -1
#         else:  # liquid
#             values[0] = -1
#             values[1] = -1



#     def value_shape(self):
#         return (2,)
    
class InitialConditions(fe.UserExpression):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize the base class

        global omega,dy,y_solid
        self.dy = dy
        self.y_solid = y_solid


    def eval(self, values, x):
        xp = x[0]
        yp = x[1]
        # # Sinusoidal perturbation with an amplitude of 5 * dy
        perturbation_amplitude = 150*self.dy
        perturbation_wavelength = 200*self.dy  # Wavelength remains as dy
        perturbation =   perturbation_amplitude *(1+ np.sin(2 * np.pi * xp / perturbation_wavelength))

        if yp < self.y_solid  :  # solid
            values[0] = 1
            values[1] = -1
        elif self.y_solid   <= yp <= self.y_solid + perturbation:  # interface with perturbation
            values[0] = 1 #random.uniform(-1, 1)
            values[1] = -1
        else:  # liquid
            values[0] = -1
            values[1] = -1

    def value_shape(self):
        return (2,)
    
def Initial_Interpolate(Phi_U, Phi_U_0):
    initial_v = InitialConditions(degree=2)

    Phi_U.interpolate(initial_v)

    Phi_U_0.interpolate(initial_v)


#############################  END  ###############################


#################### Define Variables  ################################




def define_variables(mesh):
    # Define finite elements for each variable in the system
    P1 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # Order parameter Phi
    P2 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # U: dimensionless solute supersaturation

    # Create a mixed element and define the function space
    element = fe.MixedElement([P1, P2])
    function_space = fe.FunctionSpace(mesh, element)

    # Define test functions
    test_1, test_2 = fe.TestFunctions(function_space)

    # Define functions for the current and previous solutions
    solution_vector = fe.Function(function_space)    # Current solution
    solution_vector_0 = fe.Function(function_space)  # Previous solution


    # Split functions to access individual components
    Phi_answer, U_answer = fe.split(solution_vector)    # Current solution
    Phi_prev, U_prev = fe.split(solution_vector_0)        # Last step solution

    # Collapse function spaces to individual subspaces
    num_subs = function_space.num_sub_spaces()
    spaces, maps = [], []
    for i in range(num_subs):
        space_i, map_i = function_space.sub(i).collapse(collapsed_dofs=True)
        spaces.append(space_i)
        maps.append(map_i)

    # Return all the variables
    return {

        'Phi_answer': Phi_answer, 'U_answer': U_answer,
        'Phi_prev': Phi_prev, 'U_prev': U_prev,
        'solution_vector': solution_vector, 'solution_vector_0': solution_vector_0,
        'test_2': test_2, 'test_1': test_1,
        'spaces': spaces, 'function_space': function_space
    }


def calculate_dependent_variables(variables_dict, parameters):

    # Retrieve the values from the dictionary
    w0 = parameters['w0']
    ep_4 = parameters['ep_4']
    # Retrieve the values from the dictionary
    phi_answer = variables_dict['Phi_prev']

    # Define tolerance for avoiding division by zero errors
    tolerance_d = fe.sqrt(DOLFIN_EPS)  # sqrt(1e-15)

    # Calculate gradient and derivatives for anisotropy function
    grad_phi = fe.grad(phi_answer)
    mgphi = fe.inner(grad_phi, grad_phi)
    dpx = fe.Dx(phi_answer, 0)
    dpy = fe.Dx(phi_answer, 1)
    dpx = fe.variable(dpx)
    dpy = fe.variable(dpy)

    # Normalized derivatives
    nmx = -dpx / fe.sqrt(mgphi)
    nmy = -dpy / fe.sqrt(mgphi)
    norm_phi_4 = nmx**4 + nmy**4

    # Anisotropy function
    a_n = fe.conditional(
        fe.lt(fe.sqrt(mgphi), fe.sqrt(DOLFIN_EPS)),
        fe.Constant(1 - 3 * ep_4),
        1 - 3 * ep_4 + 4 * ep_4 * norm_phi_4
    )

    # Weight function based on anisotropy
    W_n = w0 * a_n

    # Derivatives of weight function w.r.t x and y
    D_w_n_x = fe.conditional(fe.lt(fe.sqrt(mgphi), tolerance_d), 0, fe.diff(W_n, dpx))
    D_w_n_y = fe.conditional(fe.lt(fe.sqrt(mgphi), tolerance_d), 0, fe.diff(W_n, dpy))

    return  {
        'D_w_n_x': D_w_n_x,
        'D_w_n_y': D_w_n_y,
        'mgphi': mgphi,
        'W_n': W_n
    }


def calculate_equation_1(variables_dict, dep_var_dict, parameters, mesh):

    global T

    # Retrieve parameters individually from the dictionary
    w0 = parameters['w0']
    k_eq = parameters['k_eq']
    lamda = parameters['lamda']
    dt = parameters['dt']
    # Retrieve the values from the dictionary
    phi_answer = variables_dict['Phi_answer']
    u_answer = variables_dict['U_answer']
    phi_prev = variables_dict['Phi_prev']
    v_test = variables_dict['test_1']
    # retrive dependent variables
    d_w_n_x = dep_var_dict['D_w_n_x']
    d_w_n_y = dep_var_dict['D_w_n_y']
    mgphi = dep_var_dict['mgphi']
    w_n = dep_var_dict['W_n']
    G = parameters['G']
    V = parameters['V']
    W_scale = parameters['W_scale']
    Tau_scale = parameters['Tau_scale']
    m_l = parameters['m_l']
    c_0 = parameters['c_0']


    # Access the spatial coordinates
    X = fe.SpatialCoordinate(mesh)
    Y = X[1]

    term4_in = mgphi * w_n * d_w_n_x
    term5_in = mgphi * w_n * d_w_n_y

    term4 = -fe.inner(term4_in, v_test.dx(0)) * fe.dx
    term5 = -fe.inner(term5_in, v_test.dx(1)) * fe.dx

    term3 = -(w_n**2 * fe.inner(fe.grad(phi_answer), fe.grad(v_test))) * fe.dx

    term2 = (
        fe.inner(
            (phi_answer - phi_answer**3) - lamda * (u_answer  + (G* W_scale)* ( Y - V* (T*Tau_scale/W_scale) )/ (m_l* c_0/k_eq * (1-k_eq))  ) * (1 - phi_answer**2) ** 2,
            v_test,
        ) * fe.dx
    )


    tau_n = (w_n / w0) ** 2 

    term1 = - fe.inner((tau_n) * (phi_answer - phi_prev) / dt, v_test) * fe.dx

    eq1 = term1 + term2 + term3 + term4 + term5

    return eq1


def calculate_equation_2(variables_dict, dep_var_dict , parameters):
    
    # Retrieve the values from the dictionary
    at = parameters['at']
    opk = parameters['opk']
    omk = parameters['omk']
    dt = parameters['dt']
    # Retrieve the values from the dictionary
    phi_answer = variables_dict['Phi_answer']
    u_answer = variables_dict['U_answer']
    phi_prev = variables_dict['Phi_prev']
    u_prev = variables_dict['U_prev']
    q_test = variables_dict['test_2']

    w_n = dep_var_dict['W_n']
    d_s = parameters['d_s']
    d_l = parameters['d_l']

    d = d_s* ( 1+phi_answer ) / 2 + d_l * ( 1-phi_answer ) / 2





    tolerance_d = fe.sqrt(DOLFIN_EPS)  # sqrt(1e-15)

    grad_phi = fe.grad(phi_answer)
    abs_grad = fe.sqrt(fe.inner(grad_phi, grad_phi))

    norm = fe.conditional(
        fe.lt(abs_grad, tolerance_d), fe.as_vector([0, 0]), grad_phi / abs_grad
    )

    dphidt = (phi_answer - phi_prev) / dt

    term6 = -fe.inner(((opk) / 2 - (omk) * phi_answer / 2) * (u_answer - u_prev) / dt, q_test) * fe.dx
    term7 = -fe.inner(d * (1 - phi_answer) / 2 * fe.grad(u_answer), fe.grad(q_test)) * fe.dx

    term8 = -(1 * at) * (1 + (omk) * u_answer) * dphidt * fe.inner(norm, fe.grad(q_test)) * fe.dx
    # term8 = -(w_n * at) * (1 + (omk) * u_answer) * dphidt * fe.inner(norm, fe.grad(q_test)) * fe.dx
    term9 = (1 + (omk) * u_answer) * dphidt / 2 * q_test * fe.dx

    eq2 = term6 + term7 + term8 + term9


    return eq2


def define_boundary_condition(variables_dict, physical_parameters_dict) :



    Nx = physical_parameters_dict['Nx']
    Ny = physical_parameters_dict['Ny']
    W = variables_dict['function_space']

    # Define boundary conditions for velocity, pressure, and temperature
    class RightBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[0], Nx)

    class TopBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[1], Ny)

    # Instantiate boundary classes
    right_boundary = RightBoundary()
    top_boundary = TopBoundary()
    # Define Dirichlet boundary conditions

    # bc_u_right = fe.DirichletBC(W.sub(1), fe.Constant(- omega), right_boundary)
    # bc_u_top = fe.DirichletBC(W.sub(1), fe.Constant(- omega), top_boundary)
    bc_u_right = fe.DirichletBC(W.sub(1), fe.Constant(-1), right_boundary)
    bc_u_top = fe.DirichletBC(W.sub(1), fe.Constant(-1), top_boundary)

    Bc = [bc_u_top, bc_u_right ]
    

    return  Bc


def define_problem(eq1, eq2, Bc, phi_u, parameters):
    
    rel_tol = parameters['rel_tol']
    abs_tol = parameters['abs_tol']
    linear_solver_pf = parameters['linear_solver_pf']
    nonlinear_solver_pf = parameters['nonlinear_solver_pf']
    preconditioner_pf = parameters['preconditioner_pf']
    maximum_iterations_pf = parameters['maximum_iterations_pf']



    L = eq1 + eq2  # Define the Lagrangian
    J = derivative(L, phi_u)  # Compute the Jacobian

    # Define the problem
    problem = NonlinearVariationalProblem(L, phi_u, J=J) # BC = None

    solver_pf = fe.NonlinearVariationalSolver(problem)

    solver_parameters = {
        'nonlinear_solver': nonlinear_solver_pf,
        'snes_solver': {
            'linear_solver': linear_solver_pf,
            'report': False,
            "preconditioner": preconditioner_pf,
            'error_on_nonconvergence': False,
            'absolute_tolerance': abs_tol,
            'relative_tolerance': rel_tol,
            'maximum_iterations': maximum_iterations_pf,
        }
    }


    solver_pf.parameters.update(solver_parameters)

    return solver_pf



def update_solver_on_new_mesh(mesh_new, parameters, old_solution_vector= None, old_solution_vector_0=None):

    # Define variables on the new mesh
    variables_dict = define_variables(mesh_new)
    # Extract each variable from the dictionary

    solution_vector = variables_dict['solution_vector']
    solution_vector_0 = variables_dict['solution_vector_0']
    spaces = variables_dict['spaces']

    # Interpolate previous step solutions from old mesh functions else interpolate initial condition
    if old_solution_vector is not None and old_solution_vector_0 is not None:

        LagrangeInterpolator.interpolate(solution_vector, old_solution_vector)
        LagrangeInterpolator.interpolate(solution_vector_0, old_solution_vector_0)

    else:
        # Interpolate initial condition
        Initial_Interpolate(solution_vector, solution_vector_0)


    # Calculate dependent variables
    dep_var_dict = calculate_dependent_variables(variables_dict, parameters)


    # Define equations
    eq1 = calculate_equation_1(variables_dict, dep_var_dict, parameters, mesh_new) 
        
    eq2 = calculate_equation_2(variables_dict, dep_var_dict , parameters)

    Bc = define_boundary_condition(variables_dict, parameters)


    # Define problem
    solver = define_problem(eq1, eq2, Bc, solution_vector, parameters)

    # Return whatever is appropriate, such as the solver or a status message
    return solver, solution_vector, solution_vector_0, spaces, Bc

#############################  END  #############################


#################### Define Step 1 For Solving  ####################

solver, solution_vector, solution_vector_0, spaces, Bc = update_solver_on_new_mesh(mesh, parameters)


#############################  END  ###############################



############################ File Section #########################



file = fe.XDMFFile("Ghosh.xdmf" ) # File Name To Save #


def write_simulation_data(Sol_Func, time, file, variable_names ):

    
    # Configure file parameters
    file.parameters["rewrite_function_mesh"] = True
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True

    # Split the combined function into its components
    functions = Sol_Func.split(deepcopy=True)

    # Check if the number of variable names matches the number of functions
    if variable_names and len(variable_names) != len(functions):
        raise ValueError("The number of variable names must match the number of functions.")

    # Rename and write each function to the file
    for i, func in enumerate(functions):
        name = variable_names[i] if variable_names else f"Variable_{i}"
        func.rename(name, "solution")
        file.write(func, time)

    file.close()



# T = 0

variable_names = [  "Phi", "U" ]  # Adjust as needed


write_simulation_data( solution_vector_0, T, file , variable_names=variable_names )


#############################  END  ###############################


############ Initialize for Adaptive Mesh #########################


max_y = 0

for it in tqdm(range(0, 1)):

    T = T + dt

    solver.solve()

    solution_vector_0.vector()[:] = solution_vector.vector()  # update the solution


    # Refining mesh
    if it % 10 == 0 and it> 10 :

        start = time.perf_counter()
        mesh_new, mesh_info, max_y = refine_mesh(coarse_mesh, solution_vector_0, spaces, max_level, comm )
        # Update the solver and solution on the new mesh
        solver, solution_vector, solution_vector_0, spaces, Bc = update_solver_on_new_mesh(mesh_new, parameters, solution_vector, solution_vector_0)
        end = time.perf_counter()
        Time_Lentgh_of_refinment = end - start

    if it % 10 == 0 and rank == 0:
        print("")
        print(f"Max Y: {max_y}", flush=True)

    if  it % 10 == 0 :

        write_simulation_data( solution_vector_0,  T , file , variable_names )
        


    # print information of simulation
    if it % 1000 == 500 :
        n_cells = mesh_info["n_cells"]
        dx_min = mesh_info["dx_min"]
        dx_max = mesh_info["dx_max"]
        


        simulation_status_message = (
        f"Simulation Status:\n"
        f"├─ Iteration: {it}\n"
        f"├─ Simulation Time: {T:.2f} scaled unit \n"
        f"└─ Mesh Refinement Details:\n"
        f"   ├─ Number of cells: {n_cells}\n"
        f"   ├─ Minimum ΔX: {dx_min:.4f}\n"
        f"   ├─ Maximum ΔX: {dx_max:.4f}\n"
        f"   └─ Refinement Computation Time: {Time_Lentgh_of_refinment:.2f} seconds\n"
        )

        print(simulation_status_message, flush=True)

        



#############################  END  ###############################


# class SolidCells(fe.SubDomain):
#     def __init__(self, solution_vector, tol=1e-8):
#         super().__init__()
#         self.solution_vector = solution_vector
#         self.tol = tol
#     def inside(self, x, on_boundary):
#         # Assuming phi is the first component of the solution vector
#         phi_value = self.solution_vector.split()[0](x)
#         return fe.near(phi_value, 1, self.tol)

# def mark_solid_boundary(mesh, solution_vector, tol=1e-8):
#     # Create a MeshFunction to mark the boundary
#     cell_markers = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
#     # Initialize the SubDomain for solid boundary
#     solid_cells = SolidCells(solution_vector, tol)
#     # Mark the solid boundary with 1
#     solid_cells.mark(cell_markers, 1)
#     return cell_markers


# boundary_markers = mark_solid_boundary(mesh, solution_vector_0)

# # Save or visualize the boundary markers
# # Save or visualize the boundary markers
# xdmf_file = fe.XDMFFile("solid_boundary_markers.xdmf")
# xdmf_file.write(boundary_markers)
# xdmf_file.close()




def Value_Coor_dof(solution_vector_pf, spaces_pf, comm):
    """Return value of the solution at the degrees of freedom and corresponding coordinates."""


    v_phi = spaces_pf[0]
    (phi, u) = solution_vector_pf.split(deepcopy=True)
    coordinates_of_all = v_phi.tabulate_dof_coordinates()
    phi_value_on_dof = phi.vector().get_local()

    all_Val_dof = comm.gather(phi_value_on_dof, root=0)
    all_point = comm.gather(coordinates_of_all, root=0)

    # Broadcast the data to all processors
    all_point = comm.bcast(all_point, root=0)
    all_Val_dof = comm.bcast(all_Val_dof, root=0)

    # Combine the data from all processors
    all_Val_dof_1 = [val for sublist in all_Val_dof for val in sublist]
    all_point_1 = [point for sublist in all_point for point in sublist]

    point = np.array(all_point_1)
    Val_dof = np.array(all_Val_dof_1)

    return Val_dof, point

def coordinates_of_marking(solution_vector_pf, spaces_pf, comm):
    """Get the small mesh and return coordinates of the interface."""
    dof_value, dof_Coor = Value_Coor_dof(solution_vector_pf, spaces_pf, comm)
    Index_list = np.concatenate(np.argwhere(dof_value > 0.99))
    list_coordinate_points = dof_Coor[Index_list]
    return list_coordinate_points

def mark_mesh(mesh, solution_vector_pf, spaces_pf, comm):
    list_coordinate_points =    coordinates_of_marking(solution_vector_pf, spaces_pf, comm) 
    mf = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    len_mf = len(mf)
    cell_id_list = []
    tree = mesh.bounding_box_tree()
    for Cr in list_coordinate_points:
        cell_id = tree.compute_first_entity_collision(df.Point(Cr))
        if cell_id != 4294967295 and 0 <= cell_id < len_mf:
            cell_id_list.append(cell_id)
    cell_id_list = np.unique(np.array(cell_id_list, dtype=int))
    mf.array()[cell_id_list] = 1

    cell_array = mf.array()
    cell_array = binary_dilation(cell_array, iterations=6)
    cell_array = binary_erosion(cell_array, iterations=6)
    mf.set_values(cell_array.astype(np.int32))

    return mf

mf = mark_mesh(mesh, solution_vector_0, spaces, comm)



# indicator_function = create_indicator_function(mesh, mf,  spaces[0])

(phi, u) = solution_vector_0.split(deepcopy=True)
xdmf_file = fe.XDMFFile("solid_boundary_markers.xdmf")
xdmf_file.parameters["rewrite_function_mesh"] = True
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
mf.rename("Solid Boundary", "Solid Boundary")
xdmf_file.write(mf)
# phi.rename("Phi", "Phi")
# xdmf_file.write(phi,0)
xdmf_file.close()