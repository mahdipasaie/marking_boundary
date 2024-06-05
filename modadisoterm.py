import fenics as fe
import dolfin as df
import time
import numpy as np
from dolfin import LagrangeInterpolator, refine, MeshFunction, split, grad, MPI
from fenics import project, MixedElement, FunctionSpace, TestFunctions, Function, derivative
from fenics import NonlinearVariationalProblem, NonlinearVariationalSolver



def Value_Coor_dof(solution_vector_pf, spaces_pf, comm):
    """Return value of the solution at the degrees of freedom and corresponding coordinates."""


    v_phi = spaces_pf[0]
    (Phi_answer, U_answer) = split(solution_vector_pf)
    coordinates_of_all = v_phi.tabulate_dof_coordinates()
    grad_Phi = project(fe.sqrt(fe.dot(grad(Phi_answer), grad(Phi_answer))), v_phi)
    phi_value_on_dof = grad_Phi.vector().get_local()

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


def Coordinates_Of_Int(solution_vector_pf, spaces_pf, comm):
    """Get the small mesh and return coordinates of the interface."""
    dof_Val, dof_Coor = Value_Coor_dof(solution_vector_pf, spaces_pf, comm)
    Index_list = np.concatenate(np.argwhere(dof_Val > 0.00001))
    list_coordinate_points_interface = dof_Coor[Index_list]

    # Find the minimum value of the y-coordinates
    y_coordinates = list_coordinate_points_interface[:, 1]
    # min_y = np.min(y_coordinates)
    max_y = np.max(y_coordinates)
    min_yr = max_y - 100  # window of refinement is 100
    # Filter out points where the y-coordinate is less than the minimum y value
    list_coordinate_points_interface = list_coordinate_points_interface[y_coordinates > min_yr]

    return list_coordinate_points_interface


def mark_coarse_mesh(mesh_coarse, list_coordinate_points_interface):
    """Mark the cells in the coarse mesh that have the interface points in them so they can be refined."""
    mf = MeshFunction("bool", mesh_coarse, mesh_coarse.topology().dim(), False)
    len_mf = len(mf)
    Cell_Id_List = []

    tree = mesh_coarse.bounding_box_tree()

    for Cr in list_coordinate_points_interface:
        cell_id = tree.compute_first_entity_collision(df.Point(Cr))
        if cell_id != 4294967295 and 0 <= cell_id < len_mf:
            Cell_Id_List.append(cell_id)

    Cell_Id_List = np.unique(np.array(Cell_Id_List, dtype=int))
    mf.array()[Cell_Id_List] = True

    return mf


def refine_to_min(mesh_coarse, list_coordinate_points_interface):
    """Refine coarse mesh cells that contain the interface coordinate."""

    mf = mark_coarse_mesh(mesh_coarse, list_coordinate_points_interface)
    mesh_new = fe.refine(mesh_coarse, mf, redistribute=True)

    return mesh_new



def refine_mesh(coarse_mesh, solution_vector_pf, spaces_pf, max_level, comm, precentile_threshold_of_high_gradient_U=100 ):
    """Refines the mesh based on provided parameters and updates related variables and solvers."""


    coarse_mesh_it = coarse_mesh
    # Get the coordinates of the interface
    list_coordinate_points_interface = Coordinates_Of_Int(solution_vector_pf, spaces_pf, comm)

    high_gradient_U_points = high_gradient_u_points(precentile_threshold_of_high_gradient_U, spaces_pf, solution_vector_pf, comm)


    # Combine the coordinates into a single array
    coordinates_that_wants_to_be_refined = np.vstack(
            (
                list_coordinate_points_interface, 
                high_gradient_U_points
            )
        )



    # Refine the mesh up to the maximum level specified
    for res in range(max_level):
        mesh_new = refine_to_min(coarse_mesh_it, coordinates_that_wants_to_be_refined)
        coarse_mesh_it = mesh_new


    # mesh information:
    mesh_info = {
        'n_cells': df.MPI.sum(comm, mesh_new.num_cells()),
        'hmin': df.MPI.min(comm, mesh_new.hmin()),
        'hmax': df.MPI.max(comm, mesh_new.hmax()),
        'dx_min': df.MPI.min(comm, mesh_new.hmin()) / df.sqrt(2),
        'dx_max': df.MPI.max(comm, mesh_new.hmax()) / df.sqrt(2),
    }

    # find max of Y: 
    y_coordinates = list_coordinate_points_interface[:, 1]
    max_y = np.max(y_coordinates)






    return mesh_new, mesh_info, max_y


def high_gradient_u_points(precentile_threshold_of_high_gradient_U, spaces_pf, solution_vector_pf, comm):
        
    ( _ , U_answer) = solution_vector_pf.split(deepcopy=True)
    v_project_U = spaces_pf[1]
    
    coordinates_of_all = v_project_U.tabulate_dof_coordinates()
    grad_pressure = fe.project(fe.sqrt(fe.dot(fe.grad(U_answer), fe.grad(U_answer))), v_project_U)
    pressure_value_on_dof = grad_pressure.vector().get_local()

    all_Val_dof = comm.gather(pressure_value_on_dof, root=0)
    all_point = comm.gather(coordinates_of_all, root=0)

    # Broadcast the data to all processors
    all_point = comm.bcast(all_point, root=0)
    all_Val_dof = comm.bcast(all_Val_dof, root=0)

    # Combine the data from all processors
    all_Val_dof_1 = [val for sublist in all_Val_dof for val in sublist]
    all_point_1 = [point for sublist in all_point for point in sublist]

    coordinates_of_dof_U = np.array(all_point_1)
    Value_of_dof_U = np.array(all_Val_dof_1)

    high_gradient_threshold = np.percentile(Value_of_dof_U, precentile_threshold_of_high_gradient_U)
    high_gradient_indices = np.where(Value_of_dof_U > high_gradient_threshold)[0]
    high_gradient_U_points = coordinates_of_dof_U[high_gradient_indices]

    return high_gradient_U_points