//============================================================================
// Name        : mats_deform.cpp
// Author      : Jian Gong
// Version     : 1.0
// Copyright   : Freedom
// Description : Hello World in C, Ansi-style
//============================================================================

#include <deal.II/grid/tria.h>              // mesh Triangulation class
#include <deal.II/grid/tria_accessor.h>     // for stepping cells
#include <deal.II/grid/tria_iterator.h>     // for looping cells
#include <deal.II/grid/grid_generator.h>    // standard grid generator
#include <deal.II/grid/tria_boundary_lib.h> // boundary descriptors
#include <deal.II/grid/grid_out.h>          // grids visualization outputs
#include <deal.II/dofs/dof_handler.h>       // attach DOF to elements (vertices, lines, and cells)
#include <deal.II/dofs/dof_tools.h>         // tools for manipulating degrees of freedom
#include <deal.II/dofs/dof_renumbering.h>   // algorithm to renumber degrees of freedom
#include <deal.II/fe/fe_q.h>                // bilinear finite element (Lagrange)
#include <deal.II/fe/fe_values.h>           // used to assemble quadrature
#include <deal.II/base/quadrature_lib.h>    // used to assemble quadrature
#include <deal.II/base/function.h>          // boundary value treatment
#include <deal.II/numerics/vector_tools.h>  // boundary value treatment
#include <deal.II/numerics/matrix_tools.h>  // boundary value treatment
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/vector.h>             // tool for solvers
#include <deal.II/lac/full_matrix.h>        // tool for solvers
#include <deal.II/lac/solver_cg.h>          // Conjugate Gradient solver
#include <deal.II/lac/precondition.h>       // CG solver preconditioner
#include <deal.II/lac/constraint_matrix.h>  // Adaptive mesh
#include <deal.II/grid/grid_refinement.h>   // locally refine grids
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/grid/grid_in.h>           // input grids from disk
#include <deal.II/numerics/data_out.h>      // data output
#include <iostream>                         // c++ IO
#include <fstream>                          // c++ file output
#include <cmath>                            // c math functions
#include <sstream>                          // convert integers to strings

#include <deal.II/base/logstream.h>         // control logging behaviors
#include <map>

using namespace dealii;



// Find mesh info and output it
template<int dim>
void mesh_info(const Triangulation<dim> &tria,
               const std::string        &filename)
{
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << tria.n_active_cells() << std::endl;

  // Next loop over all faces of all cells and find how often each boundary
  // indicator is used:
  {
    std::map<unsigned int, unsigned int> boundary_count;
    typename Triangulation<dim>::active_cell_iterator
    cell = tria.begin_active(),
    endc = tria.end();
    for (; cell!=endc; ++cell)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
          {
            if (cell->face(face)->at_boundary())
              boundary_count[cell->face(face)->boundary_indicator()]++;
          }
      }

    std::cout << " boundary indicators: ";
    for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
         it!=boundary_count.end();
         ++it)
      {
        std::cout << it->first << "(" << it->second << " times) ";
      }
    std::cout << std::endl;
  }

  // Finally, produce a graphical representation of the mesh to an output file:
  std::ofstream out (filename.c_str());
  GridOut grid_out;
  grid_out.write_eps (tria, out);
  std::cout << " written to " << filename
            << std::endl
            << std::endl;
}



void make_rect_grid () {
	Triangulation<3> space;                 // create an empty 3D space
	GridGenerator::hyper_cube (space);      // generate a standard 3D cube
	space.refine_global (4);                // refine the mesh 3 times in each dimension (div 2)

	std::ofstream output_file ("space_grid.eps");
	GridOut grids;
	grids.write_eps (space, output_file);
	std::cout << "Grid written to space_grid.eps" << std::endl;
}



// -- CLASS CONSTRUCTION --
template <int dim>
class Coefficient : public Function<dim>
{
public:
  Coefficient () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double>            &values,
                           const unsigned int              component = 0) const;
};

template <int dim>
double Coefficient<dim>::value (const Point<dim> &p,
                                const unsigned int) const
{
  if (p.square() < 0.5*0.5)
    return 20;
  else
    return 1;
}

template <int dim>
void Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
                                   std::vector<double>            &values,
                                   const unsigned int              component) const
{
  const unsigned int n_points = points.size();

  Assert (values.size() == n_points,
          ExcDimensionMismatch (values.size(), n_points));

  Assert (component == 0,
          ExcIndexRange (component, 0, 1));

  for (unsigned int i=0; i<n_points; ++i)
    {
      if (points[i].square() < 0.5*0.5)
        values[i] = 20;
      else
        values[i] = 1;
    }
}



// -- CLASS CONSTRUCTION --
template <int dim>
class Mats_Deform
{
public:
  Mats_Deform ();
  ~Mats_Deform ();

  void run ();

private:
  void make_grid ();
  void setup_system ();
  void assemble_system ();
  void solve ();
  void refine_grid ();
  void output_results (const unsigned int cycle) const;

  Triangulation<dim>   triangulation;

  DoFHandler<dim>      dof_handler;
  FE_Q<dim>            fe;

  ConstraintMatrix     constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;
};

// The constructor of the class using Q2 quadratic element, fe (2)
template <int dim>
Mats_Deform<dim>::Mats_Deform ()
  :
  dof_handler (triangulation),
  fe (2)
{}

template <int dim>
Mats_Deform<dim>::~Mats_Deform ()
{
  dof_handler.clear ();
}

template <int dim>
void Mats_Deform<dim>::make_grid ()
{
//  GridGenerator::hyper_cube (triangulation, -1, 1);
//  triangulation.refine_global (4);


  std::cout << "   Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: "
            << triangulation.n_cells()
            << std::endl;
}

template <int dim>
void Mats_Deform<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());


  // After setting up all the degrees of freedoms, here are now the
  // differences compared to step-5, all of which are related to constraints
  // associated with the hanging nodes. In the class declaration, we have
  // already allocated space for an object <code>constraints</code> that will
  // hold a list of these constraints (they form a matrix, which is reflected
  // in the name of the class, but that is immaterial for the moment). Now we
  // have to fill this object. This is done using the following function calls
  // (the first clears the contents of the object that may still be left over
  // from computations on the previous mesh before the last adaptive
  // refinement):
  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);


  // Now we are ready to interpolate the ZeroFunction to our boundary with
  // indicator 0 (the whole boundary) and store the resulting constraints in
  // our <code>constraints</code> object. Note that we do not to apply the
  // boundary conditions after assembly, like we did in earlier steps.  As
  // almost all the stuff, the interpolation of boundary values works also for
  // higher order elements without the need to change your code for that. We
  // note that for proper results, it is important that the elimination of
  // boundary nodes from the system of equations happens *after* the
  // elimination of hanging nodes. For that reason we are filling the boundary
  // values into the ContraintMatrix after the hanging node constraints.
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<dim>(),
                                            constraints);


  // The next step is <code>closing</code> this object. After all constraints
  // have been added, they need to be sorted and rearranged to perform some
  // actions more efficiently. This postprocessing is done using the
  // <code>close()</code> function, after which no further constraints may be
  // added any more:
  constraints.close ();

  // Now we first build our compressed sparsity pattern like we did in the
  // previous examples. Nevertheless, we do not copy it to the final sparsity
  // pattern immediately.  Note that we call a variant of
  // make_sparsity_pattern that takes the ConstraintMatrix as the third
  // argument. We are letting the routine know that we will never write into
  // the locations given by <code>constraints</code> by setting the argument
  // <code>keep_constrained_dofs</code> to false (in other words, that we will
  // never write into entries of the matrix that correspond to constrained
  // degrees of freedom). If we were to condense the
  // constraints after assembling, we would have to pass <code>true</code>
  // instead because then we would first write into these locations only to
  // later set them to zero again during condensation.
  CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  c_sparsity,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);

  // Now all non-zero entries of the matrix are known (i.e. those from
  // regularly assembling the matrix and those that were introduced by
  // eliminating constraints). We can thus copy our intermediate object to the
  // sparsity pattern:
  sparsity_pattern.copy_from(c_sparsity);

  // Finally, the so-constructed sparsity pattern serves as the basis on top
  // of which we will create the sparse matrix:
  system_matrix.reinit (sparsity_pattern);
}

template <int dim>
void Mats_Deform<dim>::assemble_system ()
{
  const QGauss<dim>  quadrature_formula(3);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  const Coefficient<dim> coefficient;
  std::vector<double>    coefficient_values (n_q_points);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit (cell);

      coefficient.value_list (fe_values.get_quadrature_points(),
                              coefficient_values);

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (coefficient_values[q_index] *
                                   fe_values.shape_grad(i,q_index) *
                                   fe_values.shape_grad(j,q_index) *
                                   fe_values.JxW(q_index));

            cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                            1.0 *
                            fe_values.JxW(q_index));
          }

      // Finally, transfer the contributions from @p cell_matrix and
      // @p cell_rhs into the global objects.
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
    }
  // Now we are done assembling the linear system. The constraint matrix took
  // care of applying the boundary conditions and also eliminated hanging node
  // constraints. The constrained nodes are still in the linear system (there
  // is a one on the diagonal of the matrix and all other entries for this
  // line are set to zero) but the computed values are invalid. We compute the
  // correct values for these nodes at the end of the <code>solve</code>
  // function.
}

template <int dim>
void Mats_Deform<dim>::solve ()
{
  SolverControl      solver_control (1000, 1e-12);
  SolverCG<>         solver (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve (system_matrix, solution, system_rhs,
                preconditioner);

  constraints.distribute (solution);
}

template <int dim>
void Mats_Deform<dim>::refine_grid ()
{
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(3),
                                      typename FunctionMap<dim>::type(),
                                      solution,
                                      estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   0.33, 0.03);

  triangulation.execute_coarsening_and_refinement ();
}

template <int dim>
void Mats_Deform<dim>::output_results (const unsigned int cycle) const
{
  Assert (cycle < 10, ExcNotImplemented());

  std::string filename = "grid-";
  filename += ('0' + cycle);
  filename += ".eps";

  std::ofstream output (filename.c_str());

  GridOut grid_out;
  grid_out.write_eps (triangulation, output);
}

template <int dim>
void Mats_Deform<dim>::run ()
{
  for (unsigned int cycle=0; cycle<8; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        {
          GridGenerator::hyper_ball (triangulation);

          static const HyperBallBoundary<dim> boundary;
          triangulation.set_boundary (0, boundary);

          triangulation.refine_global (1);
        }
      else
        refine_grid ();


      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells()
                << std::endl;

      setup_system ();

      std::cout << "   Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

      assemble_system ();
      solve ();
      //output_results (cycle);
    }

  // After we have finished computing the solution on the finest mesh, and
  // writing all the grids to disk, we want to also write the actual solution
  // on this final mesh to a file. As already done in one of the previous
  // examples, we use the EPS format for output, and to obtain a reasonable
  // view on the solution, we rescale the z-axis by a factor of four.
  DataOutBase::EpsFlags eps_flags;
  eps_flags.z_scaling = 4;

  DataOut<dim> data_out;
  data_out.set_flags (eps_flags);

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();

  std::ofstream output ("solution.vtk");
  data_out.write_vtk (output);
}

// -- MAIN --
int main() {
    try {
    	deallog.depth_console (0);

    	Triangulation<2> triangulation;

    	GridIn<2> gridin;
    	gridin.attach_triangulation(triangulation);
    	std::ifstream f("mats.msh");
    	gridin.read_msh(f);

    	mesh_info(triangulation, "mats_grid.eps");
//     	  make_rect_grid ();
//        Mats_Deform<2> laplace_problem_2d;
//        laplace_problem_2d.run ();

    }
    // Error Catching Mechanisms (no need to change below)
	catch (std::exception &exc)
	    {
	    std::cerr << std::endl << std::endl
	              << "----------------------------------------------------"
	              << std::endl;
	    std::cerr << "Exception on processing: " << std::endl
	              << exc.what() << std::endl
	              << "Aborting!" << std::endl
	              << "----------------------------------------------------"
	              << std::endl;

	    return 1;
	  }
	catch (...) {
	    std::cerr << std::endl << std::endl
	              << "----------------------------------------------------"
	              << std::endl;
	    std::cerr << "Unknown exception!" << std::endl
	              << "Aborting!" << std::endl
	              << "----------------------------------------------------"
	              << std::endl;
	    return 1;
	}
	return 0;
}
