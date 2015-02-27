//============================================================================
// Name        : mats_deform.cpp
// Author      : Jian Gong
// Version     : 1.0
// Copyright   : Freedom
// Description : Simulate growth & deformation of mats using FEM in Deal.II
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

namespace mats_deform
{
  using namespace dealii;

  // -- CLASS definition
  template <int dim>
  class FluidStructureProblem
  {
  public:
    FluidStructureProblem (const unsigned int stokes_degree,
                           const unsigned int elasticity_degree);
    void run ();

  private:
    enum
    {
      fluid_domain_id,
      solid_domain_id
    };

    static bool
    cell_is_in_fluid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);

    static bool
    cell_is_in_solid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);

    void make_grid ();
    void set_active_fe_indices ();
    void setup_dofs ();
    void assemble_system ();
    void assemble_interface_term (const FEFaceValuesBase<dim>          &elasticity_fe_face_values,
                                  const FEFaceValuesBase<dim>          &stokes_fe_face_values,
                                  std::vector<Tensor<1,dim> >          &elasticity_phi,
                                  std::vector<SymmetricTensor<2,dim> > &stokes_symgrad_phi_u,
                                  std::vector<double>                  &stokes_phi_p,
                                  FullMatrix<double>                   &local_interface_matrix) const;
    void solve ();
    void output_results (const unsigned int refinement_cycle) const;
    void refine_mesh ();

    const unsigned int    stokes_degree;
    const unsigned int    elasticity_degree;

    Triangulation<dim>    triangulation;
    FESystem<dim>         stokes_fe;
    FESystem<dim>         elasticity_fe;
    hp::FECollection<dim> fe_collection;
    hp::DoFHandler<dim>   dof_handler;

    ConstraintMatrix      constraints;

    SparsityPattern       sparsity_pattern;
    SparseMatrix<double>  system_matrix;

    Vector<double>        solution;
    Vector<double>        system_rhs;

    const double          viscosity;
    const double          lambda;
    const double          mu;
  };

  // -- BOUNDARY VALUES and RHS
  template <int dim>
  class StokesBoundaryValues : public Function<dim>
  {
  public:
    StokesBoundaryValues () : Function<dim>(dim+1+dim) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };

  template <int dim>
  double
    StokesBoundaryValues<dim>::value (const Point<dim>  &p,
                                           const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));

    switch (component)
    {
    case 14:      // LEFT
      return 1;
    case 15:      // BOTTOM
      return 0;
    case 16:      // RIGHT
          return 1;
    case 17:      // TOP
          return 1;
    default:
      Assert (false, ExcNotImplemented());
    }
  }

  template <int dim>
  void
    StokesBoundaryValues<dim>::vector_value (const Point<dim> &p,
                                           Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = StokesBoundaryValues<dim>::value (p, c);
  }

  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(dim+1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };

  template <int dim>
  double
    RightHandSide<dim>::value (const Point<dim>  &/*p*/,
                             const unsigned int /*component*/) const
  {
    return 0;
  }

  template <int dim>
  void
    RightHandSide<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = RightHandSide<dim>::value (p, c);
  }

  // -- CONSTRUCTORS
  template <int dim>
  FluidStructureProblem<dim>::
  FluidStructureProblem (const unsigned int stokes_degree,
                         const unsigned int elasticity_degree)
    :
    stokes_degree (stokes_degree),
    elasticity_degree (elasticity_degree),
    triangulation (Triangulation<dim>::maximum_smoothing),
    stokes_fe (FE_Q<dim>(stokes_degree+1), dim,
               FE_Q<dim>(stokes_degree), 1,
               FE_Nothing<dim>(), dim),
    elasticity_fe (FE_Nothing<dim>(), dim,
                   FE_Nothing<dim>(), 1,
                   FE_Q<dim>(elasticity_degree), dim),
    dof_handler (triangulation),
    viscosity (2),
    lambda (1),
    mu (1)
  {
    fe_collection.push_back (stokes_fe);
    fe_collection.push_back (elasticity_fe);
  }

  template <int dim>
  bool
  FluidStructureProblem<dim>::
  cell_is_in_fluid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == fluid_domain_id);
  }

  template <int dim>
  bool
  FluidStructureProblem<dim>::
  cell_is_in_solid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == solid_domain_id);
  }

  template <int dim>
  void
  FluidStructureProblem<dim>::make_grid ()
  {
    Triangulation<2> triangulation;

    GridIn<2> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f("mats.msh");
    gridin.read_msh(f);

    mesh_info(triangulation, "mats_grid_2d.eps");

    // Setting Material ID
    for (typename Triangulation<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
      if ( cell->material_id() == 11 )
        cell->set_material_id (fluid_domain_id);
      else if ( cell->material_id() == 13 )
        cell->set_material_id (solid_domain_id);
  }

  template <int dim>
  void
  FluidStructureProblem<dim>::set_active_fe_indices ()
  {
    for (typename hp::DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
      {
        if (cell_is_in_fluid_domain(cell))
          cell->set_active_fe_index (0);
        else if (cell_is_in_solid_domain(cell))
          cell->set_active_fe_index (1);
        else
          Assert (false, ExcNotImplemented());
      }
  }

  template <int dim>
  void
  FluidStructureProblem<dim>::setup_dofs ()
  {
    set_active_fe_indices ();
    dof_handler.distribute_dofs (fe_collection);

    {
      constraints.clear ();
      DoFTools::make_hanging_node_constraints (dof_handler,
                                               constraints);

      const FEValuesExtractors::Vector velocities(0);
      VectorTools::interpolate_boundary_values (dof_handler,
                                                1,
                                                StokesBoundaryValues<dim>(),
                                                constraints,
                                                fe_collection.component_mask(velocities));

      const FEValuesExtractors::Vector displacements(dim+1);
      VectorTools::interpolate_boundary_values (dof_handler,
                                                0,
                                                ZeroFunction<dim>(dim+1+dim),
                                                constraints,
                                                fe_collection.component_mask(displacements));
    }

    // There are more constraints we have to handle, though: we have to make
    // sure that the velocity is zero at the interface between fluid and solid.
    {
      std::vector<types::global_dof_index> local_face_dof_indices (stokes_fe.dofs_per_face);
      for (typename hp::DoFHandler<dim>::active_cell_iterator
           cell = dof_handler.begin_active();
           cell != dof_handler.end(); ++cell)
        if (cell_is_in_fluid_domain (cell))
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (!cell->at_boundary(f))
            {
              bool face_is_on_interface = false;

              if ((cell->neighbor(f)->has_children() == false)
                  &&
                  (cell_is_in_solid_domain (cell->neighbor(f))))
                face_is_on_interface = true;
              else if (cell->neighbor(f)->has_children() == true)
              {
                for (unsigned int sf=0; sf<cell->face(f)->n_children(); ++sf)
                  if (cell_is_in_solid_domain (cell->neighbor_child_on_subface(f, sf)))
                  {
                    face_is_on_interface = true;
                    break;
                  }
              }

              if (face_is_on_interface)
              {
                cell->face(f)->get_dof_indices (local_face_dof_indices, 0);
                for (unsigned int i=0; i<local_face_dof_indices.size(); ++i)
                  if (stokes_fe.face_system_to_component_index(i).first < dim)
                    constraints.add_line (local_face_dof_indices[i]);
              }
            }
    }

    // At the end of all this, we can declare to the constraints object that
    // we now have all constraints ready to go and that the object can rebuild
    // its internal data structures for better efficiency:
    constraints.close ();

    std::cout << "   Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    // In the rest of this function we create a sparsity pattern as discussed
    // extensively in the introduction, and use it to initialize the matrix;
    // then also set vectors to their correct sizes:
    {
      CompressedSimpleSparsityPattern csp (dof_handler.n_dofs(),
                                           dof_handler.n_dofs());

      Table<2,DoFTools::Coupling> cell_coupling (fe_collection.n_components(),
                                                 fe_collection.n_components());
      Table<2,DoFTools::Coupling> face_coupling (fe_collection.n_components(),
                                                 fe_collection.n_components());

      for (unsigned int c=0; c<fe_collection.n_components(); ++c)
        for (unsigned int d=0; d<fe_collection.n_components(); ++d)
        {
          if (((c<dim+1) && (d<dim+1)
              && !((c==dim) && (d==dim)))
              || ((c>=dim+1) && (d>=dim+1)))
            cell_coupling[c][d] = DoFTools::always;

          if ((c>=dim+1) && (d<dim+1))
            face_coupling[c][d] = DoFTools::always;
        }

      DoFTools::make_flux_sparsity_pattern (dof_handler, csp,
                                            cell_coupling, face_coupling);
      constraints.condense (csp);
      sparsity_pattern.copy_from (csp);
    }

    system_matrix.reinit (sparsity_pattern);

    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
  }

  template <int dim>
  void
  FluidStructureProblem<dim>::assemble_system ()
  {
    system_matrix=0;
    system_rhs=0;

    const QGauss<dim> stokes_quadrature(stokes_degree+2);
    const QGauss<dim> elasticity_quadrature(elasticity_degree+2);

    hp::QCollection<dim>  q_collection;
    q_collection.push_back (stokes_quadrature);
    q_collection.push_back (elasticity_quadrature);

    hp::FEValues<dim> hp_fe_values (fe_collection, q_collection,
                                    update_values    |
                                    update_quadrature_points  |
                                    update_JxW_values |
                                    update_gradients);

    const QGauss<dim-1> common_face_quadrature(std::max (stokes_degree+2,
                                                         elasticity_degree+2));

    FEFaceValues<dim>    stokes_fe_face_values (stokes_fe,
                                                common_face_quadrature,
                                                update_JxW_values |
                                                update_normal_vectors |
                                                update_gradients);
    FEFaceValues<dim>    elasticity_fe_face_values (elasticity_fe,
                                                    common_face_quadrature,
                                                    update_values);
    FESubfaceValues<dim> stokes_fe_subface_values (stokes_fe,
                                                   common_face_quadrature,
                                                   update_JxW_values |
                                                   update_normal_vectors |
                                                   update_gradients);
    FESubfaceValues<dim> elasticity_fe_subface_values (elasticity_fe,
                                                       common_face_quadrature,
                                                       update_values);

    // ...to objects that are needed to describe the local contributions to
    // the global linear system...
    const unsigned int        stokes_dofs_per_cell     = stokes_fe.dofs_per_cell;
    const unsigned int        elasticity_dofs_per_cell = elasticity_fe.dofs_per_cell;

    FullMatrix<double>        local_matrix;
    FullMatrix<double>        local_interface_matrix (elasticity_dofs_per_cell,
                                                      stokes_dofs_per_cell);
    Vector<double>            local_rhs;

    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<types::global_dof_index> neighbor_dof_indices (stokes_dofs_per_cell);

    const RightHandSide<dim>  right_hand_side;

    // ...to variables that allow us to extract certain components of the
    // shape functions and cache their values rather than having to recompute
    // them at every quadrature point:
    const FEValuesExtractors::Vector     velocities (0);
    const FEValuesExtractors::Scalar     pressure (dim);
    const FEValuesExtractors::Vector     displacements (dim+1);

    std::vector<SymmetricTensor<2,dim> > stokes_symgrad_phi_u (stokes_dofs_per_cell);
    std::vector<double>                  stokes_div_phi_u     (stokes_dofs_per_cell);
    std::vector<double>                  stokes_phi_p         (stokes_dofs_per_cell);

    std::vector<Tensor<2,dim> >          elasticity_grad_phi (elasticity_dofs_per_cell);
    std::vector<double>                  elasticity_div_phi  (elasticity_dofs_per_cell);
    std::vector<Tensor<1,dim> >          elasticity_phi      (elasticity_dofs_per_cell);

    // Then comes the main loop over all cells and, as in step-27, the
    // initialization of the hp::FEValues object for the current cell and the
    // extraction of a FEValues object that is appropriate for the current
    // cell:
    typename hp::DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
      hp_fe_values.reinit (cell);

      const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

      local_matrix.reinit (cell->get_fe().dofs_per_cell,
                           cell->get_fe().dofs_per_cell);
      local_rhs.reinit (cell->get_fe().dofs_per_cell);

      // With all of this done, we continue to assemble the cell terms for
      // cells that are part of the Stokes and elastic regions. While we
      // could in principle do this in one formula, in effect implementing
      // the one bilinear form stated in the introduction, we realize that
      // our finite element spaces are chosen in such a way that on each
      // cell, one set of variables (either velocities and pressure, or
      // displacements) are always zero, and consequently a more efficient
      // way of computing local integrals is to do only what's necessary
      // based on an <code>if</code> clause that tests which part of the
      // domain we are in.
      //
      // The actual computation of the local matrix is the same as in
      // step-22 as well as that given in the @ref vector_valued
      // documentation module for the elasticity equations:
      if (cell_is_in_fluid_domain (cell))
      {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        Assert (dofs_per_cell == stokes_dofs_per_cell,
                ExcInternalError());

        for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
        {
          for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            stokes_symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
            stokes_div_phi_u[k]     = fe_values[velocities].divergence (k, q);
            stokes_phi_p[k]         = fe_values[pressure].value (k, q);
          }

          for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            local_matrix(i,j) += (2 * viscosity * stokes_symgrad_phi_u[i] * stokes_symgrad_phi_u[j]
                               - stokes_div_phi_u[i] * stokes_phi_p[j]
                               - stokes_phi_p[i] * stokes_div_phi_u[j])
                               * fe_values.JxW(q);
        }
      }
      else
      {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        Assert (dofs_per_cell == elasticity_dofs_per_cell,
                ExcInternalError());

        for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
        {
          for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            elasticity_grad_phi[k] = fe_values[displacements].gradient (k, q);
            elasticity_div_phi[k]  = fe_values[displacements].divergence (k, q);
          }

          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              local_matrix(i,j)
              +=  (lambda * elasticity_div_phi[i] * elasticity_div_phi[j]
              +   mu * scalar_product(elasticity_grad_phi[i], elasticity_grad_phi[j])
              +   mu * scalar_product(elasticity_grad_phi[i], transpose(elasticity_grad_phi[j]))
                  )
              * fe_values.JxW(q);
            }
        }
      }

      // Once we have the contributions from cell integrals, we copy them
      // into the global matrix (taking care of constraints right away,
      // through the ConstraintMatrix::distribute_local_to_global
      // function). Note that we have not written anything into the
      // <code>local_rhs</code> variable, though we still need to pass it
      // along since the elimination of nonzero boundary values requires the
      // modification of local and consequently also global right hand side
      // values:
      local_dof_indices.resize (cell->get_fe().dofs_per_cell);
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (local_matrix, local_rhs,
          local_dof_indices,
          system_matrix, system_rhs);

      // The more interesting part of this function is where we see about
      // face terms along the interface between the two subdomains. To this
      // end, we first have to make sure that we only assemble them once
      // even though a loop over all faces of all cells would encounter each
      // part of the interface twice. We arbitrarily make the decision that
      // we will only evaluate interface terms if the current cell is part
      // of the solid subdomain and if, consequently, a face is not at the
      // boundary and the potential neighbor behind it is part of the fluid
      // domain. Let's start with these conditions:
      if (cell_is_in_solid_domain (cell))
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->at_boundary(f) == false)
          {
            // At this point we know that the current cell is a candidate
            // for integration and that a neighbor behind face
            // <code>f</code> exists. There are now three possibilities:
            //
            // - The neighbor is at the same refinement level and has no
            //   children.
            // - The neighbor has children.
            // - The neighbor is coarser.
            //
            // In all three cases, we are only interested in it if it is
            // part of the fluid subdomain. So let us start with the first
            // and simplest case: if the neighbor is at the same level,
            // has no children, and is a fluid cell, then the two cells
            // share a boundary that is part of the interface along which
            // we want to integrate interface terms. All we have to do is
            // initialize two FEFaceValues object with the current face
            // and the face of the neighboring cell (note how we find out
            // which face of the neighboring cell borders on the current
            // cell) and pass things off to the function that evaluates
            // the interface terms (the third through fifth arguments to
            // this function provide it with scratch arrays). The result
            // is then again copied into the global matrix, using a
            // function that knows that the DoF indices of rows and
            // columns of the local matrix result from different cells:
            if ((cell->neighbor(f)->level() == cell->level())
                &&
                (cell->neighbor(f)->has_children() == false)
                &&
                cell_is_in_fluid_domain (cell->neighbor(f)))
            {
              elasticity_fe_face_values.reinit (cell, f);
              stokes_fe_face_values.reinit (cell->neighbor(f),
                                            cell->neighbor_of_neighbor(f));

              assemble_interface_term (elasticity_fe_face_values, stokes_fe_face_values,
                                       elasticity_phi, stokes_symgrad_phi_u, stokes_phi_p,
                                       local_interface_matrix);

              cell->neighbor(f)->get_dof_indices (neighbor_dof_indices);
              constraints.distribute_local_to_global(local_interface_matrix,
                                                     local_dof_indices,
                                                     neighbor_dof_indices,
                                                     system_matrix);
            }

            // The second case is if the neighbor has further children. In
            // that case, we have to loop over all the children of the
            // neighbor to see if they are part of the fluid subdomain. If
            // they are, then we integrate over the common interface,
            // which is a face for the neighbor and a subface of the
            // current cell, requiring us to use an FEFaceValues for the
            // neighbor and an FESubfaceValues for the current cell:
            else if ((cell->neighbor(f)->level() == cell->level())
                     &&
                     (cell->neighbor(f)->has_children() == true))
            {
              for (unsigned int subface=0;
                   subface<cell->face(f)->n_children();
                   ++subface)
                if (cell_is_in_fluid_domain (cell->neighbor_child_on_subface(f, subface)))
                {
                  elasticity_fe_subface_values.reinit (cell,
                                                       f,
                                                       subface);
                  stokes_fe_face_values.reinit (cell->neighbor_child_on_subface (f, subface),
                                                cell->neighbor_of_neighbor(f));

                  assemble_interface_term (elasticity_fe_subface_values,
                                           stokes_fe_face_values,
                                           elasticity_phi,
                                           stokes_symgrad_phi_u, stokes_phi_p,
                                           local_interface_matrix);

                  cell->neighbor_child_on_subface (f, subface)
                      ->get_dof_indices (neighbor_dof_indices);
                  constraints.distribute_local_to_global(local_interface_matrix,
                                                         local_dof_indices,
                                                         neighbor_dof_indices,
                                                         system_matrix);
                }
            }

            // The last option is that the neighbor is coarser. In that
            // case we have to use an FESubfaceValues object for the
            // neighbor and a FEFaceValues for the current cell; the rest
            // is the same as before:
            else if (cell->neighbor_is_coarser(f)
                     &&
                     cell_is_in_fluid_domain(cell->neighbor(f)))
            {
              elasticity_fe_face_values.reinit (cell, f);
              stokes_fe_subface_values.reinit (cell->neighbor(f),
                                               cell->neighbor_of_coarser_neighbor(f).first,
                                               cell->neighbor_of_coarser_neighbor(f).second);

              assemble_interface_term (elasticity_fe_face_values,
                                       stokes_fe_subface_values,
                                       elasticity_phi,
                                       stokes_symgrad_phi_u, stokes_phi_p,
                                       local_interface_matrix);

              cell->neighbor(f)->get_dof_indices (neighbor_dof_indices);
              constraints.distribute_local_to_global(local_interface_matrix,
                                                     local_dof_indices,
                                                     neighbor_dof_indices,
                                                     system_matrix);
            }
          }
    }
  }

  template <int dim>
  void
  FluidStructureProblem<dim>::assemble_interface_term
                          (const FEFaceValuesBase<dim>          &elasticity_fe_face_values,
                           const FEFaceValuesBase<dim>          &stokes_fe_face_values,
                           std::vector<Tensor<1,dim> >          &elasticity_phi,
                           std::vector<SymmetricTensor<2,dim> > &stokes_symgrad_phi_u,
                           std::vector<double>                  &stokes_phi_p,
                           FullMatrix<double>                   &local_interface_matrix) const
  {
    Assert (stokes_fe_face_values.n_quadrature_points ==
            elasticity_fe_face_values.n_quadrature_points,
            ExcInternalError());
    const unsigned int n_face_quadrature_points
      = elasticity_fe_face_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);
    const FEValuesExtractors::Vector displacements (dim+1);

    local_interface_matrix = 0;
    for (unsigned int q=0; q<n_face_quadrature_points; ++q)
    {
      const Tensor<1,dim> normal_vector = stokes_fe_face_values.normal_vector(q);

      for (unsigned int k=0; k<stokes_fe_face_values.dofs_per_cell; ++k)
        stokes_symgrad_phi_u[k] = stokes_fe_face_values[velocities].symmetric_gradient (k, q);
      for (unsigned int k=0; k<elasticity_fe_face_values.dofs_per_cell; ++k)
        elasticity_phi[k] = elasticity_fe_face_values[displacements].value (k,q);

      for (unsigned int i=0; i<elasticity_fe_face_values.dofs_per_cell; ++i)
        for (unsigned int j=0; j<stokes_fe_face_values.dofs_per_cell; ++j)
          local_interface_matrix(i,j) += -((2 * viscosity *
                                           (stokes_symgrad_phi_u[j] *
                                            normal_vector)
                                            +
                                            stokes_phi_p[j] *
                                            normal_vector) *
                                            elasticity_phi[i] *
                                            stokes_fe_face_values.JxW(q));
    }
  }

  template <int dim>
  void
  FluidStructureProblem<dim>::solve ()
  {
    SparseDirectUMFPACK direct_solver;
    direct_solver.initialize (system_matrix);
    direct_solver.vmult (solution, system_rhs);

    constraints.distribute (solution);
  }

  template <int dim>
  void
  FluidStructureProblem<dim>::output_results (const unsigned int refinement_cycle)  const
  {
    std::vector<std::string> solution_names (dim, "velocity");
    solution_names.push_back ("pressure");
    for (unsigned int d=0; d<dim; ++d)
      solution_names.push_back ("displacement");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation
    .push_back (DataComponentInterpretation::component_is_scalar);
    for (unsigned int d=0; d<dim; ++d)
      data_component_interpretation
      .push_back (DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim,hp::DoFHandler<dim> > data_out;
    data_out.attach_dof_handler (dof_handler);

    data_out.add_data_vector (solution, solution_names,
                              DataOut<dim,hp::DoFHandler<dim> >::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches ();

    std::ostringstream filename;
    filename << "solution-"
             << Utilities::int_to_string (refinement_cycle, 2)
             << ".vtk";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
  }

  template <int dim>
  void
  FluidStructureProblem<dim>::refine_mesh ()
  {
    Vector<float>
    stokes_estimated_error_per_cell (triangulation.n_active_cells());
    Vector<float>
    elasticity_estimated_error_per_cell (triangulation.n_active_cells());

    const QGauss<dim-1> stokes_face_quadrature(stokes_degree+2);
    const QGauss<dim-1> elasticity_face_quadrature(elasticity_degree+2);

    hp::QCollection<dim-1> face_q_collection;
    face_q_collection.push_back (stokes_face_quadrature);
    face_q_collection.push_back (elasticity_face_quadrature);

    const FEValuesExtractors::Vector velocities(0);
    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        face_q_collection,
                                        typename FunctionMap<dim>::type(),
                                        solution,
                                        stokes_estimated_error_per_cell,
                                        fe_collection.component_mask(velocities));

    const FEValuesExtractors::Vector displacements(dim+1);
    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        face_q_collection,
                                        typename FunctionMap<dim>::type(),
                                        solution,
                                        elasticity_estimated_error_per_cell,
                                        fe_collection.component_mask(displacements));

    // We then normalize error estimates by dividing by their norm and scale
    // the fluid error indicators by a factor of 4 as discussed in the
    // introduction. The results are then added together into a vector that
    // contains error indicators for all cells:
    stokes_estimated_error_per_cell
    *= 4. / stokes_estimated_error_per_cell.l2_norm();
    elasticity_estimated_error_per_cell
    *= 1. / elasticity_estimated_error_per_cell.l2_norm();

    Vector<float>
    estimated_error_per_cell (triangulation.n_active_cells());

    estimated_error_per_cell += stokes_estimated_error_per_cell;
    estimated_error_per_cell += elasticity_estimated_error_per_cell;

    // The second to last part of the function, before actually refining the
    // mesh, involves a heuristic that we have already mentioned in the
    // introduction: because the solution is discontinuous, the
    // KellyErrorEstimator class gets all confused about cells that sit at the
    // boundary between subdomains: it believes that the error is large there
    // because the jump in the gradient is large, even though this is entirely
    // expected and a feature that is in fact present in the exact solution as
    // well and therefore not indicative of any numerical error.
    //
    // Consequently, we set the error indicators to zero for all cells at the
    // interface; the conditions determining which cells this affects are
    // slightly awkward because we have to account for the possibility of
    // adaptively refined meshes, meaning that the neighboring cell can be
    // coarser than the current one, or could in fact be refined some
    // more. The structure of these nested conditions is much the same as we
    // encountered when assembling interface terms in
    // <code>assemble_system</code>.
    {
      unsigned int cell_index = 0;
      for (typename hp::DoFHandler<dim>::active_cell_iterator
           cell = dof_handler.begin_active();
           cell != dof_handler.end(); ++cell, ++cell_index)
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell_is_in_solid_domain (cell))
            {
              if ((cell->at_boundary(f) == false)
                  &&
                  (((cell->neighbor(f)->level() == cell->level())
                    &&
                    (cell->neighbor(f)->has_children() == false)
                    &&
                    cell_is_in_fluid_domain (cell->neighbor(f)))
                   ||
                   ((cell->neighbor(f)->level() == cell->level())
                    &&
                    (cell->neighbor(f)->has_children() == true)
                    &&
                    (cell_is_in_fluid_domain (cell->neighbor_child_on_subface
                                              (f, 0))))
                   ||
                   (cell->neighbor_is_coarser(f)
                    &&
                    cell_is_in_fluid_domain(cell->neighbor(f)))
                  ))
                estimated_error_per_cell(cell_index) = 0;
            }
          else
            {
              if ((cell->at_boundary(f) == false)
                  &&
                  (((cell->neighbor(f)->level() == cell->level())
                    &&
                    (cell->neighbor(f)->has_children() == false)
                    &&
                    cell_is_in_solid_domain (cell->neighbor(f)))
                   ||
                   ((cell->neighbor(f)->level() == cell->level())
                    &&
                    (cell->neighbor(f)->has_children() == true)
                    &&
                    (cell_is_in_solid_domain (cell->neighbor_child_on_subface
                                              (f, 0))))
                   ||
                   (cell->neighbor_is_coarser(f)
                    &&
                    cell_is_in_solid_domain(cell->neighbor(f)))
                  ))
                estimated_error_per_cell(cell_index) = 0;
            }
    }

    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     0.3, 0.0);
    triangulation.execute_coarsening_and_refinement ();
  }

  template <int dim>
  void FluidStructureProblem<dim>::run ()
  {
    make_grid ();

    for (unsigned int refinement_cycle = 0; refinement_cycle<10-2*dim;
        ++refinement_cycle)
    {
      std::cout << "Refinement cycle " << refinement_cycle << std::endl;

      if (refinement_cycle > 0)
        refine_mesh ();

      setup_dofs ();

      std::cout << "   Assembling..." << std::endl;
      assemble_system ();

      std::cout << "   Solving..." << std::endl;
      solve ();

      std::cout << "   Writing output..." << std::endl;
      output_results (refinement_cycle);

      std::cout << std::endl;
    }
  }
}



namespace mats_grow
{
  using namespace dealii;

  // -- HELPER CLASS definition
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

  // -- CLASS definition
  template <int dim>
  class DiffusionProblem
  {
  public:
    DiffusionProblem ();
    ~DiffusionProblem ();

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

  // -- CONSTRUCTOR, using Q2 quadratic element, fe (2)
  template <int dim>
  DiffusionProblem<dim>::DiffusionProblem ()
    :
    dof_handler (triangulation),
    fe (2)
  {}

  template <int dim>
  DiffusionProblem<dim>::~DiffusionProblem ()
  {
    dof_handler.clear ();
  }

  template <int dim>
  void DiffusionProblem<dim>::make_grid ()
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
  void DiffusionProblem<dim>::setup_system ()
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
  void DiffusionProblem<dim>::assemble_system ()
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
  void DiffusionProblem<dim>::solve ()
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
  void DiffusionProblem<dim>::refine_grid ()
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
  void DiffusionProblem<dim>::output_results (const unsigned int cycle) const
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
  void DiffusionProblem<dim>::run ()
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
}



// FUNCTION: Display and output mesh info
template<int dim>
void mesh_info(const Triangulation<dim> &tria,
               const std::string        &filename)
{
  using namespace dealii;

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
      if ( cell->material_id() == 'mats' )
        a=5;
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
  std::cout << " mesh visualization saved to: " << filename
            << std::endl
            << std::endl;
}



// -- MAIN --
int main() {
    try {
    	deallog.depth_console (0);



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
