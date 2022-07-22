/*
 * Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <modmesh/spacetime/core.hpp>

namespace spacetime
{

Grid::Grid(real_type xmin, real_type xmax, size_t ncelm, ctor_passkey const &)
    : m_xmin(xmin)
    , m_xmax(xmax)
    , m_ncelm(ncelm)
{
    if (ncelm < 1)
    {
        throw std::invalid_argument(modmesh::Formatter()
                                    << "Grid::Grid(xmin=" << xmin << ", xmax=" << xmax
                                    << ", ncelm=" << ncelm << ") invalid argument: ncelm smaller than 1");
    }
    if (xmin >= xmax)
    {
        throw std::invalid_argument(modmesh::Formatter()
                                    << "Grid::Grid(xmin=" << xmin << ", xmax=" << xmax
                                    << ", ncelm=" << ncelm << ") invalid arguments: xmin >= xmax");
    }
    // Fill the array for CCE boundary.
    const real_type xspace = (xmax - xmin) / ncelm;
    array_type xloc(std::vector<size_t>{ncelm + 1});
    xloc[0] = xmin;
    for (size_t it = 1; it < ncelm; ++it)
    {
        xloc[it] = xloc[it - 1] + xspace;
    }
    xloc[ncelm] = xmax;
    // Initialize.
    init_from_array(xloc);
}

void Grid::init_from_array(array_type const & xloc)
{
    if (xloc.size() < 2)
    {
        throw std::invalid_argument(modmesh::Formatter()
                                    << "Grid::init_from_array(xloc) invalid arguments: "
                                    << "xloc.size()=" << xloc.size() << " smaller than 2");
    }
    for (size_t it = 0; it < xloc.size() - 1; ++it)
    {
        if (xloc[it] >= xloc[it + 1])
        {
            throw std::invalid_argument(modmesh::Formatter()
                                        << "Grid::init_from_array(xloc) invalid arguments: "
                                        << "xloc[" << it << "]=" << xloc[it]
                                        << " >= xloc[" << it + 1 << "]=" << xloc[it + 1]);
        }
    }
    m_ncelm = xloc.size() - 1;
    m_xmin = xloc[0];
    m_xmax = xloc[m_ncelm];
    // Mark the boundary of conservation celms.
    const size_t nx = m_ncelm * 2 + (1 + BOUND_COUNT * 2);
    m_agrid = modmesh::AscendantGrid1d(nx);
    // Fill x-coordinates at CE boundary.
    for (size_t it = 0; it < xloc.size(); ++it)
    {
        m_agrid[it * 2 + BOUND_COUNT] = xloc[it];
    }
    // Fill x-coordinates at CE center.
    for (size_t it = 0; it < m_ncelm; ++it)
    {
        const size_t ref = it * 2 + BOUND_COUNT + 1;
        m_agrid[ref] = (m_agrid[ref - 1] + m_agrid[ref + 1]) / 2;
    }
    // Fill the front and back value.
    for (size_t it = 1; it <= BOUND_COUNT; ++it)
    {
        // Front value.
        {
            constexpr size_t ref = BOUND_COUNT;
            m_agrid[ref - it] = m_agrid[ref] + m_agrid[ref] - m_agrid[ref + it];
        }
        // Back value.
        {
            const size_t ref = nx - BOUND_COUNT - 1;
            m_agrid[ref + it] = m_agrid[ref] + m_agrid[ref] - m_agrid[ref - it];
        }
    }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Field::Field(std::shared_ptr<Grid> const & grid, Field::value_type time_increment, size_t nvar)
    : m_grid(grid)
    , m_so0(array_type(std::vector<size_t>{grid->xsize(), nvar}))
    , m_so1(array_type(std::vector<size_t>{grid->xsize(), nvar}))
    , m_cfl(array_type(std::vector<size_t>{grid->xsize()}))
{
    set_time_increment(time_increment);
}

void Field::set_time_increment(value_type time_increment)
{
    m_time_increment = time_increment;
    m_half_time_increment = 0.5 * time_increment;
    m_quarter_time_increment = 0.25 * time_increment;
}

void Celm::move_at(int_type offset)
{
    const size_t xindex = this->xindex() + offset;
    if (xindex < 2 || xindex >= grid().xsize() - 2)
    {
        throw std::out_of_range(modmesh::Formatter()
                                << "Celm(xindex=" << this->xindex() << ")::move_at(offset=" << offset
                                << "): xindex = " << xindex
                                << " outside the interval [2, " << grid().xsize() - 2 << ")");
    }
    move(offset);
}

void Selm::move_at(int_type offset)
{
    const size_t xindex = this->xindex() + offset;
    if (xindex < 1 || xindex >= grid().xsize() - 1)
    {
        throw std::out_of_range(modmesh::Formatter()
                                << "Selm(xindex=" << this->xindex() << ")::move_at(offset=" << offset
                                << "): xindex = " << xindex
                                << " outside the interval [1, " << grid().xsize() - 1 << ")");
    }
    move(offset);
}

} /* end namespace spacetime */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
