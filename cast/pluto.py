#----------------------------------------------------------------------
#   A dataset wrapper for pluto simulations
#
#   Author  : Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#   Date    : 2018-07-12
#----------------------------------------------------------------------

from cast.dataset import *
from . import units
from . import grid
from .units import Dimension as Dim
import re
import os
import numpy as np
import astropy.units as u
from .pluload import pload


""" Datafiles produced by pluto.
Filenames are given as regex expression to extract from file list."""
datafile_pattern = ".*\.dat"

outputdir_indicators = ["grid.out", "dbl.out"]

scalar_files = []

collections = ["analysis_values.dat"]

# Look for planet related files and extract the planet number
# from the filename using (\d+) group in regex pattern
particle_file_pattern = ['nbody_coordinates.dat', 'nbody_orbital_elements.dat']

known_units = {
    "niter"    : Dim(),
    "M_DISK"   : Dim(M=1),
    "KE_R"     : Dim(M=1, L=2, T=-2),
    "KE_TH"    : Dim(M=1, L=2, T=-2),
    "KE_PHI"   : Dim(M=1, L=2, T=-2),
    "RHO_MIN"  : Dim(M=1, L=-3),
    "RHO_MAX"  : Dim(M=1, L=-3),
    "J_DISK_." : Dim(M=1, L=2, T=-1),
    "F_."      : Dim(M=1, L=1, T=-2),
    "ACC_."    : Dim(L=1, T=-2),
    "rho"      : Dim(M=1, L=-3),
    "prs"      : Dim(M=1, L=-1, T=-2),
    "vx."      : Dim(L=1, T=-1)
}

def find_pluto_log(rootdir):
    try:
        rv = os.path.join(find_dir_containing("pluto.log", rootdir), "pluto.log")
    except FileNotFoundError:
        #rv = None
        raise
    return rv

def extract_pluto_log_units(datadir):
    fpath = find_pluto_log(datadir)
    if fpath is None:
        rv = units.UnitSystem()
    else:
        foundUnits = {}
        with open(fpath, 'r') as f:
            for l in f:
                # match to lines like: [Density]:      4.249e-09 (gr/cm^3), 2.540e+15 (1/cm^3)
                rs = re.match("\s*\[(\w+)\]\:\s*([0-9\.e\-\+]+).*\((.+)\)",l.split(',')[0].strip())
                if rs:
                    rs = rs.groups()
                    name = rs[0]
                    value = rs[1]
                    unitstring = rs[2].replace('^', '').replace('gr','g').replace('sec', 's')
                    foundUnits[rs[0]] = float(value)*u.Unit(unitstring)
        baseunits = {
            "L" : foundUnits['Length'],
            "T" : foundUnits['Time'],
            "Th" : foundUnits['Temperature'],
            "M" : foundUnits['Density']*foundUnits['Length']**3
            }
        rv = units.UnitSystem(baseunits)
    return rv

def parse_text_header(fpath):
    with open(fpath, 'r') as f:
        header = []
        for l in f:
            l = l.strip()
            if l[0] != '#':
                break
            header.append(l)
    return parse_text_header_pluto(header)

def parse_text_header_pluto(header):
    names = []
    for l in header:
        l = l.strip('#').strip()
        if l[:4] == "vars":
            l = l[4:].lstrip().lstrip(":")
            names = [s.strip() for s in l.split('\t')]
    timecol = None
    for n, name in enumerate(names):
        if name in ['time', 'simulation time']:
            timecol = n
    return names, timecol

def pload_to_grid(pl, unitSys = {'L' : 1}):
    if pl.geometry == "SPHERICAL":
        return grid.SphericalRegularGrid(r=pl.x1*unitSys['L'], dr=pl.dx1*unitSys['L'],
                                         theta=pl.x2, dtheta=pl.dx2,
                                         phi=pl.x3, dphi=pl.dx3 )
    elif pl.geometry == "POLAR":
        return grid.PolarRegularGrid(r=pl.x1*unitSys['L'], dr=pl.dx1*unitSys['L'],
                                         phi=pl.x3, dphi=pl.dx3 )
    else:
        raise NotImplementedError("No grid implemented for '{}' geometry.".format(pl.geometry))

def createPlutoParticles(datadir, unitSys):
    # return a list of PlutoParticle objects for all present nbody particles
    particleIds = plutoParticlesIds(datadir)
    particles = []
    for i in particleIds:
        particles.append(Particle(name = i, unitSys=unitSys, data={}))

    # register the common load function
    loadFunction = lambda : loadPlutoParticles(datadir, particles, unitSys)
    for p in particles:
        p.load = loadFunction

    return { p.name : p for p in particles }

def plutoParticlesIds(datadir):
    with open(os.path.join(datadir, 'nbody_coordinates.dat'), 'r') as f:
        # first, find the number of particles
        ids = []
        for l in f:
            if l[0] == "#":
                continue
            pid = l.strip().split()[0]
            if not pid in ids:
                ids.append(pid)
            else:
                break
    return ids

def loadPlutoParticles(datadir, particles, unitSys):
    pids = [p.name for p in particles]
    Nparticles = len(particles)

    # load cartesian positions and velocities
    data = np.genfromtxt(os.path.join(datadir, 'nbody_coordinates.dat'))
    varNames = ['id', 'time', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    units = { 'id'   : 1,
              'time' : unitSys['T'],
              'x'    : unitSys['L'],
              'y'    : unitSys['L'],
              'z'    : unitSys['L'],
              'vx'   : unitSys['L']/unitSys['T'],
              'vy'   : unitSys['L']/unitSys['T'],
              'vz'   : unitSys['L']/unitSys['T'] }

    for n, p in enumerate(particles):
        for k, name in enumerate(varNames):
            if k == 1:
                continue
            p.data[name] = data[n::Nparticles, k]*units[name]

    # load orbital elements
    data = np.genfromtxt(os.path.join(datadir, 'nbody_orbital_elements.dat'))
    varNames = ['id', 'time', 'a', 'e', 'i',
                'AscendingNode', 'Pericenter', 'TrueAnomaly',
                'PeriodInCodeUnits', 'EccentricAnomaly', 'MeanAnomaly']
    units = { 'id'   : 1,
              'time' : unitSys['T'],
              'a'    : unitSys['L'],
              'e'    : 1,
              'i'    : u.rad,
              'AscendingNode'     : u.rad,
              'Pericenter'        : u.rad,
              'TrueAnomaly'       : u.rad,
              'PeriodInCodeUnits' : unitSys['T'],
              'EccentricAnomaly'  : u.rad,
              'MeanAnomaly'       : u.rad }

    # orbital elements are not printed for the primary object
    for n, p in enumerate(particles[1:]):
        for k, name in enumerate(varNames):
            if k == 1:
                continue
            p.data[name] = data[n::Nparticles, k]*units[name]

class ScalarTimeSeries(TimeSeries):
    def __init__(self, time=None, data=None, datafile=None, name = None, unitSys = None):
        self.time = time
        self.data = data
        self.datafile = datafile
        self.name = name
        self.unitSys = unitSys

    def load(self, *args, **kwargs):
        pass

class PlutoField(Field):
    def __init__(self, *args, unitSys=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.unitSys = unitSys

    def load(self):
        if self.data is None:
            if self.resource[1] is None:
                self.resource[1] = self.pload()
            self.data = self.resource[1].__dict__[self.name]
            if self.unitSys is not None and self.name is not None:
                self.data *= self.unitSys.find(self.name)
            if self.unitSys is not None and self.name == 'rho' and self.grid.dim == 2:
                self.data *= self.unitSys['L']

    def pload(self):
        output_number = int(re.match("data\.(\d+)\.dbl", os.path.basename(self.resource[0])).groups()[0])
        return pload(output_number, os.path.dirname(self.resource[0]))


class PlutoDataset(AbstractDataset):
    def __init__(self, rootdir):
        super().__init__()
        self.rootdir = rootdir
        self.find_datadir()
        self.units = extract_pluto_log_units(self.rootdir)
        for key in known_units:
            self.units.register(key, known_units[key])
        self.find_datafiles()
        self.find_scalars()
        self.find_collections()
        self.find_particles()
        self.find_fields()

    def find_datadir(self):
        # Look for an output dir inside the rootdir
        self.datadir = find_dir_containing(outputdir_indicators, self.rootdir)

    def find_datafiles(self):
        """ Search the datadir for datafiles."""
        matches = (re.match(datafile_pattern, s) for s in os.listdir(self.datadir))
        self.datafiles = [m.string for m in matches if m is not None]

    def find_scalars(self):
        """ Find all time info."""
        for s in scalar_files:
            if s in self.datafiles:
                name = s[:-4]
                self.timeSeries[name] = ScalarTimeSeries(os.path.join(self.datadir, s)
                                                         , name=name, unitSys=self.units)

    def find_collections(self):
        """ Find files containing multiple time series and parse them. """
        for c in collections:
            if c in self.datafiles:
                self.add_collection(c)

    def add_collection(self, collection):
        fpath = os.path.join(self.datadir, collection)
        names, timecol = parse_text_header(fpath)
        if timecol is None:
            raise TypeError("Could not find time info for constructing time series in '{}' with names '{}'.".format(fpath, names))
        data = np.genfromtxt(fpath)
        time = data[:,timecol]*self.units['time']
        for n, name in enumerate(names):
            if n == timecol:
                continue
            vardata = data[:,n]*self.units.find(name)
            self.timeSeries[name] = ScalarTimeSeries(time = time, data = vardata, name = name)

    def find_particles(self):
        if all([f in self.datafiles for f in particle_file_pattern]):
            self.particles = createPlutoParticles(self.datadir, self.units)

    def find_fields(self):
        output_numbers = []
        time = []
        with open(os.path.join(self.datadir, 'dbl.out')) as f:
            got_names = False
            for l in f:
                parts = l.strip().split()
                if not got_names:
                    var_names = parts[6:]
                    got_names = True
                output_numbers.append(int(parts[0]))
                time.append(float(parts[1]))

        self.times['coarse'] = Time(data = time*self.units['T'])
        datafiles = [[int(m.groups()[0]), m.string]
                      for m in (re.match("data\.(\d*)\.dbl", s)
                                for s in os.listdir(self.datadir)) if m is not None]
        datafiles = sorted( datafiles, key=lambda item: item[0])
        # Get grid data from first output
        pl = pload(datafiles[0][0], self.datadir)
        self.grids['full'] = pload_to_grid(pl, self.units)
        self._field_resource = [ [f, None] for f in (os.path.join(self.datadir, s) for s in (l[1] for l in datafiles))]
        self._field_resource[0][1] = pl
        for name in var_names:
            self.fields[name] = FieldTimeSeries(name, self._field_resource, self.times['coarse'],
                                                self.grids['full'], unitSys=self.units,
                                                Field = PlutoField)
