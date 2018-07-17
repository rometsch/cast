#----------------------------------------------------------------------
#	A dataset wrapper for fargo-twam simulations
#
#	Author	: Thomas Rometsch (thomas.rometsch@uni-tuebingen.de)
#	Date	: 2018-07-12
#----------------------------------------------------------------------

from .dataset import *
from . import units
from .units import Dimension as Dim
from . import grid
import re
import os
import numpy as np
import astropy.units as u


""" Datafiles produced by fargo-twam.
Filenames are given as regex expression to extract from file list."""
datafile_pattern = ".*\.dat"

outputdir_indicators = ["Quantities.dat", "used_rad.dat", "misc.dat"]

time_files = { "fine": "Quantities.dat",
			   "coarse" : "misc.dat"}

scalar_files = ["mass.dat", "momx.dat", "momy.dat", "momz.dat" ]

collections = ["Quantities.dat", "misc.dat"]

# Look for planet related files and extract the planet number
# from the filename using (\d+) group in regex pattern
particle_file_pattern = ["bigplanet(\d+)\.dat", "orbit(\d+)\.dat", "a._planet_(\d+)\.dat"]

# capture output data (name, number) for fields with filenames like "gasdens7.dat"
field_pattern = "([a-z]*[A-Z]*)([0-9]+)\.dat"

known_units = {
	"physical time"				  : Dim(T=1),
	"mass"						  : Dim(M=1),
	"angular momentum"			  : Dim(M=1, L=2, T=-1),
	"total energy"				  : Dim(M=1, L=2, T=-2),
	"internal energy"			  : Dim(M=1, L=2, T=-2),
	"kinematic energy"			  : Dim(M=1, L=2, T=-2),
	"potential energy"			  : Dim(M=1, L=2, T=-2),
#	"qplus"						  : Dim(),
#	"qminus"					  : Dim(),
#	"pvdiv"						  : Dim(),
	"radial kinetic energy"		  : Dim(M=1, L=2, T=-2),
	"azimuthal kinetic energy"	  : Dim(M=1, L=2, T=-2),
	"timestep"					  : Dim(),
	"OmegaFrame"				  : Dim(T=-1),
	"LostMass"					  : Dim(M=1),
	"FrameAngle"				  : Dim(),
	"gasdens"					  : Dim(M=1, L=-2),
	"gasenergy"					  : Dim(M=1, L=2, T=-2),
	"gasvrad"					  : Dim(L=1, T=-1),
	"gasvtheta"					  : Dim(L=1, T=-1),
	"gastau"					  : Dim(Th=1)
}

default_units_fargo_twam = {
	'L' : 5.2*u.au,		# length
	'T' : 5.2**(3./2)/(2*np.pi)*u.yr,		# time
	'M' : u.solMass,			# mass
	'C' : u.ampere,		# current
	'Th' : u.Kelvin,	# temperature (Theta)
	'Ad' : u.radian,	# angular distance
	'SA' : u.steradian, # solid angle
	'I' : u.candela,	# luminous intensity
	'Mag' : u.mag,		# stellar magnitude
	'A' : u.mole,		# amount of substance
	'N' : u.photon		# photon count
}

def parse_text_header(fpath):
	with open(fpath, 'r') as f:
		header = []
		for l in f:
			l = l.strip()
			if l[0] != '#':
				break
			header.append(l)
	return parse_fargo_twam_header(header)

def parse_fargo_twam_header(header):
	names = []
	l = header[0]
	names = [s.strip() for s in l.lstrip('# ').split('\t') if s.strip() != '']
	timecol = None
	for n, name in enumerate(names):
		if name in ['physical time']:
			timecol = n
	return names, timecol

def centered_coordinates(xInterface):
    x = 0.5*(xInterface[1:] + xInterface[:-1])
    dx = xInterface[1:] - xInterface[:-1]
    return (x, dx)

def parse_FargoTwam_grid(datadir, unitSys=None):
	fpath = os.path.join(datadir, "used_rad.dat")
	rInterface = np.genfromtxt(fpath)

	Nr = len(rInterface)-1
	r, dr = centered_coordinates(rInterface)
	r = r*unitSys['L']
	dr = dr*unitSys['L']

	# Get the number of azimuthal cells by dividing the total file size
	# of the first gas density output by Nr and 8 (=1 byte)
	# to obtain the number azimuthal of azimutal cells.
	Nphi = os.path.getsize(os.path.join(datadir, "gasdens0.dat"))/Nr/8
	if (Nphi%1 != 0):
		raise ValueError("Caclulated number of azimuthal cells is not an integer!")
	else:
		Nphi = int(Nphi)
	phi = np.linspace(0, 2*np.pi, Nphi+1)
	dphi = phi[1:] - phi[:-1]
	phi = 0.5*(phi[1:] + phi[:-1])
	return grid.SphericalRegularGrid(r=r, dr=dr, phi=phi, dphi=dphi)

class FargoTwamField(Field):
	def __init__(self, *args, unitSys=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.unitSys = unitSys

	def load(self):
		self.data = np.fromfile(self.resource).reshape(self.grid.shape).transpose()
		if self.unitSys is not None and self.name is not None:
			self.data *= self.unitSys.find(self.name)
		if self.unitSys is not None and self.name == 'gasdens' and self.grid.dim == 2:
			self.data *= self.unitSys['L']

class ScalarTimeSeries(TimeSeries):
	def __init__(self, time=None, data=None, datafile=None, name = None, unitSys = None):
		self.time = time
		self.data = data
		self.datafile = datafile
		self.name = name
		self.unitSys = unitSys

	def load(self, *args, **kwargs):
		data = np.genfromtxt(self.datafile)
		self.time = data[:,0]

		for n in range(1, data.shape[1]):
			self.data = data[:,1]

		if self.unitSys is not None:
			self.time *= self.unitSys['time']
			if self.name is not None:
				self.data *= self.unitSys.find(self.name)

class FargoTwamParticle(Particle):
	def __init__(self, name, resource=None, unitSys=None):
		super().__init__(name=name, resource = {})
		self.unitSys = unitSys
		if resource:
			self.add_resource(resource)

	def add_resource(self, res):
		fname = os.path.basename(res)
		if re.match("bigplanet\d+\.dat", fname):
			self.resource['planet'] = res
		elif re.match("orbit\d+\.dat", fname):
			self.resource['orbit'] = res
		else:
			rs = re.match("a(\w+)_planet_\d\.dat", fname)
			if rs:
				self.resource["a{}".format(rs.groups()[0])] = res

	def load(self):
		if "planet" in self.resource:
			data = np.genfromtxt(self.resource["planet"])
			names = ["TimeStep", "x1", "x2", "x3", "v1",
					 "v2", "v3", "mass", "time", "OmegaFrame"]
			for k, name in enumerate(names):
				self.data[name] = data[:,k]

			for v in ["x1", "x2", "x3"]:
				self.data[v] = self.data[v]*self.unitSys['L']

			for v in ["v1", "v2", "v3"]:
				print(self.data[v])
				self.data[v] = self.data[v]*self.unitSys['L']*self.unitSys['T']**(-1)

			self.data["mass"] = self.data["mass"]*self.unitSys['M']
			self.data["time"] = self.data["time"]*self.unitSys['T']
			self.data["OmegaFrame"] = self.data["OmegaFrame"]*self.unitSys['T']**(-1)

		if "orbit" in self.resource:
			data = np.genfromtxt(self.resource["orbit"])
			names = ["time", "e", "a", "MeanAnomaly",
					 "TrueAnomaly", "Periastron"]
			for k, name in enumerate(names):
				self.data[name] = data[:,k]

			self.data["time"] = self.data["time"]*self.unitSys['T']
			self.data["a"]	  = self.data["a"]*self.unitSys['L']
			for v in ["MeanAnomaly", "TrueAnomaly"]:
				self.data[v]	= self.data[v]*self.unitSys['T']**(-1)

		for v in ["ax", "ay", "az"]:
			if v in self.resource:
				data = np.genfromtxt(self.resource[v])[:,1]
				self.data[v] = data*self.unitSys['L']*self.unitSys['T']**(-2)

class FieldTimeSeries(TimeSeries):
	def __init__(self, name, resource, time, grid, unitSys=None):
		super().__init__(name, resource)
		self.time = time
		self.data = []
		self.unitSys = unitSys
		for f in self.resource:
			self.data.append(FargoTwamField(name, grid=grid, resource=f, unitSys=unitSys))

	def get(self, varname, n = None ):
		self.load(n)
		return self.__dict__[varname][n]

	def load(self, n):
		self.data[n].load()

class FargoTwamDataset(AbstractDataset):
	def __init__(self, rootdir):
		super().__init__()
		self.rootdir = rootdir
		self.find_datadir()
		self.units = units.parse_code_units_file(self.rootdir, defaults=default_units_fargo_twam)
		for key in known_units:
			self.units.register(key, known_units[key])
		self.find_grids()
		self.find_times()
		self.find_datafiles()
		self.find_scalars()
		self.find_collections()
		self.find_particles()
		self.find_fields()

	def find_datadir(self):
		# Look for an output dir inside the rootdir
		self.datadir = find_dir_containing(outputdir_indicators, self.rootdir)

	def find_grids(self):
		self.grids["full"] = parse_FargoTwam_grid(self.datadir, unitSys=self.units)

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
		for f in self.datafiles:
			for p in particle_file_pattern:
				rs = re.match(p, f)
				if rs:
					fpath = os.path.join(self.datadir, f)
					name = rs.groups()[0]
					if "bigplanet" in f:
						name = "{}".format(int(name) - 1)
					if name not in self.particles:
						self.particles[name] = FargoTwamParticle(name, resource=fpath, unitSys=self.units)
					else:
						self.particles[name].add_resource(fpath)

	def find_times(self):
		for name in time_files:
			fpath = os.path.join(self.datadir, time_files[name])
			if time_files[name] in os.listdir(self.datadir):
				if name == "fine":
					data = np.genfromtxt(fpath)[:,0]*self.units['T']
				elif name == "coarse":
					data = np.genfromtxt(fpath)[:,1]*self.units['T']
				else:
					raise ValueError("Don't know how to parse time info for '{}'".format(time_files[name]))
				self.times[name] = Time(data=data)

	def find_fields(self):
		matches = [m for m in (re.match(field_pattern, f) for f in self.datafiles) if m is not None]
		fields = {}
		for m in matches:
			name, nr = m.groups()
			if name not in fields:
				fields[name] = [m]
			else:
				fields[name].append(m)
		for name in fields:
			fields[name] = sorted(fields[name], key=lambda item: item.groups()[1])
			files = [os.path.join(self.datadir,m.string) for m in fields[name]]
			self.fields[name] = FieldTimeSeries(name, files, self.times['coarse'],
												self.grids['full'], unitSys=self.units)
