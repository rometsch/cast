#------------------------------------------------------------------------------------------
# Author	:	Thomas Rometsch (thomas.rometsch@student.uni-tuebingen.de)
# Date		:	2017-10-25
#------------------------------------------------------------------------------------------
# Library to load PLUTO output files.
# This class is extracted from a port of the python2 script pyPLUTO.py
# (version 4-2.0 which is shipped with PLUTO 4.2.)
# to python3.
#------------------------------------------------------------------------------------------
import os;
import sys;
import array;
import numpy as np;

####### Check for h5py to Read AMR data ######
try:
	import h5py as h5
	hasH5 = True
except ImportError:
	hasH5 = False

def curdir():
	""" Get the current working directory.
	"""
	curdir = os.getcwd()+'/'
	return curdir

def get_nstepstr(ns):
	""" Convert the float input *ns* into a string that would match the data file name.

	**Inputs**:

	 ns -- Integer number that represents the time step number. E.g., The ns for data.0001.dbl is 1.\n

	**Outputs**:

	 Returns the string that would be used to complete the data file name. E.g., for data.0001.dbl, ns = 1 and pyPLUTO.get_nstepstr(1) returns '0001'
	 """
	nstepstr = str(ns)
	while len(nstepstr) < 4:
		   nstepstr= '0'+nstepstr
	return nstepstr

def nlast_info(w_dir=None,datatype=None):
	""" Prints the information of the last step of the simulation as obtained from out files

	**Inputs**:

	  w_dir -- path to the directory which has the dbl.out(or flt.out) and the data\n
	  datatype -- If the data is of 'float' type then datatype = 'float' else by default the datatype is set to 'double'.

		**Outputs**:

	  This function returns a dictionary with following keywords - \n

	  nlast -- The ns for the last file saved.\n
	  time -- The simulation time for the last file saved.\n
	  dt -- The time step dt for the last file. \n
	  Nstep -- The Nstep value for the last file saved.


	**Usage**:

	  In case the data is 'float'.

	  ``wdir = /path/to/data/directory``\n
	  ``import pyPLUTO as pp``\n
	  ``A = pp.nlast_info(w_dir=wdir,datatype='float')``
	"""
	if w_dir is None: w_dir=curdir()
	if datatype == 'float':
		fname_v = w_dir+"flt.out"
	elif datatype == 'vtk':
		fname_v = w_dir+"vtk.out"
	else:
		fname_v = w_dir+"dbl.out"
	with open(fname_v, "r") as f:
		last_line = f.readlines()[-1].split()
	nlast = int(last_line[0])
	SimTime =  float(last_line[1])
	Dt = float(last_line[2])
	Nstep = int(last_line[3])

	print("------------TIME INFORMATION--------------")
	print('nlast =',nlast)
	print('time  =',SimTime)
	print('dt	=', Dt)
	print('Nstep =',Nstep)
	print("-------------------------------------------")

	return {'nlast':nlast,'time':SimTime,'dt':Dt,'Nstep':Nstep}


class pload(object):
	class GeometryError(Exception): pass;
	class DimensionalityError(Exception): pass;

	def __init__(self, ns, w_dir=None, datatype=None, level = 0, x1range=None, x2range=None, x3range=None):
		"""Loads the data.

		**Inputs**:

		  ns -- Step Number of the data file\n
	  w_dir -- path to the directory which has the data files\n
		  datatype -- Datatype (default = 'double')

		**Outputs**:

		  pyPLUTO pload object whose keys are arrays of data values.

	"""
		self.NStep = ns
		self.Dt = 0.0

		self.n1 = 0
		self.n2 = 0
		self.n3 = 0

		self.x1 = []
		self.x2 = []
		self.x3 = []
		self.dx1 = []
		self.dx2 = []
		self.dx3 = []

		self.x1range = x1range
		self.x2range = x2range
		self.x3range = x3range

		self.NStepStr = str(self.NStep)
		while len(self.NStepStr) < 4:
			self.NStepStr = '0'+self.NStepStr

		if datatype is None:
			datatype = "double"
		self.datatype = datatype

		if ((not hasH5) and (datatype == 'hdf5')):
			print('To read AMR hdf5 files with python')
			print('Please install h5py (Python HDF5 Reader)')
			return

		self.level = level

		if w_dir is None:
			w_dir = os.getcwd() + '/'
		self.wdir = w_dir

		Data_dictionary = self.ReadDataFile(self.NStepStr)
		for keys in Data_dictionary:
			object.__setattr__(self, keys, Data_dictionary.get(keys))

	def ReadTimeInfo(self, timefile):
		""" Read time info from the outfiles.

	**Inputs**:

	  timefile -- name of the out file which has timing information.

	"""

		if (self.datatype == 'hdf5'):
			fh5 = h5.File(timefile,'r')
			self.SimTime = fh5.attrs.get('time')
			#self.Dt = 1.e-2 # Should be erased later given the level in AMR
			fh5.close()
		else:
			ns = self.NStep
			f_var = open(timefile, "r")
			tlist = []
			for line in f_var.readlines():
				tlist.append(line.split())
			self.SimTime = float(tlist[ns][1])
			self.Dt = float(tlist[ns][2])

	def ReadVarFile(self, varfile):
		""" Read variable names from the outfiles.

	**Inputs**:

	  varfile -- name of the out file which has variable information.

	"""
		if (self.datatype == 'hdf5'):
			fh5 = h5.File(varfile,'r')
			self.filetype = 'single_file'
			self.endianess = '>' # not used with AMR, kept for consistency
			self.vars = []
			for iv in range(fh5.attrs.get('num_components')):
				self.vars.append(fh5.attrs.get('component_'+str(iv)))
			fh5.close()
		else:
			vfp = open(varfile, "r")
			varinfo = vfp.readline().split()
			self.filetype = varinfo[4]
			self.endianess = varinfo[5]
			self.vars = varinfo[6:]
			vfp.close()

	def ReadGridFile(self, gridfile):
		""" Read grid values from the grid.out file.

	**Inputs**:

	  gridfile -- name of the grid.out file which has information about the grid.

	"""
		xL = []
		xR = []
		nmax = []
		gfp = open(gridfile, "r")
		for i in gfp.readlines():
			if len(i.split()) == 1:
				try:
					int(i.split()[0])
					nmax.append(int(i.split()[0]))
				except:
					pass

			if len(i.split()) == 3:
				try:
					int(i.split()[0])
					xL.append(float(i.split()[1]))
					xR.append(float(i.split()[2]))
				except:
					if (i.split()[1] == 'GEOMETRY:'):
						self.geometry=i.split()[2]
					pass

		self.n1, self.n2, self.n3 = nmax
		n1 = self.n1
		n1p2 = self.n1 + self.n2
		n1p2p3 = self.n1 + self.n2 + self.n3
		self.x1 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1)])
		self.dx1 = np.asarray([(xR[i]-xL[i]) for i in range(n1)])
		self.x2 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1, n1p2)])
		self.dx2 = np.asarray([(xR[i]-xL[i]) for i in range(n1, n1p2)])
		self.x3 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1p2, n1p2p3)])
		self.dx3 = np.asarray([(xR[i]-xL[i]) for i in range(n1p2, n1p2p3)])


		# Stores the total number of points in '_tot' variable in case only
		# a portion of the domain is loaded. Redefine the x and dx arrays
		# to match the requested ranges
		self.n1_tot = self.n1 ; self.n2_tot = self.n2 ; self.n3_tot = self.n3
		if (self.x1range != None):
			self.n1_tot = self.n1
			self.irange = list(range(abs(self.x1-self.x1range[0]).argmin(),abs(self.x1-self.x1range[1]).argmin()+1))
			self.n1  = len(self.irange)
			self.x1  = self.x1[self.irange]
			self.dx1 = self.dx1[self.irange]
		else:
			self.irange = list(range(self.n1))
		if (self.x2range != None):
			self.n2_tot = self.n2
			self.jrange = list(range(abs(self.x2-self.x2range[0]).argmin(),abs(self.x2-self.x2range[1]).argmin()+1))
			self.n2  = len(self.jrange)
			self.x2  = self.x2[self.jrange]
			self.dx2 = self.dx2[self.jrange]
		else:
			self.jrange = list(range(self.n2))
		if (self.x3range != None):
			self.n3_tot = self.n3
			self.krange = list(range(abs(self.x3-self.x3range[0]).argmin(),abs(self.x3-self.x3range[1]).argmin()+1))
			self.n3  = len(self.krange)
			self.x3  = self.x3[self.krange]
			self.dx3 = self.dx3[self.krange]
		else:
			self.krange = list(range(self.n3))
		self.Slice=(self.x1range != None) or (self.x2range != None) or (self.x3range != None)


		# Create the xr arrays containing the edges positions
		# Useful for pcolormesh which should use those
		self.x1r = np.zeros(len(self.x1)+1) ; self.x1r[1:] = self.x1 + self.dx1/2.0 ; self.x1r[0] = self.x1r[1]-self.dx1[0]
		self.x2r = np.zeros(len(self.x2)+1) ; self.x2r[1:] = self.x2 + self.dx2/2.0 ; self.x2r[0] = self.x2r[1]-self.dx2[0]
		self.x3r = np.zeros(len(self.x3)+1) ; self.x3r[1:] = self.x3 + self.dx3/2.0 ; self.x3r[0] = self.x3r[1]-self.dx3[0]


		prodn = self.n1*self.n2*self.n3
		if prodn == self.n1:
			self.nshp = (self.n1)
		elif prodn == self.n1*self.n2:
			self.nshp = (self.n2, self.n1)
		else:
			self.nshp = (self.n3, self.n2, self.n1)


	def DataScanVTK(self, fp, n1, n2, n3, endian, dtype):
		""" Scans the VTK data files.

		**Inputs**:

	 fp -- Data file pointer\n
	 n1 -- No. of points in X1 direction\n
	 n2 -- No. of points in X2 direction\n
	 n3 -- No. of points in X3 direction\n
	 endian -- Endianess of the data\n
	 dtype -- datatype

		**Output**:

		  Dictionary consisting of variable names as keys and its values.

	"""
		ks = []
		vtkvar = []
		while True:
			l = fp.readline()
			try:
				l.split()[0]
			except IndexError:
				pass
			else:
				if l.split()[0] == 'SCALARS':
					ks.append(l.split()[1])
				elif l.split()[0] == 'LOOKUP_TABLE':
					A = array.array(dtype)
					fmt = endian+str(n1*n2*n3)+dtype
					nb = np.dtype(fmt).itemsize
					A.fromstring(fp.read(nb))
					if (self.Slice):
						darr = np.zeros((n1*n2*n3))
						indxx = np.sort([n3_tot*n2_tot*k + j*n2_tot + i for i in self.irange for j in self.jrange for k in self.krange])
						if (sys.byteorder != self.endianess):
							A.byteswap()
						for ii,iii in enumerate(indxx):
							darr[ii] = A[iii]
						vtkvar_buf = [darr]
					else:
						vtkvar_buf = np.frombuffer(A,dtype=np.dtype(fmt))
					vtkvar.append(np.reshape(vtkvar_buf,self.nshp).transpose())
				else:
					pass
			if l == '':
				break

		vtkvardict = dict(list(zip(ks,vtkvar)))
		return vtkvardict

	def DataScanHDF5(self, fp, myvars, ilev):
		""" Scans the Chombo HDF5 data files for AMR in PLUTO.

		**Inputs**:

		  fp	 -- Data file pointer\n
		  myvars -- Names of the variables to read\n
		  ilev   -- required AMR level

		**Output**:

		  Dictionary consisting of variable names as keys and its values.

		**Note**:

		  Due to the particularity of AMR, the grid arrays loaded in ReadGridFile are overwritten here.

		"""
		# Read the grid information
		dim = fp['Chombo_global'].attrs.get('SpaceDim')
		nlev = fp.attrs.get('num_levels')
		il = min(nlev-1,ilev)
		lev  = []
		for i in range(nlev):
			lev.append('level_'+str(i))
		freb = np.zeros(nlev,dtype='int')
		for i in range(il+1)[::-1]:
			fl = fp[lev[i]]
			if (i == il):
				pdom = fl.attrs.get('prob_domain')
				dx = fl.attrs.get('dx')
				dt = fl.attrs.get('dt')
				ystr = 1. ; zstr = 1. ; logr = 0
				try:
					geom = fl.attrs.get('geometry')
					logr = fl.attrs.get('logr')
					if (dim == 2):
						ystr = fl.attrs.get('g_x2stretch')
					elif (dim == 3):
						zstr = fl.attrs.get('g_x3stretch')
				except:
					print('Old HDF5 file, not reading stretch and logr factors')
				freb[i] = 1
				x1b = fl.attrs.get('domBeg1')
				if (dim == 1):
					x2b = 0
				else:
					x2b = fl.attrs.get('domBeg2')
				if (dim == 1 or dim == 2):
					x3b = 0
				else:
					x3b = fl.attrs.get('domBeg3')
				jbeg = 0 ; jend = 0 ; ny = 1
				kbeg = 0 ; kend = 0 ; nz = 1
				if (dim == 1):
					ibeg = pdom[0] ; iend = pdom[1] ; nx = iend-ibeg+1
				elif (dim == 2):
					ibeg = pdom[0] ; iend = pdom[2] ; nx = iend-ibeg+1
					jbeg = pdom[1] ; jend = pdom[3] ; ny = jend-jbeg+1
				elif (dim == 3):
					ibeg = pdom[0] ; iend = pdom[3] ; nx = iend-ibeg+1
					jbeg = pdom[1] ; jend = pdom[4] ; ny = jend-jbeg+1
					kbeg = pdom[2] ; kend = pdom[5] ; nz = kend-kbeg+1
			else:
				rat = fl.attrs.get('ref_ratio')
				freb[i] = rat*freb[i+1]

		dx0 = dx*freb[0]

		## Allow to load only a portion of the domain
		if (self.x1range != None):
			if logr == 0:
				self.x1range = self.x1range-x1b
			else:
				self.x1range = [log(self.x1range[0]/x1b),log(self.x1range[1]/x1b)]
			ibeg0 = min(self.x1range)/dx0 ; iend0 = max(self.x1range)/dx0
			ibeg  = max([ibeg, int(ibeg0*freb[0])]) ; iend = min([iend,int(iend0*freb[0]-1)])
			nx = iend-ibeg+1
		if (self.x2range != None):
			self.x2range = (self.x2range-x2b)/ystr
			jbeg0 = min(self.x2range)/dx0 ; jend0 = max(self.x2range)/dx0
			jbeg  = max([jbeg, int(jbeg0*freb[0])]) ; jend = min([jend,int(jend0*freb[0]-1)])
			ny = jend-jbeg+1
		if (self.x3range != None):
			self.x3range = (self.x3range-x3b)/zstr
			kbeg0 = min(self.x3range)/dx0 ; kend0 = max(self.x3range)/dx0
			kbeg  = max([kbeg, int(kbeg0*freb[0])]) ; kend = min([kend,int(kend0*freb[0]-1)])
			nz = kend-kbeg+1

		## Create uniform grids at the required level
		if logr == 0:
			x1 = x1b + (ibeg+np.array(list(range(nx)))+0.5)*dx
		else:
			x1 = x1b*(exp((ibeg+np.array(list(range(nx)))+1)*dx)+exp((ibeg+np.array(list(range(nx))))*dx))*0.5

		x2 = x2b + (jbeg+np.array(list(range(ny)))+0.5)*dx*ystr
		x3 = x3b + (kbeg+np.array(list(range(nz)))+0.5)*dx*zstr
		if logr == 0:
			dx1 = np.ones(nx)*dx
		else:
			dx1 = x1b*(exp((ibeg+np.array(list(range(nx)))+1)*dx)-exp((ibeg+np.array(list(range(nx))))*dx))
		dx2 = np.ones(ny)*dx*ystr
		dx3 = np.ones(nz)*dx*zstr

		# Create the xr arrays containing the edges positions
		# Useful for pcolormesh which should use those
		x1r = np.zeros(len(x1)+1) ; x1r[1:] = x1 + dx1/2.0 ; x1r[0] = x1r[1]-dx1[0]
		x2r = np.zeros(len(x2)+1) ; x2r[1:] = x2 + dx2/2.0 ; x2r[0] = x2r[1]-dx2[0]
		x3r = np.zeros(len(x3)+1) ; x3r[1:] = x3 + dx3/2.0 ; x3r[0] = x3r[1]-dx3[0]
		NewGridDict = dict([('n1',nx),('n2',ny),('n3',nz),\
							('x1',x1),('x2',x2),('x3',x3),\
							('x1r',x1r),('x2r',x2r),('x3r',x3r),\
							('dx1',dx1),('dx2',dx2),('dx3',dx3),\
							('Dt',dt)])

		# Variables table
		nvar = len(myvars)
		vars = np.zeros((nx,ny,nz,nvar))

		LevelDic = {'nbox':0,'ibeg':ibeg,'iend':iend,'jbeg':jbeg,'jend':jend,'kbeg':kbeg,'kend':kend}
		AMRLevel = []
		AMRBoxes = np.zeros((nx,ny,nz))
		for i in range(il+1):
			AMRLevel.append(LevelDic.copy())
			fl = fp[lev[i]]
			data = fl['data:datatype=0']
			boxes = fl['boxes']
			nbox = len(boxes['lo_i'])
			AMRLevel[i]['nbox'] = nbox
			ncount = 0
			AMRLevel[i]['box']=[]
			for j in range(nbox): # loop on all boxes of a given level
				AMRLevel[i]['box'].append({'x0':0.,'x1':0.,'ib':0,'ie':0,\
										   'y0':0.,'y1':0.,'jb':0,'je':0,\
										   'z0':0.,'z1':0.,'kb':0,'ke':0})
				# Box indexes
				ib = boxes[j]['lo_i'] ; ie = boxes[j]['hi_i'] ; nbx = ie-ib+1
				jb = 0 ; je = 0 ; nby = 1
				kb = 0 ; ke = 0 ; nbz = 1
				if (dim > 1):
					jb = boxes[j]['lo_j'] ; je = boxes[j]['hi_j'] ; nby = je-jb+1
				if (dim > 2):
					kb = boxes[j]['lo_k'] ; ke = boxes[j]['hi_k'] ; nbz = ke-kb+1
				szb = nbx*nby*nbz*nvar
				# Rescale to current level
				kb = kb*freb[i] ; ke = (ke+1)*freb[i] - 1
				jb = jb*freb[i] ; je = (je+1)*freb[i] - 1
				ib = ib*freb[i] ; ie = (ie+1)*freb[i] - 1

				# Skip boxes lying outside ranges
				if ((ib > iend) or (ie < ibeg) or \
					(jb > jend) or (je < jbeg) or \
					(kb > kend) or (ke < kbeg)):
					ncount = ncount + szb
				else:

					### Read data
					q = data[ncount:ncount+szb].reshape((nvar,nbz,nby,nbx)).T

					### Find boxes intersections with current domain ranges
					ib0 = max([ibeg,ib]) ; ie0 = min([iend,ie])
					jb0 = max([jbeg,jb]) ; je0 = min([jend,je])
					kb0 = max([kbeg,kb]) ; ke0 = min([kend,ke])

					### Store box corners in the AMRLevel structure
					if logr == 0:
						AMRLevel[i]['box'][j]['x0'] = x1b + dx*(ib0)
						AMRLevel[i]['box'][j]['x1'] = x1b + dx*(ie0+1)
					else:
						AMRLevel[i]['box'][j]['x0'] = x1b*exp(dx*(ib0))
						AMRLevel[i]['box'][j]['x1'] = x1b*exp(dx*(ie0+1))
					AMRLevel[i]['box'][j]['y0'] = x2b + dx*(jb0)*ystr
					AMRLevel[i]['box'][j]['y1'] = x2b + dx*(je0+1)*ystr
					AMRLevel[i]['box'][j]['z0'] = x3b + dx*(kb0)*zstr
					AMRLevel[i]['box'][j]['z1'] = x3b + dx*(ke0+1)*zstr
					AMRLevel[i]['box'][j]['ib'] = ib0 ; AMRLevel[i]['box'][j]['ie'] = ie0
					AMRLevel[i]['box'][j]['jb'] = jb0 ; AMRLevel[i]['box'][j]['je'] = je0
					AMRLevel[i]['box'][j]['kb'] = kb0 ; AMRLevel[i]['box'][j]['ke'] = ke0
					AMRBoxes[ib0-ibeg:ie0-ibeg+1, jb0-jbeg:je0-jbeg+1, kb0-kbeg:ke0-kbeg+1] = il

					### Extract the box intersection from data stored in q
					cib0 = (ib0-ib)/freb[i] ; cie0 = (ie0-ib)/freb[i]
					cjb0 = (jb0-jb)/freb[i] ; cje0 = (je0-jb)/freb[i]
					ckb0 = (kb0-kb)/freb[i] ; cke0 = (ke0-kb)/freb[i]
					q1 = np.zeros((cie0-cib0+1, cje0-cjb0+1, cke0-ckb0+1,nvar))
					q1 = q[cib0:cie0+1,cjb0:cje0+1,ckb0:cke0+1,:]

					# Remap the extracted portion
					if (dim == 1):
						new_shape = (ie0-ib0+1,1)
					elif (dim == 2):
						new_shape = (ie0-ib0+1,je0-jb0+1)
					else:
						new_shape = (ie0-ib0+1,je0-jb0+1,ke0-kb0+1)

					stmp = list(new_shape)
					while stmp.count(1) > 0:
						stmp.remove(1)
					new_shape = tuple(stmp)

					myT = Tools()
					for iv in range(nvar):
						vars[ib0-ibeg:ie0-ibeg+1,jb0-jbeg:je0-jbeg+1,kb0-kbeg:ke0-kbeg+1,iv] = \
							myT.congrid(q1[:,:,:,iv].squeeze(),new_shape,method='linear',minusone=True).reshape((ie0-ib0+1,je0-jb0+1,ke0-kb0+1))
					ncount = ncount+szb

		h5vardict={}
		for iv in range(nvar):
			h5vardict[myvars[iv]] = vars[:,:,:,iv].squeeze()
		AMRdict = dict([('AMRBoxes',AMRBoxes),('AMRLevel',AMRLevel)])
		OutDict = dict(NewGridDict)
		OutDict.update(AMRdict)
		OutDict.update(h5vardict)
		return OutDict


	def DataScan(self, fp, n1, n2, n3, endian, dtype, off=None):
		""" Scans the data files in all formats.

			**Inputs**:

			  fp -- Data file pointer\n
			  n1 -- No. of points in X1 direction\n
			  n2 -- No. of points in X2 direction\n
			  n3 -- No. of points in X3 direction\n
			  endian -- Endianess of the data\n
			  dtype -- datatype, eg : double, float, vtk, hdf5\n
			  off -- offset (for avoiding staggered B fields)

			**Output**:

			  Dictionary consisting of variable names as keys and its values.

		"""
		if off is not None:
			off_fmt = endian+str(off)+dtype
			nboff = np.dtype(off_fmt).itemsize
			fp.read(nboff)

		n1_tot = self.n1_tot ; n2_tot = self.n2_tot; n3_tot = self.n3_tot

		A = array.array(dtype)
		fmt = endian+str(n1_tot*n2_tot*n3_tot)+dtype
		nb = np.dtype(fmt).itemsize
		A.fromstring(fp.read(nb))

		if (self.Slice):
			darr = np.zeros((n1*n2*n3))
			indxx = np.sort([n3_tot*n2_tot*k + j*n2_tot + i for i in self.irange for j in self.jrange for k in self.krange])
			if (sys.byteorder != self.endianess):
				A.byteswap()
			for ii,iii in enumerate(indxx):
				darr[ii] = A[iii]
			darr = [darr]
		else:
			darr = np.frombuffer(A,dtype=np.dtype(fmt))

		return np.reshape(darr[0],self.nshp).transpose()

	def ReadSingleFile(self, datafilename, myvars, n1, n2, n3, endian,
					   dtype, ddict):
		"""Reads a single data file, data.****.dtype.

		**Inputs**:

		  datafilename -- Data file name\n
	  myvars -- List of variable names to be read\n
		  n1 -- No. of points in X1 direction\n
		  n2 -- No. of points in X2 direction\n
		  n3 -- No. of points in X3 direction\n
		  endian -- Endianess of the data\n
		  dtype -- datatype\n
		  ddict -- Dictionary containing Grid and Time Information
		  which is updated

		**Output**:

		  Updated Dictionary consisting of variable names as keys and its values.
	"""
		if self.datatype == 'hdf5':
			fp = h5.File(datafilename,'r')
		else:
			fp = open(datafilename, "rb")

		print("Reading Data file : %s"%datafilename)

		if self.datatype == 'vtk':
			vtkd = self.DataScanVTK(fp, n1, n2, n3, endian, dtype)
			ddict.update(vtkd)
		elif self.datatype == 'hdf5':
			h5d = self.DataScanHDF5(fp,myvars,self.level)
			ddict.update(h5d)
		else:
			for i in range(len(myvars)):
				if myvars[i] == 'bx1s':
					ddict.update({myvars[i]: self.DataScan(fp, n1, n2, n3, endian,
														   dtype, off=n2*n3)})
				elif myvars[i] == 'bx2s':
					ddict.update({myvars[i]: self.DataScan(fp, n1, n2, n3, endian,
														   dtype, off=n3*n1)})
				elif myvars[i] == 'bx3s':
					ddict.update({myvars[i]: self.DataScan(fp, n1, n2, n3, endian,
														   dtype, off=n1*n2)})
				else:
					ddict.update({myvars[i]: self.DataScan(fp, n1, n2, n3, endian,
														   dtype)})


		fp.close()

	def ReadMultipleFiles(self, nstr, dataext, myvars, n1, n2, n3, endian,
						  dtype, ddict):
		"""Reads a  multiple data files, varname.****.dataext.

		**Inputs**:

		  nstr -- File number in form of a string\n
	  dataext -- Data type of the file, e.g., 'dbl', 'flt' or 'vtk' \n
		  myvars -- List of variable names to be read\n
		  n1 -- No. of points in X1 direction\n
		  n2 -- No. of points in X2 direction\n
		  n3 -- No. of points in X3 direction\n
		  endian -- Endianess of the data\n
		  dtype -- datatype\n
		  ddict -- Dictionary containing Grid and Time Information
		  which is updated.

		**Output**:

		  Updated Dictionary consisting of variable names as keys and its values.

	"""
		for i in range(len(myvars)):
			datafilename = self.wdir+myvars[i]+"."+nstr+dataext
			fp = open(datafilename, "rb")
			if self.datatype == 'vtk':
				ddict.update(self.DataScanVTK(fp, n1, n2, n3, endian, dtype))
			else:
				ddict.update({myvars[i]: self.DataScan(fp, n1, n2, n3, endian,
													   dtype)})
			fp.close()

	def ReadDataFile(self, num):
		"""Reads the data file generated from PLUTO code.

	**Inputs**:

	  num -- Data file number in form of an Integer.

		**Outputs**:

	  Dictionary that contains all information about Grid, Time and
	  variables.

	"""
		gridfile = os.path.join(self.wdir,"grid.out");
		if self.datatype == "float":
			dtype = "f"
			varfile = os.path.join(self.wdir,"flt.out");
			dataext = ".flt"
		elif self.datatype == "vtk":
			dtype = "f"
			varfile = os.path.join(self.wdir,"vtk.out");
			dataext=".vtk"
		elif self.datatype == 'hdf5':
			dtype = 'd'
			dataext = '.hdf5'
			nstr = num
			varfile = os.path.join(self.wdir,"data."+nstr+dataext);
		else:
			dtype = "d"
			varfile = os.path.join(self.wdir,"dbl.out");
			dataext = ".dbl"

		self.ReadVarFile(varfile)
		self.ReadGridFile(gridfile)
		self.ReadTimeInfo(varfile)
		nstr = num
		if self.endianess == 'big':
			endian = ">"
		elif self.datatype == 'vtk':
			endian = ">"
		else:
			endian = "<"

		D = [('NStep', self.NStep), ('SimTime', self.SimTime), ('Dt', self.Dt),
			 ('n1', self.n1), ('n2', self.n2), ('n3', self.n3),
			 ('x1', self.x1), ('x2', self.x2), ('x3', self.x3),
			 ('dx1', self.dx1), ('dx2', self.dx2), ('dx3', self.dx3),
			 ('endianess', self.endianess), ('datatype', self.datatype),
			 ('filetype', self.filetype)]
		ddict = dict(D)

		if self.filetype == "single_file":
			datafilename = os.path.join(self.wdir,"data."+nstr+dataext);
			self.ReadSingleFile(datafilename, self.vars, self.n1, self.n2,
								self.n3, endian, dtype, ddict)
		elif self.filetype == "multiple_files":
			self.ReadMultipleFiles(nstr, dataext, self.vars, self.n1, self.n2,
								   self.n3, endian, dtype, ddict)
		else:
			print("Wrong file type : CHECK pluto.ini for file type.")
			print("Only supported are .dbl, .flt, .vtk, .hdf5")
			sys.exit()

		return ddict

	def get_X_cells(self, axis):
		"""	Returns a three dimensional numpy array
			of the coordinate of dimension **axis**
			for every cell.
			(just like variables like density).

		**Input**:

			axis: 	select the coordinate direction
					choices are 1,2,3

		**Output**:

		  	Coordinate array in the shape of the grid (N1, N2, N3);

		**Note**:

			Only for 3-dim!

		"""
		if self.n1 != 1: # Verify that dimensionality is 3.
			if axis == 1:
				try:
					return self.X1;
				except AttributeError:
					x1 = np.array(self.x1);
					X1 = np.repeat( np.expand_dims(x1, 1), self.n2, axis=1);
					X1 = np.repeat( np.expand_dims(X1, 2), self.n3, axis=2);
					self.X1 = X1;
					return X1;

			elif axis == 2:
				try:
					return self.X2;
				except AttributeError:
					x2 = np.array(self.x2);
					X2 = np.repeat( np.expand_dims(x2, 1).transpose(), self.n1, axis=0);
					X2 = np.repeat( np.expand_dims(X2, 2), self.n3, axis=2);
					self.X2 = X2;
					return X2;

			elif axis == 3:
				try:
					return self.X3;
				except AttributeError:
					x3 = np.array(self.x3);
					X3 = np.repeat( np.expand_dims(x3, 1).transpose(), self.n2, axis=0);
					X3 = np.repeat( np.expand_dims(X3, 0), self.n1, axis=0);
					self.X3 = X3;
					return X3;
			else:
				raise ValueError("Selected axis {} is not one of [1,2,3]!".format(axis));
		else:
			raise DimensionalityError("This method only supports 3D grids");

	def get_dV(self):
		"""	Returns a three dimensional numpy array
			of the cell volumes.

		**Output**:

		  	Cell volume array of the shape (N1, N2, N3);

		**Note**:

			Only for 3-dim and only for spherical coordinates.

		"""
		try:
			return self.dV;
		except AttributeError:
			# Coordinates.
			x1 = np.array(self.x1);
			x2 = np.array(self.x2);
			x3 = np.array(self.x3);
			# Cell sizes.
			dx1 = np.array(self.dx1);
			dx2 = np.array(self.dx2);
			dx3 = np.array(self.dx3);
			# Calculate cell volume.
			if self.geometry == "SPHERICAL":
				dV1 = ( (x1+dx1/2)**3 - (x1-dx1/2)**3 )/3;
				dV2 = np.cos( x2 - dx2/2 ) - np.cos( x2 + dx2/2 );
				dV3 = dx3;
				dV = np.repeat( np.expand_dims(dV1,1), self.n2,	axis=1)*dV2;
				dV = np.repeat( np.expand_dims(dV, 2), self.n3,	axis=2)*dV3;
				self.dV = dV;
			else:
				raise type(self).GeometryError("Geometry {} not supported".format(self.geometry));
			return dV;

	def get_dm(self):
		"""	Returns a numpy array
			of the cell masses.

		**Output**:

		  	Cell mass array of the shape of the grid (N1, N2, N3);

		"""
		try:
			return self.dm;
		except AttributeError:
			self.dm = self.get_dV()*self.rho;
			return self.dm;

	def get_mass(self):
		"""	Returns the mass contained inside the computational domain.

		**Output**:

		  	Scalar value of the cell mass.

		"""
		try:
			return self.mass_total;
		except AttributeError:
			self.mass_total = np.sum(self.get_dm());
			return self.mass_total;

	def get_J_cells(self, axis):
		"""	Returns the components of the cartesian angular momentum vector
			with respect to the coordinate system's origin
			in coordinate direction axis (1,2,3) for each cell.

		**Input**:

			axis:	Components of the angular momentum vector.
					1 -> x, 2 -> y, 3 -> z

		**Output**:

		  	Array in the shape of the grid (N1, N2, N3).

		**Note**:

			Only for 3-dim and only for data in spherical coordinates.
			The return angular momenta components are in cartesian coordinates!

		"""
		if not self.n3 != 1:
			raise type(self).DimensionalityError("This method only supports 3D data");
		if self.geometry == "SPHERICAL":
			Rs = self.get_X_cells(1);
			Ths = self.get_X_cells(2);
			Phis = self.get_X_cells(3);
			vth = np.array(self.vx2);
			vphi = np.array(self.vx3);
			dm = self.get_dm();
			# Calculate the anuglar momentum
			if axis == 1:
				self.jx = dm*Rs*( -vth*np.sin(Phis) - vphi*np.cos(Ths)*np.cos(Phis) );
				return self.jx;
			elif axis == 2:
				self.jy = dm*Rs*( vth*np.cos(Phis)  - vphi*np.cos(Ths)*np.sin(Phis) );
				return self.jy;
			elif axis == 3:
				self.jz = dm*Rs*( vphi*np.sin(Ths) );
				return self.jz;
			else:
				raise ValueError("Selected axis {} is not one of [1,2,3]!".format(axis));
		else:
			raise type(self).GeometryError("Geometry {} is not supported in this method!".format(self.geometry));



	def get_J_total(self, axis):
		"""	Returns the components or the norm of the cartesian angular momentum vector
			of the whole simulation domain.

		**Input**:

			axis:	Desired component of the angular momentum vector.
					0 -> length of the vector
					1 -> x, 2 -> y, 3 -> z

		**Output**:

		  	Scalar length or component.

		**Note**:

			The return angular momenta components are in cartesian coordinates!

		"""
		if axis == 0:
			try:
				return self.j_total_norm;
			except AttributeError:
				jx = self.get_J_total(1);
				jy = self.get_J_total(2);
				jz = self.get_J_total(3);
				self.j_total_norm = np.linalg.norm(np.array([jx,jy,jz]));
				return self.j_total_norm;
		elif axis == 1:
			try:
				return self.j_total_x;
			except AttributeError:
				self.j_total_x = np.sum(self.get_J_cells(1));
				return self.j_total_x;
		elif axis == 2:
			try:
				return self.j_total_y;
			except AttributeError:
				self.j_total_y = np.sum(self.get_J_cells(2));
				return self.j_total_y;
		elif axis == 3:
			try:
				return self.j_total_z;
			except AttributeError:
				self.j_total_z = np.sum(self.get_J_cells(3));
				return self.j_total_z;
		else:
			raise ValueError("Selected axis {} is not one of [1,2,3]!".format(axis));


	def get_J_partial(self, axis, axis_nosum):
		"""	Returns the components or the norm of the cartesian angular momentum vector
			of sclices of the simulation domain.
			This function is similar to get_J_total, but instead the values are summed
			only over two instead of three dimensions.
 			In a spherical coordinate system, this allows to return the sum over
			each ring of cells with the same radius.

		**Input**:

			axis:
					Components of the angular momentum vector.
					0 -> length of the vector
					1 -> x, 2 -> y, 3 -> z

			axis_nosum:
					Number of dimension, along wich the values are not summed.

		**Output**:

		  	Numpy array of length N{axis_nosum} with values summed over the remaining
			two dimensions.

		**Note**:

			The return angular momenta components are in cartesian coordinates!

		"""
		# Bookkeeping to avoid repetitions.
		if not hasattr(self,"j_partial_axis_nosum"):
			self.j_partial_axis_nosum = None;

		if axis_nosum != self.j_partial_axis_nosum:
			self.j_partial_axis_nosum = axis_nosum;
			self.j_partial_calculated = np.zeros(4, dtype=bool);
			Ns = [self.n1, self.n2, self.n3];
			self.j_partial = np.zeros([3, Ns[axis_nosum-1]]);


		if axis == 0:
			if self.j_partial_calculated[0]:
				return self.j_total_norm;
			else:
				jx = self.get_J_partial(1, axis_nosum);
				jy = self.get_J_partial(2, axis_nosum);
				jz = self.get_J_partial(3, axis_nosum);
				self.j_partial_norm = np.linalg.norm(np.array([jx,jy,jz]), axis=0);
				self.j_partial_calculated[0] = True;
				return self.j_partial_norm;
		elif axis in [1,2,3]:
			if self.j_partial_calculated[axis]:
				return self.j_partial[axis-1];
			else:
				# Save the remaining axis numbers into a tuple and use it to sum
				# these dimension using np.sum
				axis_to_sum = tuple( [n-1 for n in [1,2,3] if n!=axis_nosum] );
				self.j_partial[axis-1] = np.sum(self.get_J_cells(axis), axis=axis_to_sum);
				self.j_partial_calculated[axis] = True;
				return self.j_partial[axis-1];
		else:
			raise ValueError("Selected axis {} is not one of [0,1,2,3]!".format(axis));



	def get_E_kin(self, axis):
		""" Return the kinetic energy in coordinate direction **axis** or the total value.

		**Input**:

			axis:	coordinate direction
					1,2,3 -> coordinate direction
					0 -> total kinetic energy

		**Output**:

		  	Scalar value of energy.

		**Note**:

			Only for 3-dim and only for data in spherical coordinates.

		"""
		if not self.n3 != 1:
			raise type(self).DimensionalityError("This method only supports 3D data");
		if self.geometry == "SPHERICAL":
			if axis == 1:
				try:
					return self.E_kin_1;
				except AttributeError:
					dm =self.get_dm();
					v1 = np.array(self.vx1);
					self.E_kin_1 = np.sum(0.5*dm*v1*v1);
					return self.E_kin_1;
			elif axis == 2:
				try:
					return self.E_kin_2;
				except AttributeError:
					dm =self.get_dm();
					v2 = np.array(self.vx2);
					self.E_kin_2 = np.sum(0.5*dm*v2*v2);
					return self.E_kin_2;
			elif axis == 3:
				try:
					return self.E_kin_3;
				except AttributeError:
					dm =self.get_dm();
					v3 = np.array(self.vx3);
					self.E_kin_3 = np.sum(0.5*dm*v3*v3);
					return self.E_kin_3;
			elif axis == 0:
				try:
					return self.E_kin_norm;
				except AttributeError:
					self.E_kin_norm = np.linalg.norm(np.array([	self.get_E_kin(1),
																self.get_E_kin(2),
																self.get_E_kin(3) ]));
					return self.E_kin_norm;
			else:
				raise ValueError("Selected axis {} is not one of [0,1,2,3]!".format(axis));

		else:
			raise type(self).GeometryError("Geometry {} is not supported in this method!".format(self.geometry));



	def get_Ncells(self):
		""" Return the number of active cells in the grid.

		**Input**:

		**Output**:

		  	Number of active cells in the grid.
		"""
		return self.rho.size;

	def get_rho_min(self):
		""" Return the minimum_density.

		**Input**:

		**Output**:

		  	Minimum density
		"""
		return np.min(self.rho);

	def get_rho_max(self):
		""" Return the minimum_density.

		**Input**:

		**Output**:

		  	Minimum density
		"""
		return np.max(self.rho);

	def get_ratio_cells_rho(self, min=None, max=None):
		""" Return the ratio of the number of cells compared to the total number,
			for which the density (rho) is larger than min (if min is set) and
			is smaller than max (if max is set).

		**Input**:

			min : lower bound to compare to
			max : upper bound to compare to

		**Output**:

			Ratio of number of cells to total number
		"""
		Ncells = self.get_Ncells();
		# Catch case that both bounds are not set.
		if min is None and max is None:
			raise AssertionError("Need to specify at least one bound!");

		mask = np.ones( self.rho.shape, dtype=bool);
		if not (min is None):
			mask = mask & (self.rho >= min);
		if not (max is None):
			mask = mask & (self.rho <= max);


		return float(mask.sum())/Ncells;


	def get_midplane_meshgrid(self):
		""" Construct a 2D meshgrid for the midplane coordinates of the disk.
		This can be used to plot variables such as the surface density.

		**Input**:

		**Output**:

			meshgrid with midplane coordinates

		"""
		if self.geometry == "SPHERICAL":
			x = np.array(self.x1) # Pluto's x1 dir -> r
			y = np.array(self.x3) # Pluto's x3 dir -> phi
			return np.meshgrid(x,y)
		else:
			raise type(self).GeometryError("Geometry {} is not supported in this method!".format(self.geometry));


	def get_surface_density_2d(self):
		""" Calculate the surface density of the disk by integrating over the
		vertical direction.
		**Input**:

		**Output**:

			2D array with surface densities

		"""
		try:
			return self.surface_density_2d
		except AttributeError:
			if self.geometry == "SPHERICAL":
				# First sum the mass array over axis 1 -> theta direction
				mass = np.sum(self.get_dm(), axis=1) # shape = (Nr, Nphi)
				r = np.array(self.x1)
				dr = np.array(self.dx1)
				dphi = np.array(self.dx3)
				area = np.repeat( np.expand_dims(r, 1), self.n3, axis=1)*\
					   np.repeat( np.expand_dims(dr, 1), self.n3, axis=1)*\
					   np.repeat( np.expand_dims(dphi, 0), self.n1, axis=0)
				self.surface_density_2d = mass/area
				return self.surface_density_2d

			else:
				raise type(self).GeometryError("Geometry {} is not supported in this method!".format(self.geometry));



	def get_surface_density_vs_radius(self):
		""" Calculate the surface density of the disk vs radius by
		averaging the 2d values over the azimutal direction.
		**Input**:

		**Output**:

			1D array with surface densities

		"""
		try:
			return self.surface_density_vs_radius
		except AttributeError:
			if self.geometry == "SPHERICAL":
				val2d = self.get_surface_density_2d()
				return np.average(val2d, axis=1)
			else:
				raise type(self).GeometryError("Geometry {} is not supported in this method!".format(self.geometry));

	#----------------------------------------------------------------------
	# Calculate the disk inclination as a function of radius.
	def	get_disk_inc_radial(self):
		disk_inc_radial = np.arccos(self.get_J_partial(3,1)/self.get_J_partial(0,1));
		return disk_inc_radial;

	def get_midplane_x2_index(self):
		""" Calculate the index corresponding to the midplane of a disk.
		This assumes that the disk is symmetic around theta = pi/2.

		**Input**:

		**Output**:

			integer index, if Ntheta is uneven
			array of the 2 integer indices around the midplane

		"""
		if self.n2 % 2 == 0:
			lower = self.n2/2 - 1
			upper = self.n2/2
			return np.array([lower, upper], dtype=np.int)
		else:
			return int(self.n2/2)

	def get_midplane_values(self, vals):
		""" Calculate the midplane values of vals.
		**Input**:

			3d array with cell values

		**Output**:

			2d array with midplane values

		"""
		idx = self.get_midplane_x2_index()
		if len(idx) == 2:
			return np.average(vals[:,idx,:], axis=1)
		else:
			return vals[:,idx,:]


	def get_soundspeed_midplane(self):
		""" Calculate the soundspeed in the midplane
		by p/rho using values from the midplane.
		**Input**:

		**Output**:

			2d array with soundspeed

		"""
		return self.get_midplane_values(self.prs)/self.get_midplane_values(self.rho)

	def get_soundspeed_radial(self):
		""" Calculate the azimutally averaged soundspeed in the midplane
		by p/rho.
		**Input**:

		**Output**:

			1d array with soundspeed

		"""
		return np.average(self.get_soundspeed_midplane(), axis=1)

	def get_rho_vert_slice_at_planet(self):
		""" Extract a vertical slice (r-theta) at the planet position
		This assumes that a planet is inside the disk on a circular orbit
		with an orbital period of T=1.0 and argument of pericenter of 0

		**Input**:

		**Output**:

			2d array with density values

		"""
		t = self.SimTime
		phi = (2*np.pi*t-np.pi/2)%(2*np.pi)
		return self.get_rho_vert_slice_at_phi(phi)


	def get_rho_vert_slice_at_phi(self, phi):
		""" Extract a vertical sclice (r-theta) at phi

		**Input**:

			phi: location of the slice

		**Output**:

			2d array with density values

		"""
		Nphi = round( ((phi/(2*np.pi))%1 * self.n3) ) % self.n3
		return self.rho[:,:,Nphi]


	def get_rho_vert_slice(self, Nphi):
		""" Extract a vertical slice (r-theta) at phi = Nphi/Nx3*2pi

		**Input**:

			Nphi: location of the slice

		**Output**:

			2d array with density values

		"""
		return self.rho[:,:,Nphi%self.n3]

	def get_rho_vert_slice_averaged(self):
		""" Extract a vertical slice (r-theta) azimutially averaged.

		**Input**:

		**Output**:

			2d array with density values

		"""
		return np.average(self.rho, axis=2)


	def get_vert_slice_meshgrid(self):
		""" Construct a 2D meshgrid for a vertical slice of the disk.

		**Input**:

		**Output**:

			meshgrid

		"""
		if self.geometry == "SPHERICAL":
			x = np.array(self.x1) # Pluto's x1 dir -> r
			y = np.array(self.x2) # Pluto's x2 dir -> theta
			return np.meshgrid(x,y)
		else:
			raise type(self).GeometryError("Geometry {} is not supported in this method!".format(self.geometry));
