# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 03:09:34 2017

RBDA v0.9

Small module containing Algorithms for calculating twistes and wrenches.
This module is just a small portion of algorithms available on:
    Rigid Body Dynamic Algorithms, by Featherstone, Roy (2008)

The latest version should be in ../moduletest/

Usage:

Download rbda.py into your computer and change working directory to where rbda is located. Example:

import os
os.chdir('c:/users/reyes fabian/my cubby/python/python_v-rep/moduletest')
os.chdir('c:/users/reyes fabian/my cubby/python/[modules]/rbda')

import rbda
plu = rbda.RBDA()

#Start using the functions. Example:
plu.rx(45, unit='deg')

----------------------------------------------------------------------------
Log:

[2018-03-17]:   Created v0.10
                -innertProduct() added
                -Fixed error inside calculation of inertial terms. gravity should be sym.Matrix when performing symbolic calculations
[2018-03-16]:   Created v0.9
                -LambdaWrapper() Class added for manipulation of symbolic expressions
[2017-09-15]:   Created v0.7
                -Basic algorithms up to FDab are ready
[2017-09-14]:   Created v0.6
                -Most algorithms up to Forward Dynamics Composite-rigid-body work symbolically and numerically
[2017-09-13]:   Updated symbolic version
[2017-09-06]:   Created v0.5 - Symbolic version
[2017-05-25]:   Created v0.4
                -Created and tested HandC. It works correctly with numerical values, but not yet with symbolic ones
                -Created FDcrb() and FDab()
                -Created forwardKinematics_sym() for obtaining jacobians in symbolic form
[2017-05-23]:   Created v0.3
                -finished createModel()
                -Created and tested forwardKinematics(), contactConstraints(), constrainedSubspace(), ID(), dimChange
[2017-05-22]:   Created v0.2
                -clean code
                -created and tested: jcalc(), Xpts(), inertiaTensor(), rbi()
                -created (but not finished) createModel()
[2017-05-15]:   Created v0.1
                -Start version control
                -Review and merge results from codenvy (ROBOMECH)

----------------------------------------------------------------------------
How to create expression dependent on time and differentiate w.r.t. time

from sym.physics.vector import dynamicsymbols
q1 = dynamicsymbols('q1')
q2 = dynamicsymbols('q2')
func = q1 + 2*q2
derFunc = sym.diff(func, sym.Symbol('t') )

#or

theta = dynamicsymbols('theta', symbolic=True)
rot = plu.rz(theta)

#This does NOT work
derRot = sym.diff(rot, sym.Symbol('t') )

#This works
derRot = rot.diff( sym.Symbol('t') )

@author: Reyes Fabian
"""

import math
import copy #for deepcopy
import sys

import numpy as np

import sympy as sym
from sympy.physics.vector import dynamicsymbols
from sympy.parsing.sympy_parser import parse_expr
#from sympy import diff, Symbol

from scipy import optimize

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
#from matplotlib.collections import PatchCollection

class RBDA(object):

    __version__ = '0.10.0'

    #Constructor
    def __init__(self):
        #self.__version__ = '0.8.0'
        self._modelCreated = False
        self.M = []

    # Print current version
    @classmethod
    def version(cls):
        return cls.__version__

    #Created model
    '''
    example:

    # robot without explicit floating base and 5 links -----------

    bodiesParams ={
    'mass':[0,0,1,1,1],
    'lengths':[0,0,0.15,0.15,0.15],
    'widths':[0,0,0.05,0.05,0.05],
    'heights':[0,0,0.05,0.05,0.05],
    'radius':[0,0,0,0,0,0],
    'Lx':[0,0,0.075,0.075,0.075],
    'Ly':[0,0,0,0,0],
    'Lz':[0,0,0,0,0],
    'shapes':["Prism","Prism","Prism","Prism","Prism"],
    'jointType':['Rz','Rz','Rz','Rz','Rz'],
    'xt':[[0,0,0],[0,0,0],[0.15,0,0],[0.15,0,0],[0.15,0,0]],
    'gravityAxis':'y'
    }

    plu.createModel(False, 5, '[kg m s]', bodiesParams, None, None)

    or

    # robot with floating base and 3 links ----------------------

    bodiesParams ={
    'mass':[1,1,1],
    'lengths':[0.15,0.15,0.15],
    'widths':[0.05,0.05,0.05],
    'heights':[0.05,0.05,0.05],
    'radius':[0,0,0],
    'Lx':[0.075,0.075,0.075],
    'Ly':[0,0,0],
    'Lz':[0,0,0],
    'shapes':["Prism","Prism","Prism"],
    'jointType':['Rz','Rz'],
    'xt':[[0,0,0],[0.15,0,0],[0.15,0,0]],
    'gravityAxis':'y'
    }

    plu.createModel(True, 2, '[kg m s]', bodiesParams, None, None)

    # 1 body (unilateral) constraint ------------------------------

    constraintsInformation={
    'nc':1,
    'body':[2],
    'contactPoint':[[0.075,0,0]],
    'contactAngle':[0],
    'constraintType':['bodyContact'],
    'contactModel':['pointContactWithoutFriction'],
    'contactingSide':['left']
    }

    # 3 constraints (one body contact and two friction cones) -----

    constraintsInformation={
    'nc':3,
    'body':[2,3,4],
    'contactPoint':[[0.075,0,0],[0.075,0,0],[0.075,0,0]],
    'contactAngle':[0,0,0],
    'constraintType':['bodyContact','non-slippageWithFriction','non-slippageWithFriction'],
    'contactModel':['pointContactWithoutFriction','pointContactWithoutFriction','pointContactWithoutFriction'],
    'contactingSide':['left','left','left']
    }

    plu.constrainedSubspace(constraintsInformation, Jacobians)
    '''
    def createModel(self, floatingBaseBool, noJoints, units, bodiesParams, DHParameters, conInformation):

        # Create deep copy of the parameters
        self.model = copy.deepcopy(bodiesParams)

        #Decide if we have a floating base
        self.model['floatingBase'] = floatingBaseBool

        # nFB: degrees of freedom of the floating base
        if floatingBaseBool is True:
            self.model['nFB'] = 3
        else:
            self.model['nFB'] = 0

        # nR: degrees of freedom (DoF) of the robot including the floating base (nFB).
        self.model['DoF'] = self.model['nFB'] + noJoints
        self.model['nR'] = self.model['DoF']

        # We have n-2 links (bodies) and n-nFB actuated joints*)
        # How many bodies are in the system
        if floatingBaseBool is True:
            bodies = noJoints + 3 #I am considering the virtual bodies too
        else:
            bodies = noJoints
        self.model['nB'] = bodies
        self.model['nA'] = noJoints

        # Create symbolic vectors

        #A symbol for time
        self.time = sym.symbols('t')

        # Option 1: Array of symbols. However, they do not depend on time
#        self.qVector = sym.symarray( 'qVector', self.model['DoF'] )
#        self.dq = sym.symarray( 'dq', self.model['DoF'] )

        # Option 2: dynamic symbols and return a list. Easiest to manipulate afterwards
        qVector = ['q_'+str(i).zfill(2) for i in range(self.model['DoF'])]
        dq = ['dq_'+str(i).zfill(2) for i in range(self.model['DoF'])]
        ddq = ['ddq_'+str(i).zfill(2) for i in range(self.model['DoF'])]

        self.qVector = dynamicsymbols(qVector)
        self.dq = dynamicsymbols(dq)
        self.ddq = dynamicsymbols(ddq)

        #1st order derivatives of the joint angles
        self.qDer = [sym.diff(var, self.time) for var in self.qVector]
        self.qDerDer = [sym.diff(var, self.time, 2) for var in self.qVector]

        #tuples for substituting Derivative(q(t),t) for dq
        #self.stateSym = zip(self.qDer, self.dq) + zip(self.qDerDer, self.ddq) # This does not work in Python 3
        self.stateSym = list( zip(self.qDer + self.qDerDer, self.dq + self.ddq) )

        # Option 3: return a numpy array
#        self.qVector = np.array( dynamicsymbols(qVector) )
#        self.dq = np.array( dynamicsymbols(dq) )
#        self.ddq = np.array( dynamicsymbols(ddq) )

        #For now, make the actuator torques not dependent on time
        self.tauAct = sym.symarray( 'tauAct', self.model['nA'] )

        #Decide gravity acceleration based on units
        if units.lower() == '[kg m s]':
            self.model['g'] = 9.8 #gravity acceleration [m/s^2]
        elif units.lower() == '[kg cm s]':
            self.model['g'] = 980 #gravity acceleration [cm/s^2]
        else:
            print("Units not recognized. Using [kg m s]")
            self.model['g'] = 9.8

        #Gravity spatial force expressed in inertial frame.
        if self.model['gravityAxis'].lower() == 'x':
            self.model['inertialGrav'] = np.array( [0.0,0.0,0.0,-(self.model['g']),0.0,0.0] )
        elif self.model['gravityAxis'].lower() == 'y':
            self.model['inertialGrav'] = np.array( [0.0,0.0,0.0,0.0,-(self.model['g']),0.0] )
        elif self.model['gravityAxis'].lower() == 'z':
            self.model['inertialGrav'] = np.array( [0.0,0.0,0.0,0.0,0.0,-(self.model['g'])] )
        else:
            print('Gravity direction not recognized. Assuming it is in the y-axis.')
            self.model['inertialGrav'] = np.array( [0.0,0.0,0.0,0.0,-(self.model['g']),0.0] )

        #assign parameters. If there is a floating base, then there should be two virtual (massless) links
        self.model.update({'com':[]})

        if floatingBaseBool is True:
            self.model['mass'][:0] = [0,0]
            self.model['lengths'][:0] = [0,0]
            self.model['widths'][:0] = [0,0]
            self.model['heights'][:0] = [0,0]
            self.model['radius'][:0] = [0,0]
            self.model['Lx'][:0] = [0,0]
            self.model['Ly'][:0] = [0,0]
            self.model['Lz'][:0] = [0,0]
            self.model['shapes'][:0] = [None,None]
            self.model['jointType'][:0] = ['Px','Py','Rz']
            self.model['xt'][:0] = [[0,0,0],[0,0,0]]

        rbInertia = []
        XT = []

        for i in range(bodies):
            #Obtain the rigid body inertia
            m = (self.model['mass'])[i]
            center = [ (self.model['Lx'])[i],(self.model['Ly'])[i],(self.model['Lz'])[i] ]
            (self.model['com']).append(center)

            inertia = self.inertiaTensor(\
            [m, (self.model['lengths'])[i], (self.model['widths'])[i], (self.model['heights'])[i],(self.model['radius'])[i]], (self.model['shapes'])[i])

            rbInertia.append( self.rbi(m, center, inertia) )

            #Create auxiliary transformations XT
            XT.append( self.xlt( (self.model['xt'])[i] ) )

        self.model['rbInertia'] = rbInertia
        self.model['XT'] = XT

        #self.model['parents'] = range(bodies)#This represents a serial chain
        self.model['parents'] = range(-1, bodies-1)#This represents a serial chain

        if floatingBaseBool is True:
            self.model['bodiesRealQ'] = [False for i in range(self.model['nFB'] - 1)] + \
            [True for i in range(noJoints+1)]
        else:
            self.model['bodiesRealQ'] =  [True for i in range(noJoints)]

        self._modelCreated = True

    # Print some values of the model
    def presentation(self):
        if self._modelCreated:
            print("The system has {} bodies and {} joints".format(self.model['nB'], self.model['nA']))
            print("Does the system has a floating base? {}".format(self.model['floatingBase']))
            print("The following are the real bodies: {}".format(self.model['bodiesRealQ']))
            print("The set of generalized coordinates is: {}".format(self.stateSym))
            print("Currently, gravity is acting along the axis {}, represented as {}".format(self.model['gravityAxis'], self.model['inertialGrav']))
        else:
            print("Model not created yet. Use createModel() first.")


    #three-dimensional rotation around the x axis. Works numerically or symbollicaly
    #IF using a symbolic variable, a numeric matrix can be obtained as rx(theta).subs(theta, 1.0), for example.
    @staticmethod
    def rx(theta, unit='rad', symbolic=False):

        #If symbolic, then solve using sympy
        if symbolic:

            if isinstance(theta, tuple(sym.core.all_classes)):
                c_theta = sym.cos(theta)
                s_theta = sym.sin(theta)

            else:
                if unit.lower() == 'deg':
                    theta=math.radians(theta)

                c_theta = math.cos(theta)
                s_theta = math.sin(theta)

            return sym.Matrix( [[1.0,0.0,0.0],[0.0,c_theta,s_theta],[0.0,-s_theta,c_theta]] )

        #else, if it is numeric use numpy
        else:

            if unit.lower() == 'deg':
                theta = math.radians(theta)

            c_theta = math.cos(theta)
            s_theta = math.sin(theta)

            return np.array([[1.0,0.0,0.0],[0.0,c_theta,s_theta],[0.0,-s_theta,c_theta]])

    #three-dimensional rotation around the y axis. Works numerically or symbollicaly
    @staticmethod
    def ry(theta, unit='rad', symbolic=False):

        #If symbolic, then solve using sympy
        if symbolic:

            if isinstance(theta, tuple(sym.core.all_classes)):
                c_theta = sym.cos(theta)
                s_theta = sym.sin(theta)

            else:
                if unit.lower() == 'deg':
                    theta=math.radians(theta)

                c_theta = math.cos(theta)
                s_theta = math.sin(theta)

            return sym.Matrix( [[c_theta,0.0,-s_theta],[0.0,1.0,0.0],[s_theta,0.0,c_theta]] )

        #else, if it is numeric use numpy
        else:

            if unit.lower() == 'deg':
                theta = math.radians(theta)

            c_theta = math.cos(theta)
            s_theta = math.sin(theta)

            return np.array([[c_theta,0.0,-s_theta],[0.0,1.0,0.0],[s_theta,0.0,c_theta]])

    #three-dimensional rotation around the z axis. Works numerically or symbollicaly
    @staticmethod
    def rz(theta, unit='rad', symbolic=False):

        #If symbolic, then solve using sympy
        if symbolic:

            if isinstance(theta, tuple(sym.core.all_classes)):
                c_theta = sym.cos(theta)
                s_theta = sym.sin(theta)

            else:
                if unit.lower() == 'deg':
                    theta=math.radians(theta)

                c_theta = math.cos(theta)
                s_theta = math.sin(theta)

            return sym.Matrix( [[c_theta,s_theta,0.0],[-s_theta,c_theta,0.0],[0.0,0.0,1.0]] )

        #else, if it is numeric use numpy
        else:

            if unit.lower() == 'deg':
                theta = math.radians(theta)

            c_theta = math.cos(theta)
            s_theta = math.sin(theta)

            return np.array([[c_theta,s_theta,0.0],[-s_theta,c_theta,0.0],[0.0,0.0,1.0]])

    #general container for six-dimensional rotations. Input 'plE' is a three dimensional rotation matrix
    @staticmethod
    def rot(plE, symbolic=False):

         #change list into np.ndarray, if necessary
        if type(plE) == list:
            plE = np.array( plE )

        #obtain output
        zero = np.zeros( (3,3) )

        #If symbolic, then solve using sympy
        if symbolic:
            return sym.Matrix( np.bmat([[plE, zero], [zero, plE]]) )
        else:
            return np.array( np.bmat([[plE, zero], [zero, plE]]) )

    #six-dimensional rotations. Input is an angle
    def rotX(self, theta, unit='rad', symbolic=False):
        return self.rot(self.rx(theta, unit, symbolic), symbolic)

    #six-dimensional rotations. Input is an angle
    def rotY(self, theta, unit='rad', symbolic=False):
        return self.rot(self.ry(theta, unit, symbolic), symbolic)

    #six-dimensional rotations. Input is an angle
    def rotZ(self, theta, unit='rad', symbolic=False):
        return self.rot(self.rz(theta, unit, symbolic), symbolic)

    #Derivate of a rotation matrix. rot must be a sym.Matrix object, and variable a sym.Symbol
    def derRot(self, rot, variable):
        return rot.diff( variable )

    '''
    Cross product operator (rx pronounced r-cross).
    Input 'r' is either a 3D (point) vector or a skew-symmetric (3x3) matrix.
    A symbolic array can be created as r = sym.symarray( 'r', 3 ) or r = sym.symarray( 'r', (3,3) )
    '''
    @staticmethod
    def skew(r, symbolic=False):

        #If symbolic, then solve using sympy
        if symbolic:
            #change list into sym.matrix, if necessary
            if type(r) == list:
                r = sym.Matrix(r)
            elif type(r) == np.ndarray:
                r = sym.Matrix(r)

            if r.shape[1] == 1:#Change from vector to matrix
                return sym.Matrix([[0.0, -r[2], r[1]],[r[2], 0.0, -r[0]],[-r[1], r[0], 0.0]])
            else:#Change from matrix to vector
                return 0.5*sym.Matrix([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])
        else:
            #change list into np.ndarray, if necessary
            if type(r) == list:
                r = np.array( r )

            if r.ndim == 1:#Change from vector to matrix
                return np.array([[0.0, -r[2], r[1]],[r[2], 0.0, -r[0]],[-r[1], r[0], 0.0]])
            elif r.ndim == 2:#Change from matrix to vector
                return 0.5*np.array([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])
            else:
                print('Wrong input')
                return [0]

    #Translation transformation. Input is a three-dimensional vector
    def xlt(self, r, symbolic=False):

        #If symbolic, then solve using sympy
        if symbolic:

            zero = sym.zeros(3)
            identity = sym.eye(3)

            out = sym.Matrix( np.bmat( [[identity, zero],[-(self.skew(r, symbolic=True)), identity]] ) )
        else:

            zero = np.zeros( (3,3) )
            identity = np.identity(3)

            out = np.array( np.bmat( [[identity, zero],[-(self.skew(r)), identity]] ) )

        return out

    #General Plücker transformation for MOTION vectors.
    #Inputs are a general 3D (or 6D) rotation matrix and a 3D traslation
    def pluX(self, plE, r, symbolic=False):

        #If symbolic, then solve using sympy
        if symbolic:

            if type(plE) == list:
                plE = sym.Matrix(plE)

            #If we received a 3D rotation matrix, change into 6D
            if plE.shape[0] == 3:
                plE = self.rot(plE, symbolic=True)

            out = plE*(self.xlt(r, symbolic=True))

        else:

            if type(plE) == list:
                plE = np.array(plE)

            #If we received a 3D rotation matrix, change into 6D
            if plE.shape[0] == 3:
                plE = self.rot(plE)

            out = np.dot( plE, self.xlt(r) )

        return out

    '''
    Plücker transformation for FORCE vectors.
    'pluX' is a 6x6 Plucker transformation or 'pluX' is a 3D rotation and 'r' is a translation vector
    '''
    def pluXf(self, pluX, r=None, symbolic=False):

        #If symbolic, then solve using sympy
        if symbolic:

            #change list into sym.Matrix, if necessary
            if type(pluX) == list:
                pluX = sym.Matrix( pluX )
            if r is not None and type(r) == list:
                r = sym.Matrix( r )

            #If a translation vector is not received, the input pluX is a 6D transformation
            if r is None:
                out11 = pluX[0:3,0:3]
                out12 = pluX[3:6,0:3]
                out21 = sym.zeros( 3 )
                out22 = pluX[0:3,0:3]

                out =  sym.Matrix( np.bmat([[out11, out12],[out21, out22]]) )
            else:
                #FIXME Improve this. Avoid inverse
                invTrans = ( self.xlt(r, symbolic=True) ).inv()
                out =  (self.rot(pluX, symbolic=True))*(invTrans.T)

        else:

            #change list into np.ndarray, if necessary
            if type(pluX) == list:
                pluX = np.array( pluX )
            if r is not None and type(r) == list:
                r = np.array( r )

            #If a translation vector is not received,
            #the input pluX is a 6D transformation, just manipulate as necessary
            if r is None:
                out11 = pluX[0:3,0:3]
                out12 = pluX[3:6,0:3]
                out21 = np.zeros( (3,3) )
                out22 = pluX[0:3,0:3]

                out =  np.bmat([[out11, out12],[out21, out22]])
            else:
                #FIXME Improve this. Avoid inverse
                invTrans = np.linalg.inv( self.xlt(r) )
                out =  np.dot( self.rot(pluX), np.transpose(invTrans) )

        return out

    #Inverse for pluX. Inputs are a general 3D rotation matrix and a 3D traslation
    #If receiving a 6 by 6 matrix ASSUME it is a Plucker transformation
    def invPluX(self, plE, r=None, symbolic=False):

        #If symbolic, then solve using sympy
        if symbolic:

            #If we receive a 6x6 matrix, do nothing. Otherwise, obtain the transformation
            if plE.shape[0] == 6:
                B_X_A = plE
            else:
                B_X_A = self.pluX(plE, r, symbolic)

            out11 = (B_X_A[0:3,0:3]).T
            out12 = sym.zeros( 3 )
            out21 = (B_X_A[3:6,0:3]).T
            out22 = out11

            #return A_X_B
            out = sym.Matrix( np.bmat([[out11, out12],[out21, out22]]) )

        else:

            #If we receive a 6x6 matrix, do nothing. Otherwise, obtain the transformation
            if plE.shape[0] == 6:
                B_X_A = plE
            else:
                B_X_A = self.pluX(plE, r)

            out11 = np.transpose( B_X_A[0:3,0:3] )
            out12 = np.zeros( (3,3) )
            out21 = np.transpose( B_X_A[3:6,0:3] )
            out22 = out11

            #return A_X_B
            out = np.array( np.bmat([[out11, out12],[out21, out22]]) )

        return out

    #Inverse for pluXf
    #pluX is a 6x6 Plucker transformation or pluX is a 3-D rotation and r is a translation vector
    #If receiving a 6 by 6 matrix ASSUME it is a (force) Plucker transformation
    def invPluXf(self, pluX, r=None, symbolic=False):

        #If symbolic, then solve using sympy
        if symbolic:

            #If we receive a 6x6 matrix, do nothing. Otherwise, obtain the transformation
            if pluX.shape[0] == 6:
                B_Xf_A = pluX
            else:
                B_Xf_A = self.pluXf(pluX, r, symbolic)

            out11 = ( B_Xf_A[0:3,0:3] ).T
            out12 = ( B_Xf_A[0:3,3:6] ).T
            out21 = sym.zeros( 3 )
            out22 = out11

            #return A_Xf_B
            out = sym.Matrix( np.bmat([[out11, out12],[out21, out22]]) )

        else:
            #If we receive a 6x6 matrix, do nothing. Otherwise, obtain the transformation
            if pluX.shape[0] == 6:
                B_Xf_A = pluX
            else:
                B_Xf_A = self.pluXf(pluX, r)

            out11 = np.transpose( B_Xf_A[0:3,0:3] )
            out12 = np.transpose( B_Xf_A[0:3,3:6] )
            out21 = np.zeros( (3,3) )
            out22 = out11

            #return A_Xf_B
            out = np.array( np.bmat([[out11, out12],[out21, out22]]) )

        return out

    #Definitions of free space for one-dimensional joints
    @staticmethod
    def freeMotionSpan(typeMotion, symbolic=False):
        if typeMotion.lower() == 'rx':
            out = [1.0,0.0,0.0,0.0,0.0,0.0]
        elif typeMotion.lower() == 'ry':
            out = [0.0,1.0,0.0,0.0,0.0,0.0]
        elif typeMotion.lower() == 'rz':
            out = [0.0,0.0,1.0,0.0,0.0,0.0]
        elif typeMotion.lower() == 'px':
            out = [0.0,0.0,0.0,1.0,0.0,0.0]
        elif typeMotion.lower() == 'py':
            out = [0.0,0.0,0.0,0.0,1.0,0.0]
        elif typeMotion.lower() == 'pz':
            out = [0.0,0.0,0.0,0.0,0.0,1.0]
        else:
            out = [0.0,0.0,0.0,0.0,0.0,0.0]

        #If symbolic, return a sym.Matrix, otherwise return a np.ndarray
        if symbolic:
            return sym.Matrix(out)
        else:
            return np.array(out).reshape(6,1)

    #Coordinate transformation. Similarity or congruence. All 6D Matrices. X is always MOTION transformation
    def coordinateTransform(self, X, A, transType, inputType, symbolic=False, simplification=False):

        #If symbolic proceed using sympy
        if symbolic:

            if transType.lower()=='similarity':
                if inputType.lower()=='motion':
                    output = X*A*(self.invPluX(X,symbolic=symbolic))
                elif inputType.lower()=='force':
                    output = ( self.pluXf(X,symbolic=symbolic) )*A*( self.pluXf(self.invPluX(X,symbolic=symbolic), symbolic=symbolic) )
                else:
                    print('Incorrect input type')
                    output = sym.zeros(6)
            elif transType.lower()=='congruence':
                if inputType.lower()=='motion':
                    output = ( self.pluXf(X, symbolic=symbolic) )*A*( self.invPluX(X,symbolic=symbolic) )
                elif inputType.lower()=='force':
                    output = X*A*( self.pluXf(self.invPluX(X,symbolic=symbolic), symbolic=symbolic) )
                else:
                    print('Incorrect input type')
                    output = sym.zeros(6)
            else:
                print('Incorrect transformation type')
                output = sym.zeros(6)

            if simplification is True:
                return sym.simplify(output)
            else:
                return output

        else:
            # A.dot(B).dot(C) = reduce(np.dot, [A1, A2, ..., An])
            # multi_dot([A1d, B, C, D])

            if transType.lower()=='similarity':
                if inputType.lower()=='motion':
                    output = reduce(np.dot, [X, A, self.invPluX(X)])
                elif inputType.lower()=='force':
                    output = reduce(np.dot, [self.pluXf(X), A, self.pluXf( self.invPluX(X) )])
                else:
                    print('Incorrect input type')
                    output = np.zeros((6,6))
            elif transType.lower()=='congruence':
                if inputType.lower()=='motion':
                    output = reduce(np.dot, [self.pluXf(X), A, self.invPluX(X)])
                elif inputType.lower()=='force':
                    output = reduce(np.dot, [X, A, self.pluXf( self.invPluX(X) )])
                else:
                    print('Incorrect input type')
                    output = np.zeros((6,6))
            else:
                print('Incorrect transformation type')
                output = np.zeros((6,6))

            return output

    #6D rotation matrix corrresponding to a spherical joint
    def eulerRot(self, angles, typeRot, outputFrame ='local', unit='rad', symbolic=False):

        #If using symbolic values, use sympy
        if symbolic:

            #Change to sympy object
            angles = sym.Matrix(angles)

            if typeRot.lower() == 'zyx':
                rotation = \
                (self.rotX(angles[2],symbolic=True))*(self.rotY(angles[1],symbolic=True))*(self.rotZ(angles[0],symbolic=True))
            else:
                rotation = sym.zeros(6)

        #Otherwise, use numpy
        else:

            #Change type if necessary
            if type(angles) == list:
                angles = np.array(angles)

            #Change units
            if unit.lower() == 'deg':
                angles = np.radians(angles)

            if typeRot.lower() == 'zyx':
                rotation = reduce(np.dot, [self.rotX(angles[2]), self.rotY(angles[1]), self.rotZ(angles[0])])
            else:
                rotation = np.zeros((6,6))

        #Change the output based on the reference frame of interest
        if outputFrame.lower() == 'local':
            return rotation
        elif outputFrame.lower() == 'global':
            if symbolic:
                return rotation.T
            else:
                return np.transpose(rotation)
        else:
            print('Desired output frame not recognized. Returning local')
            return rotation

    #Jacobian corresponding to a rotation matrix from a spherical joint
    def eulerJacobian(self, angles, typeRot, outputFrame = 'local', unit='rad', symbolic=False):

        #If using symbolic values, use sympy
        if symbolic:
            #Change type if necessary
            if type(angles) == list:
                angles = sym.Matrix(angles)

            if typeRot.lower() == 'zyx':
                spanS = sym.zeros(3,6)
                spanS[0,:] = self.freeMotionSpan('rz').reshape(1,6)
                spanS[1,:] = self.freeMotionSpan('ry').reshape(1,6)
                spanS[2,:] = self.freeMotionSpan('rx').reshape(1,6)
                #spanS = [self.freeMotionSpan('rz'), self.freeMotionSpan('ry'), self.freeMotionSpan('rx')]
                rots = \
                [self.rotZ(angles[0], symbolic=True), self.rotY(angles[1], symbolic=True), self.rotX(angles[2], symbolic=True)]
                rot = self.rotZ(angles[0], symbolic=True)

                #acumulate rotations
                for i in range(1,3):
                    rot = rots[i]*rot #total rotation matrix

                    #propagate them to each matrix spanS
                    for j in range(i):
                        spanS[j,:] = ( rots[i]*(spanS[j,:].reshape(6,1)) ).reshape(1,6)

                #return a 6x3 matrix
                spanS = spanS.T
            else:
                spanS = sym.zeros(6,3)

        #Otherwise use numpy
        else:
            #Change type if necessary
            if type(angles) == list:
                angles = np.array(angles)

            #Change units
            if unit.lower() == 'deg':
                angles = np.radians(angles)

            if typeRot.lower() == 'zyx':
                spanS = np.bmat( [self.freeMotionSpan('rz'), self.freeMotionSpan('ry'), self.freeMotionSpan('rx')] )
                spanS = np.array(spanS.transpose())

                rots = [self.rotZ(angles[0]), self.rotY(angles[1]), self.rotX(angles[2])]
                rot = self.rotZ(angles[0])

                #acumulate rotations
                for i in range(1,3):
                    rot = np.dot( rots[i], rot ) #total rotation matrix

                    #propagate them to each matrix spanS
                    for j in range(i):
                        spanS[j] =  np.dot( rots[i], spanS[j])

                #return a 6x3 matrix
                spanS = spanS.transpose()
            else:
                spanS = np.zeros( (6,3) )

        #if the frame is the global, multiply by the corresponding matrix
        if outputFrame.lower() == 'local':
            return spanS
        elif outputFrame.lower() == 'global':
            if symbolic:
                return (rot.T)*spanS
            else:
                return np.dot( np.transpose(rot), spanS )
        else:
            print('Desired output frame not recognized. Returning local')
            return spanS

    '''
    Calculates Xj and S in body coordinates.
    Input: 'typeJoint' is joint type (e.g., 'Rx') and coordinate.
    Output: Plücker transformation Xj and motion subspace matrix S
    For planr floating base (type='fbPlanar'), the order of coordinates is [theta,x,y]
    FIXME: Currently, only 1DOF joints are supported
    '''
    def jcalc(self, typeJoint, q, unit='rad', symbolic=False):

        if not symbolic:
            #Change units
            if unit.lower() == 'deg':
                q = math.radians(q)

        if typeJoint.lower() == 'rx':
            XJ = self.rotX(q, unit, symbolic)
            S = self.freeMotionSpan(typeJoint, symbolic)
        elif typeJoint.lower() == 'ry':
            XJ = self.rotY(q, unit, symbolic)
            S = self.freeMotionSpan(typeJoint, symbolic)
        elif typeJoint.lower() == 'rz':
            XJ = self.rotZ(q, unit, symbolic)
            S = self.freeMotionSpan(typeJoint, symbolic)
        elif typeJoint.lower() == 'px':
            XJ = self.xlt([q,0,0], symbolic)
            S = self.freeMotionSpan(typeJoint, symbolic)
        elif typeJoint.lower() == 'py':
            XJ = self.xlt([0,q,0], symbolic)
            S = self.freeMotionSpan(typeJoint, symbolic)
        elif typeJoint.lower() == 'pz':
            XJ = self.xlt([0,0,q], symbolic)
            S = self.freeMotionSpan(typeJoint, symbolic)
        else:
            print("Joint type not recognized")

            if symbolic:
                XJ = sym.eye(6)
                S = sym.zeros(6)
            else:
                XJ = np.identity(6)
                S = np.zeros((6,1))

        return XJ,S

    '''
    Xpts(pluX, pts): Transform points between coordinate frames.
    'pts' can be a vector or a list of vectors.
    'pluX' is a 6x6 Plucker transformation.
    points 'pts' are expressed in reference frame A, and pluX is a transformation from frame A to B.
    Output is a list of points w.r.t. frame B.
    '''
    def Xpts(self, pluX, pts, symbolic=False):

        #If symbolic proceed using sympy
        if symbolic:

            #Change type if necessary
            if type(pluX) == list:
                pluX = sym.Matrix(pluX)
            if type(pts) == list:
                pts = np.array(pts)#Keep this as an array

            E = pluX[0:3,0:3]#Rotation component of pluX
            r = -self.skew( ( E.T )*( pluX[3:6,0:3] ) , symbolic=True)#Translation component of pluX

            if pts.ndim == 1:
                newPoints = E*sym.Matrix(pts-r)
            else:
                newPoints = []
                for i,point in enumerate(pts):
                    point = sym.Matrix(point)
                    newPoints.append( E*(point-r) )

            return newPoints

        else:
            #Change type if necessary
            if type(pluX) == list:
                pluX = np.array(pluX).astype(float)
            if type(pts) == list:
                pts = np.array(pts).astype(float)

            E = pluX[0:3,0:3]#Rotation component of pluX
            r = -self.skew( np.dot( np.transpose(E), pluX[3:6,0:3] ) )#Translation component of pluX

            if pts.ndim == 1:
                newPoints = np.dot( E, pts-r )
            else:
                newPoints = pts
                #newPoints = [(np.dot(E,point-r)) for point in pts]
                for i,point in enumerate(pts):
                    newPoints[i] = np.dot(E, point-r)

            return newPoints

    '''
    forwardKinematics(): Automatic process to obtain all transformations from the inertial \
    frame {0} to Body-i. The geometric Jacobians of Body i represents the overall motion and \
    not only the CoM (w.r.t. local coordinates). Use createModel() first.
    'q' is the vector of generalized coordinates (including floating base)
    '''
    def forwardKinematics(self, q, unit='rad', symbolic=False):

        if self._modelCreated:

            #If symbolic proceed using sympy
            if symbolic:

                #Change types
                if type(q) == list:
                    q = sym.Matrix(q)

                parentArray = self.model['parents']
                Xup = [sym.zeros( 6, 6 ) for i in range(self.model['nB'])]
                S = [sym.zeros( 6, 1 ) for i in range(self.model['nB'])]
                X0 = [sym.zeros( 6, 6 ) for i in range(self.model['nB'])]
                invX0 = [sym.zeros( 6, 6 ) for i in range(self.model['nB'])]
                jacobian = [sym.zeros( 6, self.model['nB'] ) for i in range(self.model['nB'])]

                for i in range(self.model['nB']):

                    #Obtain JX and S
                    XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i], symbolic=True)

                    S[i] = tempS
                    Xup[i] = XJ*( np.array( (self.model['XT'])[i] ) )

                    #If the parent is the base
                    if parentArray[i] == -1:
                        X0[i] = Xup[i]
                    else:
                        X0[i] =  Xup[i]*X0[i-1]

                    #Obtain inverse mappings
                    #invX0[i] = np.linalg.inv(X0[i])
                    #invX0[i] = (X0[i]).inv()
                    invX0[i] = self.invPluX(X0[i], None, symbolic=True)

                    #We change the previous S into local coordinates of Body-i
                    for j in range(i):
                        S[j] = Xup[i]*S[j]

                    jacobian[i] = sym.Matrix( np.transpose(S) )

            #Solve numerically
            else:

                #Change types and unit if necessary
                if type(q) == list:
                    q = np.array(q)
                if unit.lower() == 'deg':
                    q = np.radians(q)

                if self._modelCreated:

                    parentArray = self.model['parents']
                    Xup = np.zeros( (self.model['nB'], 6, 6) )
                    S = np.zeros( (self.model['nB'], 6) )
                    X0 = np.zeros( (self.model['nB'], 6, 6) )
                    invX0 = np.zeros( (self.model['nB'], 6, 6) )
                    jacobian = np.zeros( (self.model['nB'], 6, self.model['nB']) )

                    for i in range(self.model['nB']):
                        XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i])
                        S[i] = tempS.reshape(1,6)
                        Xup[i] = np.dot( XJ, (self.model['XT'])[i] )

                        #If the parent is the base
                        if parentArray[i] == -1:
                            X0[i] = Xup[i]
                        else:
                            X0[i] = np.dot( Xup[i], X0[i-1] )

                        #Obtain inverse mappings
                        invX0[i] = np.linalg.inv(X0[i])

                        #We change the previous S into local coordinates of Body-i
                        for j in range(i):
                            S[j] = np.dot( Xup[i], S[j] )

                        jacobian[i] = np.transpose(S)

            #Return as numpy arrays. Convert to sympy Matrices outside if necessary
            return Xup,X0,invX0,jacobian

        else:
            print("Model not yet created. Use createModel() first.")
            return

    '''
    crossM[v]: Spatial cross product for MOTION vectors.
    The input ' v' is a twist (either 3 D or 6 D).*)
    '''
    def crossM(self, v, symbolic=False):

        #Change type and make unidimensional array
        v = np.array(v).flatten()


        #IF we received a 6 dimensional twist (omega, upsilon)
        if v.size == 6:
            out11 = self.skew(v[0:3])
            out12 = np.zeros( (3,3) )
            out21 = self.skew(v[3:6])
            out22 = out11

            out =  np.bmat([[out11, out12],[out21, out22]])
        #Otherwise, we received a 3 dimensional twist (omega_z, upsilon_x, upsilon_y)
        elif v.size == 3:
            out = np.array( [[0,0,0],[v[2],0,-v[0]],[-v[1],v[0],0]] )
        else:
            print('Wrong size')
            out = np.array([0,0,0,0,0,0])

        #If symbolic proceed using sympy
        if symbolic:
            return sym.Matrix(out)
        else:
            return out

    '''
    crossF[v]. Spatial cross product for FORCE vectors.
    The input ' v' is a twist (either 3 D or 6 D).
    '''
    def crossF(self, v, symbolic=False):

        #If symbolic proceed using sympy
        if symbolic:
            return -(self.crossM(v, symbolic=True)).T
        else:
            return -np.transpose(self.crossM(v))

    '''
    inertiaTensor[params, type, connectivity] : Calculate inertia tensor of a body
    Assume input is an array of parameters in the form:
    params={mass, l, w, h, r} and typeObj is a string (e.g., "SolidCylinder")
    l:= length along x     w:= length along y     h:= length along z      r:=radius
    This should always be a np.ndarray
    '''
    def inertiaTensor(self, params, typeObj):

        #Unload the parameters
        mass = params[0]
        length = params[1]
        width = params[2]
        height = params[3]

        if len(params)==5:
            radius = params[4]
        else:
            radius = 0

        if typeObj is None:
            Ixx = Iyy = Izz = 0.0
        elif typeObj.lower() == 'prism':
            Ixx = (1.0/12.0)*(height*height+width*width)
            Iyy = (1.0/12.0)*(height*height+length*length)
            Izz = (1.0/12.0)*(length*length+width*width)
        elif typeObj.lower() == 'cylinder':
            Ixx = (1.0/12.0)*(height*height + 3*radius*radius)
            Iyy = (1.0/12.0)*(height*height + 3*radius*radius)
            Izz = 0.5*radius*radius
        elif typeObj.lower() == 'verticalcylinder':
            Ixx = (1.0/12.0)*(height*height + 3*radius*radius)
            Iyy = (1.0/12.0)*(height*height + 3*radius*radius)
            Izz = 0.5*radius*radius
        elif typeObj.lower() == 'horizontalcylinder':
            Ixx = 0.5*radius*radius
            Iyy = (1.0/12.0)*(height*height + 3*radius*radius)
            Izz = (1.0/12.0)*(height*height + 3*radius*radius)
        else:
            print("Shape not supported")
            Ixx = Iyy = Izz = 0.0

        inertiaT = np.array( [[Ixx,0.0,0.0],[0.0,Iyy,0.0],[0.0,0.0,Izz]] )

        return mass*inertiaT

    '''
    rbi(m,c, I): RigidBodyInertia of a body.
    'mass' is the mass, 'center' is the position of the CoM (2D or 3D),
    inertiaT is the rotational inertia around CoM (can be obtained with intertiaTensor())
    '''
    def rbi(self, mass, center, inertiaT, symbolic=False):

        #If the input is a 3D vector, obtain the 6x6 rbi
        if len(center) == 3:
            skewC = self.skew(center, symbolic=symbolic)
            skewC = np.array(skewC)#Change into np.ndarray in case we are using symbolic values
            tranSkewC = np.transpose(skewC)

            out11 = inertiaT + mass*np.dot( skewC,tranSkewC )
            out12 = mass*skewC
            out21 = mass*tranSkewC
            out22 = mass*np.identity(3)

            out =  np.array( np.bmat([[out11, out12],[out21, out22]]) )
        elif len(center) == 2:
            Izz = inertiaT
            out11 = Izz + mass*np.dot(center,center)
            out12 = -mass*center[1]
            out13 = mass*center[0]
            out21 = out12
            out22 = mass
            out23 = 0.0
            out31 = out13
            out32 = 0.0
            out33 = mass

            out = np.array( [[out11,out12,out13],[out21,out22,out23],[out31,out32,out33]] )
        else:
            print("Wrong dimensions")
            out = np.zeros((3,3))

        #If symbolic proceed using sympy
        if symbolic:
            return sym.Matrix(out)
        else:
            return out

    '''
    contactConstraints(): Create spanning matrix T for forces and for free motions S.
    Assumes contact is in the y-direction of the local frame.
    '''
    def contactConstraints(self, contactType, normalAxis='y', symbolic=False):

        #If symbolic proceed using sympy
        if symbolic:

            if contactType.lower() == "pointcontactwithoutfriction":
                if normalAxis.lower() == 'x':
                    T = sym.Matrix( [0.0,0.0,0.0,1.0,0.0,0.0] )
                    S = np.delete( np.identity(6), 3, 1 )
                elif normalAxis.lower() == 'y':
                    T = sym.Matrix( [0.0,0.0,0.0,0.0,1.0,0.0] )
                    S = np.delete( np.identity(6), 4, 1 )
                elif normalAxis.lower() == 'z':
                    T = sym.Matrix( [0.0,0.0,0.0,0.0,0.0,1.0] )
                    S = np.delete( np.identity(6), 5, 1 )
            elif contactType.lower() == "planarhardcontact":
                T = sym.Matrix( [[0.0,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0,0.0]] )
                T = T.T
                S = np.delete( np.identity(6), [3,4], 1 )
            else:
                print('Contact type not supported')
                T = [[0]]
                S = [[0]]

            S = sym.Matrix(S)

        #Using numeric values
        else:

            if contactType.lower() == "pointcontactwithoutfriction":
                if normalAxis.lower() == 'x':
                    T = np.array( [[0,0,0,1,0,0]] ).transpose()
                    S = np.delete( np.identity(6), 3, 1 )
                elif normalAxis.lower() == 'y':
                    T = np.array( [[0,0,0,0,1,0]] ).transpose()
                    S = np.delete( np.identity(6), 4, 1 )
                elif normalAxis.lower() == 'z':
                    T = np.array( [[0,0,0,0,0,1]] ).transpose()
                    S = np.delete( np.identity(6), 5, 1 )
            elif contactType.lower() == "planarhardcontact":
                T = np.array( [[0,0,0,1,0,0],[0,0,0,0,1,0]] ).transpose()
                S = np.delete( np.identity(6), [3,4], 1 )
            else:
                print('Contact type not supported')
                T = [[0]]
                S = [[0]]

        return T,S

    #def constrainedSubspace(self, constraintsInformation, Jacobians, beta, velocities, unit='rad'):
    def constrainedSubspace(self, constraintsInformation, Jacobians, unit='rad', symbolic=False):

        if self._modelCreated:

            nc = constraintsInformation['nc']
            dof = self.model['DoF']

            #Constraint matrix (nc times nDoF)
            A = np.zeros( (nc,dof) )

            #If symbolic proceed using sympy
            if symbolic:
                A = sym.Matrix( A )

            # Fill the constraint matrix. One row per constraint
            for i in range(nc):

                 # Body of the robot where the constraint is located
                constrainedBody = (constraintsInformation['body'])[i]
                # Contact point in body's local coordinates
                contactPoint = (constraintsInformation['contactPoint'])[i]
                # Is the contact to the left or right of the link?
                contactingSide = (constraintsInformation['contactingSide'])[i]
                # Angle that the normal direction of the contact has w.r.t. the link
                contactAngle = (constraintsInformation['contactAngle'])[i]
                if unit.lower() == 'deg':
                    contactAngle = math.radians(contactAngle)
                #type of constraint
                constraintType = (constraintsInformation['constraintType'])[i]
                #contact model
                contactModel = (constraintsInformation['contactModel'])[i]

                if constraintType.lower() == 'joint':
                    pass

                elif constraintType.lower() == 'non-slippage':
                    T,S = self.contactConstraints(contactModel, symbolic=symbolic)

                    if symbolic:
                        A[i,:] = ( T.T )*( self.xlt(contactPoint, symbolic=True) )*( Jacobians[constrainedBody] )
                    else:
                        A[i] = reduce(np.dot, [np.transpose(T), self.xlt(contactPoint), Jacobians[constrainedBody]])

                elif constraintType.lower() == 'non-slippagewithfriction':
                    T,S = self.contactConstraints(contactModel, symbolic=symbolic)

                    if symbolic:
                        A[i,:] = ( T.T )*( self.xlt(contactPoint, symbolic=True) )*( Jacobians[constrainedBody] )
                    else:
                        A[i] = reduce(np.dot, [np.transpose(T), self.xlt(contactPoint), Jacobians[constrainedBody]])

                elif constraintType.lower() == 'bodycontact':
                    T,S = self.contactConstraints(contactModel)

                    if contactingSide.lower() == 'left':
                        if symbolic:
                            A[i,:] = \
                            ( T.T )*( self.pluX(self.rz(contactAngle, symbolic=True), contactPoint, symbolic=True) )*( Jacobians[constrainedBody] )
                        else:
                            A[i] = reduce(np.dot, [np.transpose(T), self.pluX(self.rz(contactAngle), contactPoint), Jacobians[constrainedBody]])
                    elif contactingSide.lower() == 'right':
                        contactAngle = contactAngle + math.pi

                        if symbolic:
                            A[i,:] = \
                            ( T.T )*( self.pluX(self.rz(contactAngle, symbolic=True), contactPoint, symbolic=True) )*( Jacobians[constrainedBody] )
                        else:
                            A[i] = reduce(np.dot, [np.transpose(T), self.pluX(self.rz(contactAngle), contactPoint), Jacobians[constrainedBody]])

                    else:
                        print('Wrong side')

                else:
                        print('Wrong contraint type')

            #If symbolic proceed using sympy
            if symbolic:

                # How to get derivative of the constraint matrix? derConstraintMatrixA = d A/dt
                derA = A.diff(sym.symbols('t'))

                # kappa = np.dot( derConstraintMatrixA, velocities )
                kappa = derA*(sym.Matrix(self.dq))

            #Use numeric values
            else:

                # How to get derivative of the constraint matrix? derConstraintMatrixA = d A/dt
                derA = np.array([[0]])

                # kappa = np.dot( derConstraintMatrixA, velocities )
                kappa = np.array([[0]])

            # kappa_stab = beta*np.dot( constraintMatrix, velocities )
            beta = 0.1
            kappa_stab = beta*kappa

            #Return the constraint matrix and stabilization terms
            return A, kappa, kappa_stab
        else:
            print("Model not yet created. Use createModel() first.")
            return

    #Inverse dynamics. Given the state of the system (to the 2nd order), obtain the torques
    def ID(self, q, qd, qdd, fext = [], gravTerms=True, symbolic=False):

        # Only continue if createModel has been called
        if self._modelCreated:

            dof = self.model['DoF']
            nBodies = self.model['nB']

            #If symbolic proceed using sympy
            if symbolic:

                #change list into sym.Matrix
                q = sym.Matrix( q )
                qd = sym.Matrix( qd )
                qdd = sym.Matrix( qdd )
                fext = sym.Matrix( fext ) #Each row represents a wrench applied to a body

                Xup = list(sym.zeros(6,6) for i in range(nBodies))
                S = list(sym.zeros(6,1) for i in range(nBodies))
                X0 = list(sym.zeros(6,6) for i in range(nBodies))
                parentArray = self.model['parents']

                v = list(sym.zeros(6,1) for i in range(nBodies))
                a = list(sym.zeros(6,1) for i in range(nBodies))
                f = list(sym.zeros(6,1) for i in range(nBodies))

                tau = sym.zeros( dof,1 )
                #aGrav = self.model['inertialGrav']
                aGrav = sym.Matrix( self.model['inertialGrav'] )

                if fext.shape[0] == 0:
                    fext = list(sym.zeros(6,1) for i in range(nBodies))

                for i in range(nBodies):

                    XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i], symbolic=True)
                    S[i] = tempS
                    vJ = S[i]*qd[i]
                    Xup[i] =  XJ*( (self.model['XT'])[i] )

                    #If the parent is the base
                    if parentArray[i] == -1:
                        X0[i] = Xup[i]
                        v[i] = vJ

                        #Consider gravitational terms
                        if gravTerms:
                            a[i] = Xup[i]*(-aGrav) + S[i]*qdd[i]
                        else:
                            a[i] = S[i]*qdd[i]
                    else:
                        X0[i] = Xup[i]*X0[i-1]
                        v[i] = Xup[i]*v[parentArray[i]] + vJ
                        a[i] = Xup[i]*a[parentArray[i]] + S[i]*qdd[i] + self.crossM(v[i], symbolic=True)*vJ


                    RBInertia = (self.model['rbInertia'])[i]

                    #f[i] = np.dot( RBInertia, a[i] ) + reduce(np.dot, [self.crossF(v[i]), RBInertia, v[i]]) - np.dot( np.transpose( np.linalg.inv( X0[i] ) ), fext[i] )
                    f1 = RBInertia*a[i]
                    f2 = ( self.crossF(v[i], symbolic=True) )*( RBInertia )*( v[i] )
                    f3 = ( (self.invPluX(X0[i], symbolic=True)).T )*( fext[i] )

                    f[i] = f1 + f2 - f3

                for i in range(nBodies-1,-1,-1):
                    tau[i] = (S[i].T)*f[i]

                    #If the parent is not the base
                    if parentArray[i] != -1:
                        f[parentArray[i]] = f[parentArray[i]] + ( (Xup[i]).T )*(f[i] )

            #Otherwise, use numpy
            else:

                #change list into np.ndarray, if necessary
                if type(q) == list:
                    q = np.array( q )
                if type(qd) == list:
                    qd = np.array( qd )
                if type(q) == list:
                    qdd = np.array( qdd )
                if type(fext) == list:
                    fext = np.array( fext )

                Xup = np.zeros( (nBodies, 6, 6) )
                S = np.zeros( (nBodies, 6) )
                X0 = np.zeros( (nBodies, 6, 6) )
                parentArray = self.model['parents']

                v = np.zeros( (nBodies, 6) )
                a = np.zeros( (nBodies, 6) )
                f = np.zeros( (nBodies, 6) )

                tau = np.zeros( dof )
                aGrav = self.model['inertialGrav']

                if fext.size == 0:
                    fext = np.zeros( (nBodies, 6) )

                for i in range(nBodies):

                    XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i])
                    S[i] = tempS.flatten()#Make a flat array. This is easier when using numpy
                    vJ = S[i]*qd[i]
                    Xup[i] = np.dot( XJ, (self.model['XT'])[i] )

                    #If the parent is the base
                    if parentArray[i] == -1:
                        X0[i] = Xup[i]
                        v[i] = vJ
                        a[i] = np.dot( Xup[i], -aGrav ) + S[i]*qdd[i]
                    else:
                        X0[i] = np.dot( Xup[i], X0[i-1] )
                        v[i] = np.dot( Xup[i], v[parentArray[i]] ) + vJ
                        a[i] = np.dot( Xup[i], a[parentArray[i]] ) + S[i]*qdd[i] + np.dot( self.crossM(v[i]), vJ )

                    RBInertia = (self.model['rbInertia'])[i]

                    f1 = np.dot( RBInertia, a[i] )
                    f2 = reduce(np.dot, [self.crossF(v[i]), RBInertia, v[i]])
                    f3 = np.dot( np.transpose( np.linalg.inv( X0[i] ) ), fext[i] )

                    f[i] = f1 + f2 - f3
                    #f[i] = np.dot( RBInertia, a[i] ) + reduce(np.dot, [self.crossF(v[i]), RBInertia, v[i]]) - np.dot( np.transpose( np.linalg.inv( X0[i] ) ), fext[i] )

                for i in range(nBodies-1,-1,-1):
                    tau[i] = np.dot( S[i], f[i] )

                    #If the parent is not the base
                    if parentArray[i] != -1:
                        f[parentArray[i]] = f[parentArray[i]] + np.dot( np.transpose(Xup[i]), f[i] )

                #Convert into column vector. The change is in-place
                tau.resize( (dof,1) )

            #Return tau
            return tau

        else:
            print("Model not yet created. Use createModel() first.")
            return

    '''
    HandC(q,qd,fext,gravityTerms): Coefficients of the eqns. of motion.
    gravityTerms is a boolean variable to decide if gravitational terms should be included in the output.
    Set as False if only Coriolis/centripetal effects are desired.
    '''
    def HandC(self, q, qd, fext = [], gravityTerms = True, symbolic=False, simple=False):

        # Only continue if createModel has been called
        if self._modelCreated:

            dof = self.model['DoF']
            nBodies = self.model['nB']

            #If symbolic proceed using sympy
            if symbolic:

                #change list into sym.Matrix
                q = sym.Matrix( q )
                qd = sym.Matrix( qd )
                fext = sym.Matrix( fext ) #Each row represents a wrench applied to a body

                Xup = list(sym.zeros(6,6) for i in range(nBodies))
                S = list(sym.zeros(6,1) for i in range(dof))
                X0 = list(sym.zeros(6,6) for i in range(nBodies))
                parentArray = self.model['parents']

                v = list(sym.zeros(6,1) for i in range(nBodies))
                avp = list(sym.zeros(6,1) for i in range(nBodies))
                fvp = list(sym.zeros(6,1) for i in range(nBodies))
                Cor = sym.zeros( nBodies,1 )

                if gravityTerms:
                    #aGrav = self.model['inertialGrav'] # This does not work
                    aGrav = sym.Matrix( self.model['inertialGrav'] )
                else:
                    aGrav = sym.zeros(6,1)  # This works.

                if fext.shape[0] == 0:
                    fext = list(sym.zeros(6,1) for i in range(nBodies))

                # Calculation of Coriolis, centrifugal, and gravitational terms
                for i in range(nBodies):

                    XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i], symbolic=True)
                    S[i] = tempS
                    vJ = S[i]*qd[i]
                    Xup[i] =  XJ*( (self.model['XT'])[i] )

                    #If the parent is the base
                    if parentArray[i] == -1:
                        X0[i] = Xup[i]
                        v[i] = vJ
                        avp[i] = Xup[i]*(-aGrav)
                    else:
                        X0[i] = Xup[i]*X0[i-1]
                        v[i] = Xup[i]*v[parentArray[i]] + vJ
                        avp[i] = Xup[i]*avp[parentArray[i]] + self.crossM(v[i], symbolic=True)*vJ

                    RBInertia = (self.model['rbInertia'])[i]

                    f1 = RBInertia*avp[i]
                    f2 = ( self.crossF(v[i], symbolic=True) )*( RBInertia )*( v[i] )
                    f3 = ( (self.invPluX(X0[i], symbolic=True)).T )*( fext[i] )

                    fvp[i] = f1 + f2 - f3

                for i in range(nBodies-1,-1,-1):
                    Cor[i] = (S[i].T)*fvp[i]

                    #If the parent is not the base
                    if parentArray[i] != -1:
                        fvp[parentArray[i]] = fvp[parentArray[i]] + ( (Xup[i]).T )*fvp[i]

                IC = list(sym.zeros(6,6) for i in range(nBodies))
                for i in range(nBodies):
                    IC[i] = (self.model['rbInertia'])[i]

                for i in range(nBodies-1,-1,-1):
                    if parentArray[i] != -1:
                        IC[parentArray[i]] = IC[parentArray[i]] + ( ( Xup[i] ).T )*IC[i]*Xup[i]

                H = sym.zeros( nBodies )
                for i in range(nBodies):
                    fh = IC[i]*S[i]
                    H[i,i] = (S[i].T)*fh
                    j = i

                    while parentArray[j] > -1:
                        fh = ( (Xup[j]).T )*fh
                        j = parentArray[j]
                        H[i, j] = (S[j].T)*fh
                        H[j, i] = H[i, j]

                if simple:
                    H = sym.simplify(H)
                    Cor = sym.simplify(Cor)

            #Otherwise, use numpy
            else:
                #change list into np.ndarray, if necessary
                if type(q) == list:
                    q = np.array( q )
                if type(qd) == list:
                    qd = np.array( qd )
                if type(fext) == list:
                    fext = np.array( fext )

                Xup = np.zeros( (nBodies, 6, 6) )
                S = np.zeros( (dof, 6) )
                X0 = np.zeros( (nBodies, 6, 6) )
                parentArray = self.model['parents']

                v = np.zeros( (nBodies, 6) )
                avp = np.zeros( (nBodies, 6) )
                fvp = np.zeros( (nBodies, 6) )
                Cor = np.zeros( nBodies )

                if gravityTerms:
                    aGrav = self.model['inertialGrav']
                else:
                    aGrav = np.zeros( 6 )

                if fext.size == 0:
                    fext = np.zeros( (nBodies, 6) )

                # Calculation of Coriolis, centrifugal, and gravitational terms
                for i in range(nBodies):

                    XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i])
                    S[i] = tempS.flatten()#Make a flat array. This is easier when using numpy
                    vJ = S[i]*qd[i]
                    XT = (self.model['XT'])[i]

                    Xup[i] = np.dot( XJ, XT )

                    #If the parent is the base
                    if parentArray[i] == -1:
                        X0[i] = Xup[i]
                        v[i] = vJ
                        avp[i] = np.dot( Xup[i], -aGrav )
                    else:
                        X0[i] = np.dot( Xup[i], X0[i-1] )
                        v[i] = np.dot( Xup[i], v[parentArray[i]] ) + vJ
                        avp[i] = np.dot( Xup[i], avp[parentArray[i]] ) + np.dot( self.crossM(v[i]), vJ )

                    RBInertia = (self.model['rbInertia'])[i]

                    f1 = np.dot( RBInertia, avp[i] )
                    f2 = reduce(np.dot, [self.crossF(v[i]), RBInertia, v[i]])
                    f3 = np.dot( np.transpose( np.linalg.inv( X0[i] ) ), fext[i] )

                    fvp[i] = f1 + f2 - f3

                for i in range(nBodies-1,-1,-1):
                    Cor[i] = np.dot( S[i], fvp[i] )

                    #If the parent is not the base
                    if parentArray[i] != -1:
                        fvp[parentArray[i]] = fvp[parentArray[i]] + np.dot( np.transpose(Xup[i]), fvp[i] )

                IC = np.zeros( (nBodies, 6, 6) )
                for i in range(nBodies):
                    IC[i] = (self.model['rbInertia'])[i]

                for i in range(nBodies-1,-1,-1):
                    if parentArray[i] != -1:
                        IC[parentArray[i]] = IC[parentArray[i]] + reduce(np.dot, [np.transpose( Xup[i] ), IC[i], Xup[i]])

                H = np.zeros( (nBodies, nBodies) )
                for i in range(nBodies):
                    fh = np.dot( IC[i], S[i] )
                    H[i,i] = np.dot( S[i], fh )
                    j = i

                    while parentArray[j] > -1:
                        fh = np.dot( np.transpose(Xup[j]), fh )
                        j = parentArray[j]
                        H[i, j] = np.dot( S[j], fh )
                        H[j, i] = H[i, j]

                #Convert into column vector. The change is in-place
                Cor.resize( (nBodies,1) )

            #Return the inertia matrix and coriolis effects
            return H,Cor

        else:
            print("Model not yet created. Use createModel() first.")
            return

    '''
    FDcrb(q, qd, tau, fext): Composite-rigid-Body algorithm
    '''
    def FDcrb(self, q, qd, tau, fext = [], symbolic=False):

        localH, localCor = self.HandC(q, qd, fext, gravityTerms=True, symbolic=symbolic)

        if symbolic:
            tau = sym.Matrix(tau)
            RHS = tau - localCor
            ddq = localH.LUsolve(RHS)
        else:
            tau = np.array(tau)
            tau.resize( (tau.size,1) )
            RHS = tau - localCor
            ddq = np.linalg.solve( localH, RHS )

        return ddq

    '''
    FDab(q, qd, tau, fext, gravityTerms):
    '''
    def FDab(self, q, qd, tau, fext = [], gravityTerms=True, symbolic=False):

        # Only continue if createModel has been called
        if self._modelCreated:

            dof = self.model['DoF']
            nBodies = self.model['nB']

            if symbolic:
                #change list into sym.Matrix
                q = sym.Matrix( q )
                qd = sym.Matrix( qd )
                fext = sym.Matrix( fext ) #Each row represents a wrench applied to a body

                Xup = list(sym.zeros(6,6) for i in range(nBodies))
                S = list(sym.zeros(6,1) for i in range(dof))
                X0 = list(sym.zeros(6,6) for i in range(nBodies))
                parentArray = self.model['parents']

                v = list(sym.zeros(6,1) for i in range(nBodies))
                c = list(sym.zeros(6,1) for i in range(nBodies))

                if gravityTerms:
                    #aGrav = self.model['inertialGrav']
                    aGrav = sym.Matrix( self.model['inertialGrav'] )
                else:
                    aGrav = sym.zeros(6,1)

                IA = list(sym.zeros(6,6) for i in range(nBodies))
                pA = list(sym.zeros(6,1) for i in range(nBodies))

                if fext.shape[0] == 0:
                    fext = list(sym.zeros(6,1) for i in range(nBodies))

                # Calculation of Coriolis, centrifugal, and gravitational terms
                for i in range(nBodies):

                    XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i], symbolic=True)
                    S[i] = tempS
                    vJ = S[i]*qd[i]
                    XT = (self.model['XT'])[i]

                    Xup[i] =  XJ*XT

                    #If the parent is the base
                    if parentArray[i] == -1:
                        X0[i] = Xup[i]
                        v[i] = vJ
                    else:
                        X0[i] = Xup[i]*X0[i-1]
                        v[i] = Xup[i]*v[parentArray[i]] + vJ
                        c[i] = self.crossM(v[i], symbolic=True)*vJ

                    RBInertia = (self.model['rbInertia'])[i]
                    IA[i] = sym.Matrix( RBInertia )

                    pA1 = ( self.crossF(v[i], symbolic=True) )*( IA[i] )*( v[i] )
                    pA2 = ( (self.invPluX(X0[i], symbolic=True)).T )*( fext[i] )

                    pA[i] = pA1 - pA2

                # 2nd pass. Calculate articulated-body inertias
                U = list(sym.zeros(6,1) for i in range(nBodies))
                d = sym.zeros( nBodies,1 )
                u = sym.zeros( nBodies,1 )

                for i in range(nBodies-1,-1,-1):

                    U[i] = IA[i]*S[i]
                    d[i] = ( S[i].T )*U[i]
                    #u[i] = tau[i] - ( S[i].T )*pA[i]
                    u[i] = tau[i] - S[i].dot(pA[i])

                    #invd = 1.0/(d[i] + np.finfo(float).eps)
                    invd = 1.0/d[i]

                    #If the parent is the base
                    if parentArray[i] != -1:
                        Ia = IA[i] - invd*U[i]*(U[i].T)
                        pa = pA[i] + Ia*c[i] + invd*u[i]*U[i]

                        IA[parentArray[i]] = IA[parentArray[i]] + (Xup[i].T)*Ia*Xup[i]
                        pA[parentArray[i]] = pA[parentArray[i]] + (Xup[i].T)*pa

                # 3rd pass. Calculate spatial accelerations
                a = list(sym.zeros(6,1) for i in range(nBodies))
                qdd = sym.zeros(dof,1)

                for i in range(nBodies):

                    invd = 1.0/d[i]

                    #If the parent is the base
                    if parentArray[i] == -1:
                        a[i] = Xup[i]*(-aGrav) + c[i]
                    else:
                        a[i] =  Xup[i]*a[parentArray[i]] + c[i]

                    #qdd[i] = invd*( u[i] - (U[i].T)*a[i] )
                    qdd[i] = invd*( u[i] - U[i].dot(a[i]) )
                    a[i] = a[i] + S[i]*qdd[i]

            #Use numpy for numeric computations
            else:
                #change list into np.ndarray, if necessary
                if type(q) == list:
                    q = np.array( q )
                if type(qd) == list:
                    qd = np.array( qd )
                if type(fext) == list:
                    fext = np.array( fext ) #Each row represents a wrench applied to a body

                Xup = np.zeros( (nBodies, 6, 6) )
                S = np.zeros( (dof, 6) )
                X0 = np.zeros( (nBodies, 6, 6) )
                parentArray = self.model['parents']
                v = np.zeros( (nBodies, 6) )
                c = np.zeros( (nBodies, 6) )

                if gravityTerms:
                    aGrav = self.model['inertialGrav']
                else:
                    aGrav = np.zeros( 6 )

                IA = np.zeros( (nBodies, 6, 6) )
                pA = np.zeros( (nBodies, 6) )

                if fext.size == 0:
                    fext = np.zeros( (nBodies, 6) )

                # Calculation of Coriolis, centrifugal, and gravitational terms
                for i in range(nBodies):

                    XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i])
                    S[i] = tempS.flatten()#Make a flat array. This is easier when using numpy
                    #S[i] = tempS
                    vJ = S[i]*qd[i]
                    XT = (self.model['XT'])[i]

                    Xup[i] = np.dot( XJ, XT )

                    #If the parent is the base
                    if parentArray[i] == -1:
                        X0[i] = Xup[i]
                        v[i] = vJ
                        #c[i] = np.zeros( 6 )
                    else:
                        X0[i] = np.dot( Xup[i], X0[i-1] )
                        v[i] = np.dot( Xup[i], v[parentArray[i]] ) + vJ
                        c[i] = np.dot( self.crossM(v[i]), vJ )

                    RBInertia = (self.model['rbInertia'])[i]
                    IA[i] = RBInertia

                    pA1 = reduce(np.dot, [self.crossF(v[i]), IA[i], v[i]])
                    pA2 = np.dot( np.transpose( np.linalg.inv( X0[i] ) ), fext[i] )

                    pA[i] = pA1 - pA2

                # 2nd pass. Calculate articulated-body inertias
                U = np.zeros( (nBodies, 6) )
                d = np.zeros( nBodies )
                u = np.zeros( nBodies )

                for i in range(nBodies-1,-1,-1):

                    U[i] = np.dot( IA[i], S[i] )
                    d[i] = np.dot( S[i], U[i] )
                    u[i] = tau[i] - np.dot( S[i], pA[i])

                    #invd = 1.0/(d[i] + np.finfo(float).eps)
                    invd = 1.0/d[i]

                    #If the parent is the base
                    if parentArray[i] != -1:
                        Ia = IA[i] - invd*np.outer( U[i], U[i] )
                        pa = pA[i] + np.dot( Ia, c[i] ) + invd*u[i]*U[i]

                        IA[parentArray[i]] = IA[parentArray[i]] + reduce( np.dot, [np.transpose(Xup[i]), Ia, Xup[i]] )
                        pA[parentArray[i]] = pA[parentArray[i]] + np.dot( np.transpose(Xup[i]), pa )

                # 3rd pass. Calculate spatial accelerations
                a = np.zeros( (nBodies, 6) )
                qdd = np.zeros( dof )

                for i in range(nBodies):

                    invd = 1.0/d[i]

                    #If the parent is the base
                    if parentArray[i] == -1:
                        a[i] = np.dot( Xup[i], -aGrav ) + c[i]
                    else:
                        a[i] = np.dot( Xup[i], a[parentArray[i]] ) + c[i]

                    qdd[i] = invd*(u[i] - np.dot( U[i], a[i] ))
                    a[i] = a[i] + S[i]*qdd[i]

                #Convert into column vector. The change is in-place
                qdd.resize( (dof,1) )

            #Return the acceleration of the joints
            return qdd

        else:
            print("Model not yet created. Use createModel() first.")
            return

#--------------Until here, both symbolic and numeric versions [2017-09-15]-----------------

    #Absolute angles (assuming open serial kinematic chain). angles is the angles of the joints
    @staticmethod
    def absAngles(angles):

        #Change type if necessary
        if type(angles) == list:
            angles = np.array(angles)
        absangles = np.zeros( angles.size )

        for i in range(angles.size):
            absangles[i] = np.sum( angles[0:i+1] )

        return absangles

    #Change dimensions, from planar to spatial, or spatial to planar
    def dimChange(self, quantity):

        #Change into np.ndarray if necessary
        if type(quantity) == list:
            quantity = np.array(quantity)

        #If the dimensions is 3, then we received a planar vector (twist or wrench). Convert to spatial vector
        if quantity.shape == (3,1):
            quantity = np.insert( quantity, 0, [[0.0],[0.0]], axis=0 )
            quantity = np.append(quantity, [[0.0]], axis=0)
        elif quantity.shape == (3,):
            quantity = np.insert(quantity, 0, [0.0,0.0])
            quantity = np.append(quantity, 0)
        elif quantity.shape == (1,3):
            quantity = np.insert(quantity, 0, [[0.0],[0.0]], axis=1)
            quantity = np.append(quantity,[[0.0]], axis=1)
        #If the dimensions is 3by3, then we received a 3D Matrix
        elif quantity.shape == (3,3):
#            temp = np.zeros( (6,6) )
#            temp[2:5,2:5] = quantity
#            quantity = temp
            quantity = np.insert(quantity, 0, [[0.0],[0.0]], axis=0)
            quantity = np.append(quantity, np.zeros((1,3)), axis=0)

            quantity = np.insert(quantity, 0, [[0.0],[0.0]],axis=1)
            quantity = np.append(quantity, np.transpose([[0.0,0.0,0.0,0.0,0.0,0.0]]), axis=1)
        #If the dimensions is 6, then we received a spatial vector (twist or wrench). Convert to planar vector
        elif quantity.shape == (6,1) or quantity.shape == (6,):
            quantity = quantity[2:5]
        elif quantity.shape == (1,6):
            temp = quantity[0,2:5]
            #quantity = np.reshape( quantity[2:5],(1,3) )
            quantity = np.reshape( temp,(1,3) )
        #If the dimensions is 6by6, then we received a 6D Matrix
        elif quantity.shape == (6,6):
            quantity = quantity[2:5,2:5]

        return quantity

    # Transform a symbolic Matrix into a array (with float type values)
    # input array is a Matrix object and
    #substitutions is either a tuple (oldvalue, new value) or a list of tuples [(old, new), ..., (old, new)]
    def symToNum(self, inputArray, substitutions):

        if type(substitutions) is not list:
            substitutions = [substitutions]

        output = inputArray.subs( substitutions )
        output = np.array( output ).astype(np.float)

        return output

    # Make a list of tuples from two lists/arrays
    # TODO: Test in Python 3
    def tuplesFromLists(self, x, y):

        # If both inputs are lists
        if (type(x) is list) and (type(y) is list):
            if len(x) == len(y):
                return zip(x, y)
            else:
                print("x and y should have the same number of elements")

        elif (type(x) is np.ndarray) and (type(y) is np.ndarray):
            if x.size == y.size:
                return zip(x, y)
            else:
                print("x and y should have the same number of elements")
        else:
            return (x,y)

    # Obtain a numeric expression from two lists
    # x should be symbols, y should be numeric values
    def numExp(self, expr, x, y):
        subs = self.tuplesFromLists(x,y)
        return self.symToNum(expr, subs)

    # Create a function to return scalar metric from a vector and metric tensor <vec1, M vec2>
    # vec1 and vec2 are vectors
    # M is an optional metric tensor
    # simple is a flag to indicate in simplification of the expression is desired
    @staticmethod
    def innerProduct(vec1, M=[], vec2=None, symbolic=False, simple=False):

        if symbolic:

            vec1 = sym.Matrix( vec1 )
            M = sym.Matrix( M )
            if vec2 == None:
                vec2 = vec1
            else:
                vec2 = sym.Matrix( vec2 )

            n = vec1.shape[0] # number of elements
            if M.shape[0] == 0:
                M = sym.eye(n)

            if simple is True:
                return sym.simplify(vec1.dot(M*vec2))
            else:
                return vec1.dot(M*vec2)

        else:

            vec1 = np.array( vec1 )
            M = np.array( M )
            if vec2 == None:
                vec2 = vec1
            else:
                vec2 = np.array( vec2 )

            n = vec1.shape[0] # number of elements
            if M.shape[0] == 0:
                M = np.identity(n)

            return np.dot( vec1, np.dot(M, vec2) )

    # Save an internal copy of inertia matrix
    def saveInertiaMatrix(self, M):
        self.M = sym.ImmutableMatrix(M)

    # Create a matrix from a list of four sub-block Matrices
    @staticmethod
    def createBlockMatrix(blocks, symbolic=False):
        if symbolic:

            m11 = sym.Matrix( blocks[0][0] )
            m12 = sym.Matrix( blocks[0][1] )
            m21 = sym.Matrix( blocks[1][0] )
            m22 = sym.Matrix( blocks[1][1] )

            if (m11.shape[0] == m12.shape[0]) and  (m21.shape[0] == m22.shape[0]) and (m11.shape[1] == m21.shape[1]) and (m12.shape[1] == m22.shape[1]):
                m1 = m11.row_join( m12)
                m2 = m21.row_join( m22)
                return m1.col_join(m2)
            else:
                print("Incompatible dimensions")
                return []
        else:

            m11 = np.array( blocks[0][0] )
            m12 = np.array( blocks[0][1] )
            m21 = np.array( blocks[1][0] )
            m22 = np.array( blocks[1][1] )

            if (m11.shape[0] == m12.shape[0]) and  (m21.shape[0] == m22.shape[0]) and (m11.shape[1] == m21.shape[1]) and (m12.shape[1] == m22.shape[1]):
                m1 = np.concatenate((m11, m12), axis=1)
                m2 = np.concatenate((m21, m22), axis=1)
                return np.concatenate((m1, m2), axis=0)
            else:
                print("Incompatible dimensions")
                return []


    # Projection onto the constrained subspace C
    # T is the matrix that spans the constrained subspace
    # I is the metric tensor
    @staticmethod
    def projectT(T, I, method='GE', symbolic=False, simple=False):
        if symbolic:
            T = sym.Matrix( T )
            I = sym.Matrix( I )

            TIT = (T.T)*I*T
            invTIT = TIT.inv(method=method)

            if simple:
                return sym.simplify( invTIT*(T.T)*I )
            else:
                return invTIT*(T.T)*I

        else:
            T = np.array( T )
            I = np.array( I )

            TIT = reduce(np.dot, [np.transpose(T), I, T])
            invTIT = np.linalg.inv( TIT )
            return reduce(np.dot, [invTIT, np.transpose(T), I])

    #Oscillator. t:time. i:joint number.
    @staticmethod
    def oscillator(t, i, alpha, beta, gamma, omega):
        phi = alpha*(math.sin(omega*t + (i-1)*beta)) + gamma
        return phi

    @staticmethod
    #derivative of an oscillator
    def oscillator_d(t,i,alpha, beta, gamma, omega):
        phi_d = alpha*omega*(math.cos(omega*t + (i-1)*beta))
        return phi_d

    @staticmethod
    #second derivative of an oscillator
    def oscillator_dd(t,i,alpha, beta, gamma, omega):
        phi_dd = -alpha*omega*omega*(math.sin(omega*t + (i-1)*beta))
        return phi_dd

    #Serpenoid curve. t:time. i:joint number.
    def serpenoidCurve(self, t, noJoints, alpha, beta, gamma, omega):
        phi = [self.oscillator(t, i, alpha, beta, gamma, omega) for i in range(1, noJoints+1)]
        return phi

    #derivative of a serpenoid curve
    def serpenoidCurve_d(self, t, noJoints, alpha, beta, gamma, omega):
        phi_d = [self.oscillator_d(t, i, alpha, beta, gamma, omega) for i in range(1, noJoints+1)]
        return phi_d

    #second derivative of a serpenoid curve
    def serpenoidCurve_dd(self, t, noJoints, alpha, beta, gamma, omega):
        phi_dd = [self.oscillator_dd(t, i, alpha, beta, gamma, omega) for i in range(1, noJoints+1)]
        return phi_dd

# ------------Mathematical manipulation----------------

    #convert from an array of (unsigned) integers to numerical values. Input should be one dimensional
    #If the input is bytearray then 'bytearrayToInt' is called first
    def mapToValues(self, data, in_min, in_max, out_min, out_max, noBytes=1):

        #convert bytearray to a np.array containing integers
        if type(data) == bytearray:
            data = self.bytearrayToInt(data, noBytes)

        values = [(x-in_min)*(out_max-out_min)/(in_max-in_min)+out_min for x in data]
        return np.array(values)

# ------------DRAWING------------------

    # Given a list of transformations and pts, transform them
    # transformation is a list of o transformations. Each transformation is (m,n)
    # pts is a list of lists, or list of arrays. Each element represents a set (n,p) of pts (n,1) to be transformed
    # If several points are transformed at the same time, give them as a matrix (each row represents a vector)
    def worldPts(self, trans, inPts, symbolic=False):
        return [ self.Xpts(trans[i], pts, symbolic=symbolic) for i,pts in enumerate(inPts) ]

    # using forward kinematics mappings, calculate all origins.
    # invX0 is a list of transformations. invX0[i] transforms from frame-i to frame-0
    def origins(self, invX0):

        if type(invX0) == list:
            invX0 = np.array(invX0)
        depth = invX0.shape[0]

        zero = [ np.zeros( (1,3) ) for i in range(depth) ]
        origins = [ np.zeros( (1,3) ) for i in range(depth+1) ]

        # Calculate the origins of each link and return as matrix (each row represents a origin)
        origins[1:] = self.worldPts(invX0, zero)
        origins = np.reshape( origins, (depth+1,3) )

        # Calculate end of last link
        end = np.reshape(self.worldPts(invX0[-1:], [[self.model['lengths'][-1],0.0,0.0]]), (1,3) )
        origins = np.append(origins, end, axis=0)

        return origins

    def drawSkeleton(self, invX0, autoS=True, xlim=None, ylim=None):

        # Total length
        totalLength = sum( self.model['lengths'] )
        scale = self.mapToValues( [1,5,10], 0.0, 100.0, 0.0, totalLength ) # Scale representin 1%,5%, and 10% of the snake robot's length

        # Obtain all the origins
        origins3D = self.origins(invX0)

        # Remove a dimension
        origins2D = np.delete( origins3D, 2, 1)

        # Remove origin and virtual links
        if self.model['floatingBase']:
            origins2D = np.delete( origins2D, [0,1,2], 0)

        fig = plt.figure()
        axes = fig.add_subplot(1,1,1)
        #patch = []

        skeleton = Polygon(origins2D, closed=False, fill=False, edgecolor='#677a04', lw=3, alpha=0.6)
        #skeleton = plt.Polygon(origins2D, closed=None, fill=None, edgecolor='r')

        axes.add_patch(skeleton)
        #patch.append(skeleton)

        # Obtain COM
        COMs = self.worldPts(invX0, self.model['com'])
        COMs = np.delete( COMs, 2, 1)
        if self.model['floatingBase']:
            COMs = np.delete( COMs, [0,1], 0)

        for i in COMs:
            circle = Circle(i, scale[0], facecolor='#fffe40')
            #patch.append(circle)
            axes.add_patch(circle)

        # Joints
        joints = np.delete( origins3D, 2, 1)
        joints = np.delete( origins3D, -1, 0)
        if self.model['floatingBase']:
            joints = np.delete( joints, [0,1,2,3], 0)
        else:
            joints = np.delete( joints, [0], 0)

        for i in joints:
            #circle = Circle(i, 2*scale[0], fill=False)
            circle = Circle(i, 2*scale[0], facecolor='#aa2704', alpha=0.6)
            #patch.append(circle)
            axes.add_patch(circle)

        # Add the patches to the axes
#        p = PatchCollection(patch, alpha=0.4)
#        colors = 100*np.random.rand(len(patch))
#        p.set_array(np.array(colors))
#
#        axes.add_collection(p)

#        if autoS:
#            axes.autoscale_view(True,True,True)
#        else:
#            if xlim is not None:
#                axes.set_xlim(xlim[0], xlim[1])
#            if ylim is not None:
#                axes.set_ylim(ylim[0], ylim[1])

        axes.set_alpha(0.1)
        axes.axis("equal")

        plt.show()


# --------------------PARSING-----------------------

    # Parse a string into a sympy matrix
    def toPython(self, inputString, oldValues=None, newValues=None, transpose=False):

        # Change type
        inputString = self.stringToList(inputString)

        if type(inputString) is not list:
            inputString = [inputString]

        # Separate expression
        expr = [s.split(",") for s in inputString]

        # Convert into a list of sympy expressions
        expr = [ [parse_expr(x) for x in s] for s in expr]

        # Convert into a sympy matrix
        expr = sym.Matrix(expr)

        # Transpose, if we have a vector
        if transpose:
            expr = expr.T

        # Make substitutions and return
        if oldValues is not None and newValues is not None:
            subs = self.tuplesFromLists(oldValues, newValues)
            return self.symToNum(expr, subs)
        else:
            return expr

    # Parse a string of a list(or matrix) into a list of strings. Each element of the string is a row of a matrix
    def stringToList(self, inputString):

        # Remove double left brackets and double right brackets
        inputString = inputString.replace('[[','').replace(']]','').replace('{{','').replace('}}','')

#        repls = ('[[',''),(']]',''),('{{',''),('}}','')
#        inputString = reduce(lambda a, kv: a.replace(*kv), repls, inputString)

        # Replace middle ][ brackets with semicolon and convert into list
        inputString = inputString.replace('],[',';').replace('][',';').replace('},{',';').replace('}{',';')

#        repls = ('],[',';'),('][',';'),('},{',';'),('}{',';')
#        inputString = reduce(lambda a, kv: a.replace(*kv), repls, inputString)

        # Remove single left and right brackets
        inputString = inputString.replace('[','').replace(']','').replace('{','').replace('}','')

        return inputString.split(";")

# Wrapper functions for symbolic expressions
# Outputs are adequate for scipy.optimize
class LambdaWrapper(object):

    # Returns a lambda-type expression
    # expr is a sympy expression
    # x is a tuple of main symbols to be substituted (e.g., symbols of an input vector)
    # y is a tuple of auxiliary symbols (e.g., symbols inside a matrix)
    @staticmethod
    def fun_l(expr, x, y=()):
        return sym.lambdify(x+y, expr)


    # Returns a list of lambda-type expressions
    # expr is a sympy expression (supposed to be a list or a column matrix)
    # x is a tuple of main symbols to be substituted (e.g., symbols of an input vector)
    # y is a tuple of auxiliary symbols (e.g., symbols inside a matrix)
    @staticmethod
    def funcs_l(expr, x, y=()):
        return [f for f in expr], [sym.lambdify(x+y, f) for f in expr]


    # Returns symbolic gradient, and a list of lambda-type expression, corresponding to the gradient (w.r.t. main symbols)
    # expr is a sympy expression (scalar)
    # x is a tuple of main symbols to be substituted (e.g., symbols of an input vector)
    # y is a tuple of auxiliary symbols (e.g., symbols inside a matrix)
    @staticmethod
    def grad_l(expr, x, y=()):
        grad_sym = [expr.diff(var) for var in x]
        grad_lambda = [sym.lambdify(x+y, grad) for grad in grad_sym]
        return grad_sym, grad_lambda


    # Returns list of symbolic Jacobian, and a list of lambda-type expression, corresponding to the jacobians
    # expr is a list of sympy expressions
    # x is a tuple of main symbols to be substituted (e.g., symbols of an input vector)
    # y is a tuple of auxiliary symbols (e.g., symbols inside a matrix)
    @classmethod
    def jac_l(cls, expr, x, y=()):
        gradients = [cls.grad_l(f, x, y) for f in expr]
        return [a for a, b in gradients], [b for a, b in gradients]


    # General wrapper for lambda expressions
    # x and y are tuples
    # funType is a string indicating the type of expr
    @classmethod
    def wrapper_l(cls, expr, x, y=(), funType="scalar"):
        if funType.lower() == "scalar":
            return expr, cls.fun_l(expr, x, y)
        elif funType.lower() == "vector":
            return cls.funcs_l(expr, x, y)
        elif funType.lower() == "gradient":
            return cls.grad_l(expr, x, y)
        elif funType.lower() == "jacobian":
            return cls.jac_l(expr, x, y)
        else:
            print("Incorrect type of function")
            return [], []


    # Wrapper for lambda expressions
    # Return is either a evaluated expression, or list of evaluated expressions
    # l_expr is a lambdified expression (e.g., output from fun_l) or list of expressions
    # x is a list of numeric values
    # y is a list of auxiliary numeric values
    @staticmethod
    def fun_v(l_expr, x, y=[]):
        x = np.array(x)
        y = np.array(y)
        args = np.append(x, y)

        if type(l_expr) is list:
            return np.array([expr(*tuple(args)) for expr in l_expr])
        else:
            return l_expr(*tuple(args))


    # Returns a list of dictionaries of constraints
    # This function assumes the secondary symbols y are the same for all constraints/Jacobians
    @classmethod
    def con_v(cls, l_expr, y=[], l_jac=[], type_con=[]):
        keys = ['fun', 'jac', 'type', 'args']

        if type(l_expr) is list:
            con_dic = []
            if len(l_expr) == len(l_jac) == len(type_con):
                for i in range(len(l_expr)):
                    def f_v(x, y=[], i=i):
                        return cls.fun_v(l_expr[i], x, y)

                    if l_jac[i] != None:
                        def j_v(x, y=[], i=i):
                            return cls.fun_v(l_jac[i], x, y)
                    else:
                        j_v = None

                    values = [f_v, j_v, type_con[i], y[i]]
                    local_dic = dict(zip(keys, values))

                    # Remove empty entries
                    if sys.version_info[0] < 3:
                        local_dic = dict((k, v) for k, v in local_dic.iteritems() if v)
                    else:
                        local_dic = {k: v for k, v in local_dic.items() if v}
                    con_dic.append(local_dic)
            else:
                print("Incorrect length of arguments")
        else:
            def f_v(x, y=[], i=-1):
                return cls.fun_v(l_expr, x, y)

            def j_v(x, y=[], i=-1):
                return cls.fun_v(l_jac, x, y)

            values = [f_v, j_v, type_con, y]
            con_dic = dict(zip(keys, values))

            # Remove empty entries
            if sys.version_info[0] < 3:
                con_dic = dict((k, v) for k, v in con_dic.iteritems() if v)
            else:
                con_dic = {k: v for k, v in con_dic.items() if v}

        return con_dic


    # Wrapper for minimize function
    # cost_fun is a symbolic scalar function
    # const_fun is a list of [constraint functions, args, type of constraints]
    # Returns (function cost, vector) if successful
    # TODO: add constraints
    @classmethod
    def min_max(cls, cost_fun, x, x_0, const_fun=None, y=(), y_0=[], obj='min', **kwargs):
        _,cost_fun_sym = cls.wrapper_l(cost_fun, x, y, funType='scalar')
        _,jac_sym = cls.wrapper_l(cost_fun, x, y, funType='gradient')

        def local_cost(x, y=[], sign=1.0, i=-1):
            return sign * (cls.fun_v(cost_fun_sym, x, y))
        def local_jac(x, y=[], sign=1.0, i=-1):
            return sign * (cls.fun_v(jac_sym, x, y))

        if const_fun == None:
            cons = []
        else:
            cons = []
        #     _, con2_lambda = wrapper_l(const_fun[0], x, y=y, funType="vector")
        #     _, jac_con2 = wrapper_l(const_fun[0], x, y=y, funType="jacobian")
        #     cons = con_v(con2_lambda, y=const_fun[1], l_jac=jac_con2, type_con=const_fun[2])

        if obj == 'min':
            obj_sign = 1.0
        elif obj == 'max':
            obj_sign = -1.0
        else:
            obj_sign = 1.0

        sol = optimize.minimize(local_cost, x_0, args=(y_0, obj_sign), jac=local_jac, constraints=cons, **kwargs)

        if sol.success:
            return sol.fun, sol.x
        else:
            print("Failure")
            return 0, []
