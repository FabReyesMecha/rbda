# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 03:09:34 2017

RBDA v0.7

Small module containing Algorithms for calculating twistes and wrenches.
This module is just a small portion of algorithms available on:
    Rigid Body Dynamic Algorithms, by Featherstone, Roy (2008)
    
The latest version should be in ../moduletest/

Usage:

Download rbda.py into your computer. For example, if the file is in the directory
'c:/users/reyes fabian/my cubby/python/python_v-rep/moduletest', then use the following code:

import os
os.chdir('c:/users/reyes fabian/my cubby/python/python_v-rep/moduletest')
os.chdir('c:/users/reyes fabian/my cubby/python/[modules]/rbda')

import rbda
plu = rbda.RBDA()

#Start using the functions. Example:
plu.rx(45, unit='deg')

----------------------------------------------------------------------------
Log:


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

from sympy.physics.vector import dynamicsymbols
q1 = dynamicsymbols('q1')
q2 = dynamicsymbols('q2')
func = q1 + 2*q2
derFunc = sympy.diff(func, sympy.Symbol('t') )

#or

theta = dynamicsymbols('theta')
rot = plu.rz(theta)

#This does NOT work
derRot = sympy.diff(rot, sympy.Symbol('t') )

#This works
derRot = rot.diff( sympy.Symbol('t') )

@author: Reyes Fabian
"""

import numpy
import math
import copy #for deepcopy
#import numbers

import sympy
from sympy.physics.vector import dynamicsymbols
from sympy.parsing.sympy_parser import parse_expr
#from sympy import diff, Symbol

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection

class RBDA(object):

    #Constructor
    def __init__(self):
        self.__version__ = '0.7.0'
        self._modelCreated = False
        
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
        self.time = sympy.symbols('t')
        
        # Option 1: Array of symbols. However, they do not depend on time
#        self.qVector = sympy.symarray( 'qVector', self.model['DoF'] )
#        self.dq = sympy.symarray( 'dq', self.model['DoF'] )

        # Option 2: dynamic symbols and return a list. Easiest to manipulate afterwards
        qVector = ['q_'+str(i).zfill(2) for i in range(self.model['DoF'])]
        dq = ['dq_'+str(i).zfill(2) for i in range(self.model['DoF'])]
        ddq = ['ddq_'+str(i).zfill(2) for i in range(self.model['DoF'])]
        
        self.qVector = dynamicsymbols(qVector)
        self.dq = dynamicsymbols(dq)
        self.ddq = dynamicsymbols(ddq)
        
        #1st order derivatives of the joint angles
        self.qDer = [sympy.diff(var, self.time) for var in self.qVector]
        self.qDerDer = [sympy.diff(var, self.time, 2) for var in self.qVector]
        
        #tuples for substituting Derivative(q(t),t) for dq
        self.stateSym = zip(self.qDer, self.dq) + zip(self.qDerDer, self.ddq)
        
        # Option 3: return a numpy array
#        self.qVector = numpy.array( dynamicsymbols(qVector) )
#        self.dq = numpy.array( dynamicsymbols(dq) )
#        self.ddq = numpy.array( dynamicsymbols(ddq) )
        
        #For now, make the actuator torques not dependent on time
        self.tauAct = sympy.symarray( 'tauAct', self.model['nA'] )

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
            self.model['inertialGrav'] = numpy.array( [0.0,0.0,0.0,-(self.model['g']),0.0,0.0] )
        elif self.model['gravityAxis'].lower() == 'y':
            self.model['inertialGrav'] = numpy.array( [0.0,0.0,0.0,0.0,-(self.model['g']),0.0] )
        elif self.model['gravityAxis'].lower() == 'z':
            self.model['inertialGrav'] = numpy.array( [0.0,0.0,0.0,0.0,0.0,-(self.model['g'])] )
        else:
            print('Gravity direction not recognized. Assuming it is in the y-axis.')
            self.model['inertialGrav'] = numpy.array( [0.0,0.0,0.0,0.0,-(self.model['g']),0.0] )
        
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
        print('Model created')
    
    #three-dimensional rotation around the x axis. Works numerically or symbollicaly
    #IF using a symbolic variable, a numeric matrix can be obtained as rx(theta).subs(theta, 1.0), for example.
    @staticmethod
    def rx(theta, unit='rad', symbolic=False):
        
        #If symbolic, then solve using sympy
        if symbolic:
            
            if isinstance(theta, tuple(sympy.core.all_classes)):
                c_theta = sympy.cos(theta)
                s_theta = sympy.sin(theta)
    
            else:    
                if unit.lower() == 'deg':
                    theta=math.radians(theta)
    
                c_theta = math.cos(theta)
                s_theta = math.sin(theta)
            
            return sympy.Matrix( [[1.0,0.0,0.0],[0.0,c_theta,s_theta],[0.0,-s_theta,c_theta]] )
        
        #else, if it is numeric use numpy
        else:
            
            if unit.lower() == 'deg':
                theta = math.radians(theta)

            c_theta = math.cos(theta)
            s_theta = math.sin(theta)

            return numpy.array([[1.0,0.0,0.0],[0.0,c_theta,s_theta],[0.0,-s_theta,c_theta]])

    #three-dimensional rotation around the y axis. Works numerically or symbollicaly
    @staticmethod
    def ry(theta, unit='rad', symbolic=False):
        
        #If symbolic, then solve using sympy
        if symbolic:
            
            if isinstance(theta, tuple(sympy.core.all_classes)):
                c_theta = sympy.cos(theta)
                s_theta = sympy.sin(theta)

            else:    
                if unit.lower() == 'deg':
                    theta=math.radians(theta)

                c_theta = math.cos(theta)
                s_theta = math.sin(theta)
        
            return sympy.Matrix( [[c_theta,0.0,-s_theta],[0.0,1.0,0.0],[s_theta,0.0,c_theta]] )
            
        #else, if it is numeric use numpy
        else:
            
            if unit.lower() == 'deg':
                theta = math.radians(theta)

            c_theta = math.cos(theta)
            s_theta = math.sin(theta)

            return numpy.array([[c_theta,0.0,-s_theta],[0.0,1.0,0.0],[s_theta,0.0,c_theta]])

    #three-dimensional rotation around the z axis. Works numerically or symbollicaly
    @staticmethod
    def rz(theta, unit='rad', symbolic=False):

        #If symbolic, then solve using sympy
        if symbolic:

            if isinstance(theta, tuple(sympy.core.all_classes)):
                c_theta = sympy.cos(theta)
                s_theta = sympy.sin(theta)
            
            else:    
                if unit.lower() == 'deg':
                    theta=math.radians(theta)

                c_theta = math.cos(theta)
                s_theta = math.sin(theta)
            
            return sympy.Matrix( [[c_theta,s_theta,0.0],[-s_theta,c_theta,0.0],[0.0,0.0,1.0]] )
            
        #else, if it is numeric use numpy
        else:
            
            if unit.lower() == 'deg':
                theta = math.radians(theta)

            c_theta = math.cos(theta)
            s_theta = math.sin(theta)

            return numpy.array([[c_theta,s_theta,0.0],[-s_theta,c_theta,0.0],[0.0,0.0,1.0]])

    #general container for six-dimensional rotations. Input 'plE' is a three dimensional rotation matrix
    @staticmethod
    def rot(plE, symbolic=False):
        
         #change list into numpy.ndarray, if necessary
        if type(plE) == list:
            plE = numpy.array( plE )

        #obtain output
        zero = numpy.zeros( (3,3) )

        #If symbolic, then solve using sympy
        if symbolic:
            return sympy.Matrix( numpy.bmat([[plE, zero], [zero, plE]]) )
        else:
            return numpy.array( numpy.bmat([[plE, zero], [zero, plE]]) )
        
    #six-dimensional rotations. Input is an angle
    def rotX(self, theta, unit='rad', symbolic=False):
        return self.rot(self.rx(theta, unit, symbolic), symbolic)

    #six-dimensional rotations. Input is an angle
    def rotY(self, theta, unit='rad', symbolic=False):
        return self.rot(self.ry(theta, unit, symbolic), symbolic)

    #six-dimensional rotations. Input is an angle
    def rotZ(self, theta, unit='rad', symbolic=False):
        return self.rot(self.rz(theta, unit, symbolic), symbolic)

    #Derivate of a rotation matrix. rot must be a sympy.Matrix object, and variable a sympy.Symbol
    def derRot(self, rot, variable):
        return rot.diff( variable )
        
    '''
    Cross product operator (rx pronounced r-cross).
    Input 'r' is either a 3D (point) vector or a skew-symmetric (3x3) matrix.
    A symbolic array can be created as r = sympy.symarray( 'r', 3 ) or r = sympy.symarray( 'r', (3,3) )
    '''
    @staticmethod
    def skew(r, symbolic=False):
    
        #If symbolic, then solve using sympy
        if symbolic:
            #change list into sympy.matrix, if necessary
            if type(r) == list:
                r = sympy.Matrix(r)
            elif type(r) == numpy.ndarray:
                r = sympy.Matrix(r)
    
            if r.shape[1] == 1:#Change from vector to matrix
                return sympy.Matrix([[0.0, -r[2], r[1]],[r[2], 0.0, -r[0]],[-r[1], r[0], 0.0]])
            else:#Change from matrix to vector
                return 0.5*sympy.Matrix([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])
        else:
            #change list into numpy.ndarray, if necessary
            if type(r) == list:
                r = numpy.array( r )
    
            if r.ndim == 1:#Change from vector to matrix
                return numpy.array([[0.0, -r[2], r[1]],[r[2], 0.0, -r[0]],[-r[1], r[0], 0.0]])
            elif r.ndim == 2:#Change from matrix to vector
                return 0.5*numpy.array([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])
            else:
                print('Wrong input')
                return [0]

    #Translation transformation. Input is a three-dimensional vector
    def xlt(self, r, symbolic=False):
        
        #If symbolic, then solve using sympy
        if symbolic:
            
            zero = sympy.zeros(3)
            identity = sympy.eye(3)
    
            out = sympy.Matrix( numpy.bmat( [[identity, zero],[-(self.skew(r, symbolic=True)), identity]] ) )
        else:
    
            zero = numpy.zeros( (3,3) )
            identity = numpy.identity(3)
    
            out = numpy.array( numpy.bmat( [[identity, zero],[-(self.skew(r)), identity]] ) )
            
        return out
        
    #General Plücker transformation for MOTION vectors.
    #Inputs are a general 3D (or 6D) rotation matrix and a 3D traslation
    def pluX(self, plE, r, symbolic=False):
        
        #If symbolic, then solve using sympy
        if symbolic:
            
            if type(plE) == list:
                plE = sympy.Matrix(plE)
    
            #If we received a 3D rotation matrix, change into 6D
            if plE.shape[0] == 3:
                plE = self.rot(plE, symbolic=True)
                
            out = plE*(self.xlt(r, symbolic=True))
        
        else:
            
            if type(plE) == list:
                plE = numpy.array(plE)

            #If we received a 3D rotation matrix, change into 6D
            if plE.shape[0] == 3:
                plE = self.rot(plE)

            out = numpy.dot( plE, self.xlt(r) )
            
        return out

    '''
    Plücker transformation for FORCE vectors.
    'pluX' is a 6x6 Plucker transformation or 'pluX' is a 3D rotation and 'r' is a translation vector
    '''    
    def pluXf(self, pluX, r=None, symbolic=False):
        
        #If symbolic, then solve using sympy
        if symbolic:
            
            #change list into sympy.Matrix, if necessary
            if type(pluX) == list:
                pluX = sympy.Matrix( pluX )
            if r is not None and type(r) == list:
                r = sympy.Matrix( r )
    
            #If a translation vector is not received, the input pluX is a 6D transformation
            if r is None:
                out11 = pluX[0:3,0:3]
                out12 = pluX[3:6,0:3]
                out21 = sympy.zeros( 3 )
                out22 = pluX[0:3,0:3]
    
                out =  sympy.Matrix( numpy.bmat([[out11, out12],[out21, out22]]) )
            else:
                #FIXME Improve this. Avoid inverse
                invTrans = ( self.xlt(r, symbolic=True) ).inv()
                out =  (self.rot(pluX, symbolic=True))*(invTrans.T)
                
        else:
            
            #change list into numpy.ndarray, if necessary
            if type(pluX) == list:
                pluX = numpy.array( pluX )
            if r is not None and type(r) == list:
                r = numpy.array( r )
    
            #If a translation vector is not received,
            #the input pluX is a 6D transformation, just manipulate as necessary
            if r is None:
                out11 = pluX[0:3,0:3]
                out12 = pluX[3:6,0:3]
                out21 = numpy.zeros( (3,3) )
                out22 = pluX[0:3,0:3]
    
                out =  numpy.bmat([[out11, out12],[out21, out22]])
            else:
                #FIXME Improve this. Avoid inverse
                invTrans = numpy.linalg.inv( self.xlt(r) )
                out =  numpy.dot( self.rot(pluX), numpy.transpose(invTrans) )
                
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
            out12 = sympy.zeros( 3 )
            out21 = (B_X_A[3:6,0:3]).T
            out22 = out11
    
            #return A_X_B
            out = sympy.Matrix( numpy.bmat([[out11, out12],[out21, out22]]) )
            
        else:
            
            #If we receive a 6x6 matrix, do nothing. Otherwise, obtain the transformation
            if plE.shape[0] == 6:
                B_X_A = plE
            else:
                B_X_A = self.pluX(plE, r)

            out11 = numpy.transpose( B_X_A[0:3,0:3] )
            out12 = numpy.zeros( (3,3) )
            out21 = numpy.transpose( B_X_A[3:6,0:3] )
            out22 = out11
    
            #return A_X_B
            out = numpy.array( numpy.bmat([[out11, out12],[out21, out22]]) )
            
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
            out21 = sympy.zeros( 3 )
            out22 = out11

            #return A_Xf_B
            out = sympy.Matrix( numpy.bmat([[out11, out12],[out21, out22]]) )
            
        else:
            #If we receive a 6x6 matrix, do nothing. Otherwise, obtain the transformation
            if pluX.shape[0] == 6:
                B_Xf_A = pluX
            else:
                B_Xf_A = self.pluXf(pluX, r)

            out11 = numpy.transpose( B_Xf_A[0:3,0:3] )
            out12 = numpy.transpose( B_Xf_A[0:3,3:6] )
            out21 = numpy.zeros( (3,3) )
            out22 = out11
    
            #return A_Xf_B
            out = numpy.array( numpy.bmat([[out11, out12],[out21, out22]]) )
            
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
            
        #If symbolic, return a sympy.Matrix, otherwise return a numpy.ndarray
        if symbolic:
            return sympy.Matrix(out)
        else:
            return numpy.array(out).reshape(6,1)

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
                    output = sympy.zeros(6)
            elif transType.lower()=='congruence':
                if inputType.lower()=='motion':
                    output = ( self.pluXf(X, symbolic=symbolic) )*A*( self.invPluX(X,symbolic=symbolic) )
                elif inputType.lower()=='force':
                    output = X*A*( self.pluXf(self.invPluX(X,symbolic=symbolic), symbolic=symbolic) )
                else:
                    print('Incorrect input type')
                    output = sympy.zeros(6)
            else:
                print('Incorrect transformation type')
                output = sympy.zeros(6)
                
            if simplification is True:
                return sympy.simplify(output)
            else:
                return output
                
        else:          
            #A.dot(B).dot(C)
            #reduce(numpy.dot, [A1, A2, ..., An])
            #multi_dot([A1d, B, C, D])#
        
            if transType.lower()=='similarity':
                if inputType.lower()=='motion':
                    output = reduce(numpy.dot, [X, A, self.invPluX(X)])
                elif inputType.lower()=='force':
                    output = reduce(numpy.dot, [self.pluXf(X), A, self.pluXf( self.invPluX(X) )])
                else:
                    print('Incorrect input type')
                    output = numpy.zeros((6,6))
            elif transType.lower()=='congruence':
                if inputType.lower()=='motion':
                    output = reduce(numpy.dot, [self.pluXf(X), A, self.invPluX(X)])
                elif inputType.lower()=='force':
                    output = reduce(numpy.dot, [X, A, self.pluXf( self.invPluX(X) )])
                else:
                    print('Incorrect input type')
                    output = numpy.zeros((6,6))
            else:
                print('Incorrect transformation type')
                output = numpy.zeros((6,6))
                
            return output
            
    #6D rotation matrix corrresponding to a spherical joint
    def eulerRot(self, angles, typeRot, outputFrame ='local', unit='rad', symbolic=False):

        #If using symbolic values, use sympy
        if symbolic:
            
            #Change to sympy object
            angles = sympy.Matrix(angles)
    
            if typeRot.lower() == 'zyx':
                rotation = \
                (self.rotX(angles[2],symbolic=True))*(self.rotY(angles[1],symbolic=True))*(self.rotZ(angles[0],symbolic=True))
            else:
                rotation = sympy.zeros(6)
                
        #Otherwise, use numpy
        else:
            
            #Change type if necessary
            if type(angles) == list:
                angles = numpy.array(angles)
    
            #Change units
            if unit.lower() == 'deg':
                angles = numpy.radians(angles)
                
            if typeRot.lower() == 'zyx':
                rotation = reduce(numpy.dot, [self.rotX(angles[2]), self.rotY(angles[1]), self.rotZ(angles[0])])
            else:
                rotation = numpy.zeros((6,6))

        #Change the output based on the reference frame of interest                
        if outputFrame.lower() == 'local':
            return rotation
        elif outputFrame.lower() == 'global':
            if symbolic:
                return rotation.T
            else:
                return numpy.transpose(rotation)
        else:
            print('Desired output frame not recognized. Returning local')
            return rotation

    #Jacobian corresponding to a rotation matrix from a spherical joint
    def eulerJacobian(self, angles, typeRot, outputFrame = 'local', unit='rad', symbolic=False):

        #If using symbolic values, use sympy
        if symbolic:
            #Change type if necessary
            if type(angles) == list:
                angles = sympy.Matrix(angles)
                
            if typeRot.lower() == 'zyx':
                spanS = sympy.zeros(3,6)
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
                spanS = sympy.zeros(6,3)

        #Otherwise use numpy
        else:
            #Change type if necessary
            if type(angles) == list:
                angles = numpy.array(angles)
    
            #Change units
            if unit.lower() == 'deg':
                angles = numpy.radians(angles)
                
            if typeRot.lower() == 'zyx':
                spanS = numpy.bmat( [self.freeMotionSpan('rz'), self.freeMotionSpan('ry'), self.freeMotionSpan('rx')] )
                spanS = numpy.array(spanS.transpose())
                
                rots = [self.rotZ(angles[0]), self.rotY(angles[1]), self.rotX(angles[2])]
                rot = self.rotZ(angles[0])
    
                #acumulate rotations
                for i in range(1,3):
                    rot = numpy.dot( rots[i], rot ) #total rotation matrix
    
                    #propagate them to each matrix spanS
                    for j in range(i):
                        spanS[j] =  numpy.dot( rots[i], spanS[j])
    
                #return a 6x3 matrix
                spanS = spanS.transpose()
            else:
                spanS = numpy.zeros( (6,3) )
                
        #if the frame is the global, multiply by the corresponding matrix
        if outputFrame.lower() == 'local':
            return spanS
        elif outputFrame.lower() == 'global':
            if symbolic:
                return (rot.T)*spanS
            else:
                return numpy.dot( numpy.transpose(rot), spanS )
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
                XJ = sympy.eye(6)
                S = sympy.zeros(6)
            else:
                XJ = numpy.identity(6)
                S = numpy.zeros((6,1))
                
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
                pluX = sympy.Matrix(pluX)
            if type(pts) == list:
                pts = numpy.array(pts)#Keep this as an array
                
            E = pluX[0:3,0:3]#Rotation component of pluX
            r = -self.skew( ( E.T )*( pluX[3:6,0:3] ) , symbolic=True)#Translation component of pluX
        
            if pts.ndim == 1:
                newPoints = E*sympy.Matrix(pts-r)
            else:
                newPoints = []
                for i,point in enumerate(pts):
                    point = sympy.Matrix(point)
                    newPoints.append( E*(point-r) )
            
            return newPoints
        
        else:
            #Change type if necessary
            if type(pluX) == list:
                pluX = numpy.array(pluX).astype(float)
            if type(pts) == list:
                pts = numpy.array(pts).astype(float)
                
            E = pluX[0:3,0:3]#Rotation component of pluX
            r = -self.skew( numpy.dot( numpy.transpose(E), pluX[3:6,0:3] ) )#Translation component of pluX
        
            if pts.ndim == 1:
                newPoints = numpy.dot( E, pts-r )
            else:
                newPoints = pts
                #newPoints = [(numpy.dot(E,point-r)) for point in pts]
                for i,point in enumerate(pts):
                    newPoints[i] = numpy.dot(E, point-r)
            
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
                    q = sympy.Matrix(q)
                
                parentArray = self.model['parents']
                Xup = [sympy.zeros( 6, 6 ) for i in range(self.model['nB'])]
                S = [sympy.zeros( 6, 1 ) for i in range(self.model['nB'])]
                X0 = [sympy.zeros( 6, 6 ) for i in range(self.model['nB'])]
                invX0 = [sympy.zeros( 6, 6 ) for i in range(self.model['nB'])]
                jacobian = [sympy.zeros( 6, self.model['nB'] ) for i in range(self.model['nB'])]
                
                for i in range(self.model['nB']):
                    
                    #Obtain JX and S
                    XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i], symbolic=True)
                    
                    S[i] = tempS
                    Xup[i] = XJ*( numpy.array( (self.model['XT'])[i] ) )
                    
                    #If the parent is the base
                    if parentArray[i] == -1:
                        X0[i] = Xup[i]
                    else:
                        X0[i] =  Xup[i]*X0[i-1]
                        
                    #Obtain inverse mappings
                    #invX0[i] = numpy.linalg.inv(X0[i])
                    #invX0[i] = (X0[i]).inv()
                    invX0[i] = self.invPluX(X0[i], None, symbolic=True)
                    
                    #We change the previous S into local coordinates of Body-i
                    for j in range(i):
                        S[j] = Xup[i]*S[j]
    
                    jacobian[i] = sympy.Matrix( numpy.transpose(S) )
            
            #Solve numerically
            else:
                
                #Change types and unit if necessary
                if type(q) == list:
                    q = numpy.array(q)
                if unit.lower() == 'deg':
                    q = numpy.radians(q)
                
                if self._modelCreated:
                    
                    parentArray = self.model['parents']
                    Xup = numpy.zeros( (self.model['nB'], 6, 6) )
                    S = numpy.zeros( (self.model['nB'], 6) )
                    X0 = numpy.zeros( (self.model['nB'], 6, 6) )
                    invX0 = numpy.zeros( (self.model['nB'], 6, 6) )
                    jacobian = numpy.zeros( (self.model['nB'], 6, self.model['nB']) )
                    
                    for i in range(self.model['nB']):
                        XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i])
                        S[i] = tempS.reshape(1,6)
                        Xup[i] = numpy.dot( XJ, (self.model['XT'])[i] )
                        
                        #If the parent is the base
                        if parentArray[i] == -1:
                            X0[i] = Xup[i]
                        else:
                            X0[i] = numpy.dot( Xup[i], X0[i-1] )
                            
                        #Obtain inverse mappings
                        invX0[i] = numpy.linalg.inv(X0[i])
                        
                        #We change the previous S into local coordinates of Body-i
                        for j in range(i):
                            S[j] = numpy.dot( Xup[i], S[j] )
                            
                        jacobian[i] = numpy.transpose(S)
                        
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
        v = numpy.array(v).flatten()
        
            
        #IF we received a 6 dimensional twist (omega, upsilon)
        if v.size == 6:
            out11 = self.skew(v[0:3])
            out12 = numpy.zeros( (3,3) )
            out21 = self.skew(v[3:6])
            out22 = out11
            
            out =  numpy.bmat([[out11, out12],[out21, out22]])
        #Otherwise, we received a 3 dimensional twist (omega_z, upsilon_x, upsilon_y)
        elif v.size == 3:
            out = numpy.array( [[0,0,0],[v[2],0,-v[0]],[-v[1],v[0],0]] )
        else:
            print('Wrong size')
            out = numpy.array([0,0,0,0,0,0])
        
        #If symbolic proceed using sympy
        if symbolic:
            return sympy.Matrix(out)
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
            return -numpy.transpose(self.crossM(v))
            
    '''
    inertiaTensor[params, type, connectivity] : Calculate inertia tensor of a body
    Assume input is an array of parameters in the form:
    params={mass, l, w, h, r} and typeObj is a string (e.g., "SolidCylinder")
    l:= length along x     w:= length along y     h:= length along z      r:=radius
    This should always be a numpy.ndarray
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
            
        inertiaT = numpy.array( [[Ixx,0.0,0.0],[0.0,Iyy,0.0],[0.0,0.0,Izz]] )
        
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
            skewC = numpy.array(skewC)#Change into numpy.ndarray in case we are using symbolic values
            tranSkewC = numpy.transpose(skewC)
            
            out11 = inertiaT + mass*numpy.dot( skewC,tranSkewC )
            out12 = mass*skewC
            out21 = mass*tranSkewC
            out22 = mass*numpy.identity(3)
            
            out =  numpy.array( numpy.bmat([[out11, out12],[out21, out22]]) )
        elif len(center) == 2:
            Izz = inertiaT
            out11 = Izz + mass*numpy.dot(center,center)
            out12 = -mass*center[1]
            out13 = mass*center[0]
            out21 = out12
            out22 = mass
            out23 = 0.0
            out31 = out13
            out32 = 0.0
            out33 = mass
            
            out = numpy.array( [[out11,out12,out13],[out21,out22,out23],[out31,out32,out33]] )
        else:
            print("Wrong dimensions")
            out = numpy.zeros((3,3))
            
        #If symbolic proceed using sympy
        if symbolic:
            return sympy.Matrix(out)
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
                    T = sympy.Matrix( [0.0,0.0,0.0,1.0,0.0,0.0] )
                    S = numpy.delete( numpy.identity(6), 3, 1 ) 
                elif normalAxis.lower() == 'y':
                    T = sympy.Matrix( [0.0,0.0,0.0,0.0,1.0,0.0] )
                    S = numpy.delete( numpy.identity(6), 4, 1 ) 
                elif normalAxis.lower() == 'z':
                    T = sympy.Matrix( [0.0,0.0,0.0,0.0,0.0,1.0] )               
                    S = numpy.delete( numpy.identity(6), 5, 1 ) 
            elif contactType.lower() == "planarhardcontact":
                T = sympy.Matrix( [[0.0,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0,0.0]] )
                T = T.T
                S = numpy.delete( numpy.identity(6), [3,4], 1 ) 
            else:
                print('Contact type not supported')
                T = [[0]]
                S = [[0]]
                
            S = sympy.Matrix(S)
        
        #Using numeric values
        else:
            
            if contactType.lower() == "pointcontactwithoutfriction":
                if normalAxis.lower() == 'x':
                    T = numpy.array( [[0,0,0,1,0,0]] ).transpose()
                    S = numpy.delete( numpy.identity(6), 3, 1 ) 
                elif normalAxis.lower() == 'y':
                    T = numpy.array( [[0,0,0,0,1,0]] ).transpose()       
                    S = numpy.delete( numpy.identity(6), 4, 1 ) 
                elif normalAxis.lower() == 'z':
                    T = numpy.array( [[0,0,0,0,0,1]] ).transpose()
                    S = numpy.delete( numpy.identity(6), 5, 1 ) 
            elif contactType.lower() == "planarhardcontact":
                T = numpy.array( [[0,0,0,1,0,0],[0,0,0,0,1,0]] ).transpose()    
                S = numpy.delete( numpy.identity(6), [3,4], 1 ) 
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
            A = numpy.zeros( (nc,dof) )
            
            #If symbolic proceed using sympy
            if symbolic:
                A = sympy.Matrix( A )
        
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
                        A[i] = reduce(numpy.dot, [numpy.transpose(T), self.xlt(contactPoint), Jacobians[constrainedBody]])
                
                elif constraintType.lower() == 'non-slippagewithfriction':
                    T,S = self.contactConstraints(contactModel, symbolic=symbolic)
                    
                    if symbolic:
                        A[i,:] = ( T.T )*( self.xlt(contactPoint, symbolic=True) )*( Jacobians[constrainedBody] )
                    else:
                        A[i] = reduce(numpy.dot, [numpy.transpose(T), self.xlt(contactPoint), Jacobians[constrainedBody]])
                
                elif constraintType.lower() == 'bodycontact':
                    T,S = self.contactConstraints(contactModel)
                    
                    if contactingSide.lower() == 'left':
                        if symbolic:
                            A[i,:] = \
                            ( T.T )*( self.pluX(self.rz(contactAngle, symbolic=True), contactPoint, symbolic=True) )*( Jacobians[constrainedBody] )
                        else:
                            A[i] = reduce(numpy.dot, [numpy.transpose(T), self.pluX(self.rz(contactAngle), contactPoint), Jacobians[constrainedBody]])               
                    elif contactingSide.lower() == 'right':
                        contactAngle = contactAngle + math.pi
                            
                        if symbolic:
                            A[i,:] = \
                            ( T.T )*( self.pluX(self.rz(contactAngle, symbolic=True), contactPoint, symbolic=True) )*( Jacobians[constrainedBody] )
                        else:
                            A[i] = reduce(numpy.dot, [numpy.transpose(T), self.pluX(self.rz(contactAngle), contactPoint), Jacobians[constrainedBody]])
                            
                    else:
                        print('Wrong side')
                        
                else:
                        print('Wrong contraint type')
                
            #If symbolic proceed using sympy
            if symbolic:
                
                # How to get derivative of the constraint matrix? derConstraintMatrixA = d A/dt
                derA = A.diff(sympy.symbols('t'))
                
                # kappa = numpy.dot( derConstraintMatrixA, velocities )
                kappa = derA*(sympy.Matrix(self.dq))
                
            #Use numeric values
            else:
                
                # How to get derivative of the constraint matrix? derConstraintMatrixA = d A/dt
                derA = numpy.array([[0]])
                
                # kappa = numpy.dot( derConstraintMatrixA, velocities )
                kappa = numpy.array([[0]])
                
            # kappa_stab = beta*numpy.dot( constraintMatrix, velocities )
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
                
                #change list into sympy.Matrix
                q = sympy.Matrix( q )
                qd = sympy.Matrix( qd )
                qdd = sympy.Matrix( qdd )
                fext = sympy.Matrix( fext ) #Each row represents a wrench applied to a body
             
                Xup = list(sympy.zeros(6,6) for i in range(nBodies))
                S = list(sympy.zeros(6,1) for i in range(nBodies))
                X0 = list(sympy.zeros(6,6) for i in range(nBodies))
                parentArray = self.model['parents']
                
                v = list(sympy.zeros(6,1) for i in range(nBodies))
                a = list(sympy.zeros(6,1) for i in range(nBodies))
                f = list(sympy.zeros(6,1) for i in range(nBodies))
                
                tau = sympy.zeros( dof,1 )
                aGrav = self.model['inertialGrav']
                
                if fext.shape[0] == 0:
                    fext = list(sympy.zeros(6,1) for i in range(nBodies))
                    
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
                    
                    #f[i] = numpy.dot( RBInertia, a[i] ) + reduce(numpy.dot, [self.crossF(v[i]), RBInertia, v[i]]) - numpy.dot( numpy.transpose( numpy.linalg.inv( X0[i] ) ), fext[i] )
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
                
                #change list into numpy.ndarray, if necessary
                if type(q) == list:
                    q = numpy.array( q )
                if type(qd) == list:
                    qd = numpy.array( qd )
                if type(q) == list:
                    qdd = numpy.array( qdd )
                if type(fext) == list:
                    fext = numpy.array( fext )
            
                Xup = numpy.zeros( (nBodies, 6, 6) )
                S = numpy.zeros( (nBodies, 6) )
                X0 = numpy.zeros( (nBodies, 6, 6) )
                parentArray = self.model['parents']
            
                v = numpy.zeros( (nBodies, 6) )
                a = numpy.zeros( (nBodies, 6) )
                f = numpy.zeros( (nBodies, 6) )
            
                tau = numpy.zeros( dof )
                aGrav = self.model['inertialGrav']
            
                if fext.size == 0:
                    fext = numpy.zeros( (nBodies, 6) )
                
                for i in range(nBodies):
                    
                    XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i])
                    S[i] = tempS.flatten()#Make a flat array. This is easier when using numpy
                    vJ = S[i]*qd[i]
                    Xup[i] = numpy.dot( XJ, (self.model['XT'])[i] )
                    
                    #If the parent is the base
                    if parentArray[i] == -1:
                        X0[i] = Xup[i]
                        v[i] = vJ
                        a[i] = numpy.dot( Xup[i], -aGrav ) + S[i]*qdd[i]
                    else:
                        X0[i] = numpy.dot( Xup[i], X0[i-1] )
                        v[i] = numpy.dot( Xup[i], v[parentArray[i]] ) + vJ
                        a[i] = numpy.dot( Xup[i], a[parentArray[i]] ) + S[i]*qdd[i] + numpy.dot( self.crossM(v[i]), vJ )
                    
                    RBInertia = (self.model['rbInertia'])[i]
                    
                    f1 = numpy.dot( RBInertia, a[i] )
                    f2 = reduce(numpy.dot, [self.crossF(v[i]), RBInertia, v[i]])
                    f3 = numpy.dot( numpy.transpose( numpy.linalg.inv( X0[i] ) ), fext[i] )
                    
                    f[i] = f1 + f2 - f3
                    #f[i] = numpy.dot( RBInertia, a[i] ) + reduce(numpy.dot, [self.crossF(v[i]), RBInertia, v[i]]) - numpy.dot( numpy.transpose( numpy.linalg.inv( X0[i] ) ), fext[i] )
    
                for i in range(nBodies-1,-1,-1):
                    tau[i] = numpy.dot( S[i], f[i] )
                    
                    #If the parent is not the base
                    if parentArray[i] != -1:
                        f[parentArray[i]] = f[parentArray[i]] + numpy.dot( numpy.transpose(Xup[i]), f[i] )
                        
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
    def HandC(self, q, qd, fext = [], gravityTerms = True, symbolic=False):
        
        # Only continue if createModel has been called                
        if self._modelCreated:
            
            dof = self.model['DoF']
            nBodies = self.model['nB']
            
            #If symbolic proceed using sympy
            if symbolic:
                
                #change list into sympy.Matrix
                q = sympy.Matrix( q )
                qd = sympy.Matrix( qd )
                fext = sympy.Matrix( fext ) #Each row represents a wrench applied to a body
             
                Xup = list(sympy.zeros(6,6) for i in range(nBodies))
                S = list(sympy.zeros(6,1) for i in range(dof))
                X0 = list(sympy.zeros(6,6) for i in range(nBodies))
                parentArray = self.model['parents']
                
                v = list(sympy.zeros(6,1) for i in range(nBodies))
                avp = list(sympy.zeros(6,1) for i in range(nBodies))
                fvp = list(sympy.zeros(6,1) for i in range(nBodies))
                Cor = sympy.zeros( nBodies,1 )
                
                if gravityTerms:
                    aGrav = self.model['inertialGrav']
                else:
                    aGrav = sympy.zeros(6,1)
                
                if fext.shape[0] == 0:
                    fext = list(sympy.zeros(6,1) for i in range(nBodies))
                    
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
                
                IC = list(sympy.zeros(6,6) for i in range(nBodies))
                for i in range(nBodies):
                    IC[i] = (self.model['rbInertia'])[i]
                    
                for i in range(nBodies-1,-1,-1):
                    if parentArray[i] != -1:
                        IC[parentArray[i]] = IC[parentArray[i]] + ( ( Xup[i] ).T )*IC[i]*Xup[i]
        
                H = sympy.zeros( nBodies )
                for i in range(nBodies):
                    fh = IC[i]*S[i]
                    H[i,i] = (S[i].T)*fh
                    j = i
                    
                    while parentArray[j] > -1:
                        fh = ( (Xup[j]).T )*fh
                        j = parentArray[j]
                        H[i, j] = (S[j].T)*fh
                        H[j, i] = H[i, j]
                        
            #Otherwise, use numpy
            else:
                #change list into numpy.ndarray, if necessary
                if type(q) == list:
                    q = numpy.array( q )
                if type(qd) == list:
                    qd = numpy.array( qd )
                if type(fext) == list:
                    fext = numpy.array( fext )
             
                Xup = numpy.zeros( (nBodies, 6, 6) )
                S = numpy.zeros( (dof, 6) )
                X0 = numpy.zeros( (nBodies, 6, 6) )
                parentArray = self.model['parents']
                
                v = numpy.zeros( (nBodies, 6) )
                avp = numpy.zeros( (nBodies, 6) )
                fvp = numpy.zeros( (nBodies, 6) )
                Cor = numpy.zeros( nBodies )
                
                if gravityTerms:
                    aGrav = self.model['inertialGrav']
                else:
                    aGrav = numpy.zeros( 6 )
                    
                if fext.size == 0:
                    fext = numpy.zeros( (nBodies, 6) )
                    
                # Calculation of Coriolis, centrifugal, and gravitational terms
                for i in range(nBodies):
                    
                    XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i])
                    S[i] = tempS.flatten()#Make a flat array. This is easier when using numpy
                    vJ = S[i]*qd[i]
                    XT = (self.model['XT'])[i]
                    
                    Xup[i] = numpy.dot( XJ, XT )
    
                    #If the parent is the base
                    if parentArray[i] == -1:
                        X0[i] = Xup[i]
                        v[i] = vJ
                        avp[i] = numpy.dot( Xup[i], -aGrav )
                    else:
                        X0[i] = numpy.dot( Xup[i], X0[i-1] )
                        v[i] = numpy.dot( Xup[i], v[parentArray[i]] ) + vJ
                        avp[i] = numpy.dot( Xup[i], avp[parentArray[i]] ) + numpy.dot( self.crossM(v[i]), vJ )
                    
                    RBInertia = (self.model['rbInertia'])[i]
                    
                    f1 = numpy.dot( RBInertia, avp[i] )
                    f2 = reduce(numpy.dot, [self.crossF(v[i]), RBInertia, v[i]])
                    f3 = numpy.dot( numpy.transpose( numpy.linalg.inv( X0[i] ) ), fext[i] )
                    
                    fvp[i] = f1 + f2 - f3
                    
                for i in range(nBodies-1,-1,-1):
                    Cor[i] = numpy.dot( S[i], fvp[i] )
                    
                    #If the parent is not the base
                    if parentArray[i] != -1:
                        fvp[parentArray[i]] = fvp[parentArray[i]] + numpy.dot( numpy.transpose(Xup[i]), fvp[i] )
                
                IC = numpy.zeros( (nBodies, 6, 6) )
                for i in range(nBodies):
                    IC[i] = (self.model['rbInertia'])[i]
                    
                for i in range(nBodies-1,-1,-1):
                    if parentArray[i] != -1:
                        IC[parentArray[i]] = IC[parentArray[i]] + reduce(numpy.dot, [numpy.transpose( Xup[i] ), IC[i], Xup[i]])
                        
                H = numpy.zeros( (nBodies, nBodies) )
                for i in range(nBodies):
                    fh = numpy.dot( IC[i], S[i] )
                    H[i,i] = numpy.dot( S[i], fh )
                    j = i
                    
                    while parentArray[j] > -1:
                        fh = numpy.dot( numpy.transpose(Xup[j]), fh )
                        j = parentArray[j]
                        H[i, j] = numpy.dot( S[j], fh )
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
            tau = sympy.Matrix(tau)
            RHS = tau - localCor
            ddq = localH.LUsolve(RHS)
        else:
            tau = numpy.array(tau)
            tau.resize( (tau.size,1) )
            RHS = tau - localCor
            ddq = numpy.linalg.solve( localH, RHS )
        
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
                #change list into sympy.Matrix
                q = sympy.Matrix( q )
                qd = sympy.Matrix( qd )
                fext = sympy.Matrix( fext ) #Each row represents a wrench applied to a body
             
                Xup = list(sympy.zeros(6,6) for i in range(nBodies))
                S = list(sympy.zeros(6,1) for i in range(dof))
                X0 = list(sympy.zeros(6,6) for i in range(nBodies))
                parentArray = self.model['parents']
                
                v = list(sympy.zeros(6,1) for i in range(nBodies))
                c = list(sympy.zeros(6,1) for i in range(nBodies))
                
                if gravityTerms:
                    aGrav = self.model['inertialGrav']
                else:
                    aGrav = sympy.zeros(6,1)
                    
                IA = list(sympy.zeros(6,6) for i in range(nBodies))
                pA = list(sympy.zeros(6,1) for i in range(nBodies))
                                
                if fext.shape[0] == 0:
                    fext = list(sympy.zeros(6,1) for i in range(nBodies))
                    
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
                    IA[i] = sympy.Matrix( RBInertia )
                    
                    pA1 = ( self.crossF(v[i], symbolic=True) )*( IA[i] )*( v[i] )
                    pA2 = ( (self.invPluX(X0[i], symbolic=True)).T )*( fext[i] )
                    
                    pA[i] = pA1 - pA2
                    
                # 2nd pass. Calculate articulated-body inertias
                U = list(sympy.zeros(6,1) for i in range(nBodies))
                d = sympy.zeros( nBodies,1 )
                u = sympy.zeros( nBodies,1 )
                
                for i in range(nBodies-1,-1,-1):
                    
                    U[i] = IA[i]*S[i]
                    d[i] = ( S[i].T )*U[i]
                    #u[i] = tau[i] - ( S[i].T )*pA[i]
                    u[i] = tau[i] - S[i].dot(pA[i])
                    
                    #invd = 1.0/(d[i] + numpy.finfo(float).eps)
                    invd = 1.0/d[i]
                    
                    #If the parent is the base
                    if parentArray[i] != -1:
                        Ia = IA[i] - invd*U[i]*(U[i].T)
                        pa = pA[i] + Ia*c[i] + invd*u[i]*U[i]
                        
                        IA[parentArray[i]] = IA[parentArray[i]] + (Xup[i].T)*Ia*Xup[i]
                        pA[parentArray[i]] = pA[parentArray[i]] + (Xup[i].T)*pa
                        
                # 3rd pass. Calculate spatial accelerations
                a = list(sympy.zeros(6,1) for i in range(nBodies))
                qdd = sympy.zeros(dof,1)
                
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
                #change list into numpy.ndarray, if necessary
                if type(q) == list:
                    q = numpy.array( q )
                if type(qd) == list:
                    qd = numpy.array( qd )
                if type(fext) == list:
                    fext = numpy.array( fext ) #Each row represents a wrench applied to a body
             
                Xup = numpy.zeros( (nBodies, 6, 6) )
                S = numpy.zeros( (dof, 6) )
                X0 = numpy.zeros( (nBodies, 6, 6) )
                parentArray = self.model['parents']            
                v = numpy.zeros( (nBodies, 6) )
                c = numpy.zeros( (nBodies, 6) )
                
                if gravityTerms:
                    aGrav = self.model['inertialGrav']
                else:
                    aGrav = numpy.zeros( 6 )
                    
                IA = numpy.zeros( (nBodies, 6, 6) )
                pA = numpy.zeros( (nBodies, 6) )
                    
                if fext.size == 0:
                    fext = numpy.zeros( (nBodies, 6) )
                    
                # Calculation of Coriolis, centrifugal, and gravitational terms
                for i in range(nBodies):
                    
                    XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i])
                    S[i] = tempS.flatten()#Make a flat array. This is easier when using numpy
                    #S[i] = tempS
                    vJ = S[i]*qd[i]
                    XT = (self.model['XT'])[i]
                    
                    Xup[i] = numpy.dot( XJ, XT )
    
                    #If the parent is the base
                    if parentArray[i] == -1:
                        X0[i] = Xup[i]
                        v[i] = vJ
                        #c[i] = numpy.zeros( 6 )
                    else:
                        X0[i] = numpy.dot( Xup[i], X0[i-1] )
                        v[i] = numpy.dot( Xup[i], v[parentArray[i]] ) + vJ
                        c[i] = numpy.dot( self.crossM(v[i]), vJ )
                    
                    RBInertia = (self.model['rbInertia'])[i]
                    IA[i] = RBInertia
                    
                    pA1 = reduce(numpy.dot, [self.crossF(v[i]), IA[i], v[i]])
                    pA2 = numpy.dot( numpy.transpose( numpy.linalg.inv( X0[i] ) ), fext[i] )
                    
                    pA[i] = pA1 - pA2
                
                # 2nd pass. Calculate articulated-body inertias
                U = numpy.zeros( (nBodies, 6) )
                d = numpy.zeros( nBodies )
                u = numpy.zeros( nBodies )
                
                for i in range(nBodies-1,-1,-1):
                    
                    U[i] = numpy.dot( IA[i], S[i] )
                    d[i] = numpy.dot( S[i], U[i] )
                    u[i] = tau[i] - numpy.dot( S[i], pA[i])
                    
                    #invd = 1.0/(d[i] + numpy.finfo(float).eps)
                    invd = 1.0/d[i]
                    
                    #If the parent is the base
                    if parentArray[i] != -1:
                        Ia = IA[i] - invd*numpy.outer( U[i], U[i] )
                        pa = pA[i] + numpy.dot( Ia, c[i] ) + invd*u[i]*U[i]
                        
                        IA[parentArray[i]] = IA[parentArray[i]] + reduce( numpy.dot, [numpy.transpose(Xup[i]), Ia, Xup[i]] )
                        pA[parentArray[i]] = pA[parentArray[i]] + numpy.dot( numpy.transpose(Xup[i]), pa )      
                
                # 3rd pass. Calculate spatial accelerations
                a = numpy.zeros( (nBodies, 6) )
                qdd = numpy.zeros( dof )
                
                for i in range(nBodies):
                    
                    invd = 1.0/d[i]
                    
                    #If the parent is the base
                    if parentArray[i] == -1:
                        a[i] = numpy.dot( Xup[i], -aGrav ) + c[i]
                    else:
                        a[i] = numpy.dot( Xup[i], a[parentArray[i]] ) + c[i]
                    
                    qdd[i] = invd*(u[i] - numpy.dot( U[i], a[i] ))
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
            angles = numpy.array(angles)
        absangles = numpy.zeros( angles.size )        
        
        for i in range(angles.size):
            absangles[i] = numpy.sum( angles[0:i+1] )
        
        return absangles
        
    #Change dimensions, from planar to spatial, or spatial to planar
    def dimChange(self, quantity):
        
        #Change into numpy.ndarray if necessary
        if type(quantity) == list:
            quantity = numpy.array(quantity)
            
        #If the dimensions is 3, then we received a planar vector (twist or wrench). Convert to spatial vector
        if quantity.shape == (3,1):
            quantity = numpy.insert( quantity, 0, [[0.0],[0.0]], axis=0 )
            quantity = numpy.append(quantity, [[0.0]], axis=0)
        elif quantity.shape == (3,):
            quantity = numpy.insert(quantity, 0, [0.0,0.0])
            quantity = numpy.append(quantity, 0)
        elif quantity.shape == (1,3):
            quantity = numpy.insert(quantity, 0, [[0.0],[0.0]], axis=1)
            quantity = numpy.append(quantity,[[0.0]], axis=1)
        #If the dimensions is 3by3, then we received a 3D Matrix
        elif quantity.shape == (3,3):
#            temp = numpy.zeros( (6,6) )
#            temp[2:5,2:5] = quantity
#            quantity = temp
            quantity = numpy.insert(quantity, 0, [[0.0],[0.0]], axis=0)
            quantity = numpy.append(quantity, numpy.zeros((1,3)), axis=0)
            
            quantity = numpy.insert(quantity, 0, [[0.0],[0.0]],axis=1)
            quantity = numpy.append(quantity, numpy.transpose([[0.0,0.0,0.0,0.0,0.0,0.0]]), axis=1)
        #If the dimensions is 6, then we received a spatial vector (twist or wrench). Convert to planar vector
        elif quantity.shape == (6,1) or quantity.shape == (6,):
            quantity = quantity[2:5]
        elif quantity.shape == (1,6):
            temp = quantity[0,2:5]
            #quantity = numpy.reshape( quantity[2:5],(1,3) )
            quantity = numpy.reshape( temp,(1,3) )
        #If the dimensions is 6by6, then we received a 6D Matrix
        elif quantity.shape == (6,6):
            quantity = quantity[2:5,2:5]

        return quantity
    
    # Transform a symbolic Matrix into a array (with float type values)
    # input array is a Matrix object and 
    #substitutions is either a tuple (oldvalue, new value) or a list fo tuples [(old, new), ..., (old, new)]
    def symToNum(self, inputArray, substitutions):
        
        if type(substitutions) is not list:
            substitutions = [substitutions]
            
        output = inputArray.subs( substitutions )
        output = numpy.array( output ).astype(numpy.float)
        
        return output
        
    #returns a list of tuples from two lists
    def tuplesFromLists(self, x, y):
        
        #If we received a list, make a list of tuples. Otherwise just return one tuple
        if type(x) is list:
            return zip(x, y)
        else:
            return (x,y)
    
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
        
# ------------FILTERING----------------
        
    #convert from an array of (unsigned) integers to numerical values. Input should be one dimensional
    #If the input is bytearray then 'bytearrayToInt' is called first
    def mapToValues(self, data, in_min, in_max, out_min, out_max, noBytes=1):

        #convert bytearray to a numpy.array containing integers
        if type(data) == bytearray:
            data = self.bytearrayToInt(data, noBytes)

        values = [(x-in_min)*(out_max-out_min)/(in_max-in_min)+out_min for x in data]
        return numpy.array(values)
        
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
            invX0 = numpy.array(invX0)
        depth = invX0.shape[0]
        
        zero = [ numpy.zeros( (1,3) ) for i in range(depth) ]
        origins = [ numpy.zeros( (1,3) ) for i in range(depth+1) ]
        
        # Calculate the origins of each link and return as matrix (each row represents a origin)
        origins[1:] = self.worldPts(invX0, zero)
        origins = numpy.reshape( origins, (depth+1,3) )
            
        # TODO: Calculate end of last link
            
        return origins
        
    def drawSkeleton(self, invX0, autoS=True, xlim=None, ylim=None):
        
        # Total length
        totalLength = sum( self.model['lengths'] )
        scale = self.mapToValues( [1,5,10], 0.0, 100.0, 0.0, totalLength ) # Scale representin 1%,5%, and 10% of the snake robot's length
        
        # Obtain all the origins
        origins3D = self.origins(invX0)
        
        # Remove a dimension
        origins2D = numpy.delete( origins3D, 2, 1)
        
        # Remove origin and virtual links
        if self.model['floatingBase']:
            origins2D = numpy.delete( origins2D, [0,1,2], 0)
        
        fig = plt.figure()
        axes = fig.add_subplot(1,1,1)
        #patch = []
        
        skeleton = Polygon(origins2D, closed=False, fill=False, edgecolor='r', lw=1)
        #skeleton = plt.Polygon(origins2D, closed=None, fill=None, edgecolor='r')
        
        axes.add_patch(skeleton)
        #patch.append(skeleton)
        
        # Obtain COM            
        COMs = self.worldPts(invX0, self.model['com'])
        COMs = numpy.delete( COMs, 2, 1)
        if self.model['floatingBase']:
            COMs = numpy.delete( COMs, [0,1], 0)
        
        for i in COMs:
            circle = Circle(i, scale[0], facecolor='y')
            #patch.append(circle)
            axes.add_patch(circle)
            
        # Joints
        joints = numpy.delete( origins3D, 2, 1)
        if self.model['floatingBase']:
            joints = numpy.delete( joints, [0,1,2,3], 0)
        else:
            joints = numpy.delete( joints, [0], 0)
            
        for i in joints:
            circle = Circle(i, 2*scale[0], fill=False)
            #patch.append(circle)
            axes.add_patch(circle)
        
        # Add the patches to the axes
#        p = PatchCollection(patch, alpha=0.4)
#        colors = 100*numpy.random.rand(len(patch))
#        p.set_array(numpy.array(colors))
#        
#        axes.add_collection(p)
        
#        if autoS:
#            axes.autoscale_view(True,True,True)
#        else:
#            if xlim is not None:
#                axes.set_xlim(xlim[0], xlim[1])
#            if ylim is not None:
#                axes.set_ylim(ylim[0], ylim[1])
                
        axes.axis("equal")
        
        plt.show()
        
        
# --------------------PARSING-----------------------
    
    # Parse a string into a sympy expression
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
        expr = sympy.Matrix(expr)
        
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
        
