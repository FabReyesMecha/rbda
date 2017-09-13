# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 03:09:34 2017

Small module containing Algorithms for calculating twistes and wrenches.
This module is just a small portion of algorithms available on:
    Rigid Body Dynamic Algorithms, by Featherstone, Roy (2008)
    
The latest version should be in ../moduletest/

Usage:

Download rbda.py into your computer. For example, if the file is in the directory
'c:/users/reyes fabian/my cubby/python/python_v-rep/moduletest', then use the following code:

import os
os.chdir('c:/users/reyes fabian/my cubby/python/python_v-rep/moduletest')

import rbda
plu = rbda.rbdaClass()

#Start using the functions. Example:
plu.rx(45, unit='deg')

----------------------------------------------------------------------------
Log:

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
import numbers

import sympy
from sympy.physics.vector import dynamicsymbols
from sympy import diff, Symbol

class rbdaClass():

    #Constructor
    def __init__(self):
        self.__version__ = '0.5.0'
        self._modelCreated = False
        
    #Created model
    '''
    example:
    
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
    
    # robot with floating base ---------------------
    
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
    
    constraintsInformation={
    'nc':1,
    'body':[2],
    'contactPoint':[[0.075,0,0]],
    'contactAngle':[0],
    'constraintType':['bodyContact'],
    'contactModel':['pointContactWithoutFriction'],
    'contactingSide':['left']
    }
    
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
        
        # Option 3: return a numpy array
#        self.qVector = numpy.array( dynamicsymbols(qVector) )
#        self.dq = numpy.array( dynamicsymbols(dq) )
        
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
            print('Gravity direction not recogniez. Assuming it is in the y-axis.')
            self.model['inertialGrav'] = numpy.array( [0.0,0.0,0.0,0.0,-(self.model['g']),0.0] )
        
        #assign parameters. If there is a floating base, then there should be two virtual (massless) links
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
            center = [(self.model['Lx'])[i],(self.model['Ly'])[i],(self.model['Lz'])[i]]
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
    def rx(theta, unit='rad'):
        
        if isinstance(theta, tuple(sympy.core.all_classes)):
            c_theta = sympy.cos(theta)
            s_theta = sympy.sin(theta)

        else:    
            if unit.lower() == 'deg':
                theta=math.radians(theta)

            c_theta = math.cos(theta)
            s_theta = math.sin(theta)
        
        return sympy.Matrix( [[1.0,0.0,0.0],[0.0,c_theta,s_theta],[0.0,-s_theta,c_theta]] )

    #three-dimensional rotation around the y axis. Works numerically or symbollicaly
    @staticmethod
    def ry(theta, unit='rad'):

        if isinstance(theta, tuple(sympy.core.all_classes)):
            c_theta = sympy.cos(theta)
            s_theta = sympy.sin(theta)

        else:    
            if unit.lower() == 'deg':
                theta=math.radians(theta)

            c_theta = math.cos(theta)
            s_theta = math.sin(theta)
        
        return sympy.Matrix( [[c_theta,0.0,-s_theta],[0.0,1.0,0.0],[s_theta,0.0,c_theta]] )

    #three-dimensional rotation around the z axis. Works numerically or symbollicaly
    @staticmethod
    def rz(theta, unit='rad'):

        if isinstance(theta, tuple(sympy.core.all_classes)):
            c_theta = sympy.cos(theta)
            s_theta = sympy.sin(theta)
            
        else:    
            if unit.lower() == 'deg':
                theta=math.radians(theta)

            c_theta = math.cos(theta)
            s_theta = math.sin(theta)
            
        return sympy.Matrix( [[c_theta,s_theta,0.0],[-s_theta,c_theta,0.0],[0.0,0.0,1.0]] )

    #general container for six-dimensional rotations. Input 'plE' is a three dimensional rotation matrix
    @staticmethod
    def rot(plE):
        
         #change list into numpy.ndarray, if necessary
        if type(plE) == list:
            plE = numpy.array( plE )

        #obtain output
        zero = sympy.zeros(3)
        #return sympy.BlockMatrix( [[plE, zero], [zero, plE]] )
        return sympy.Matrix( numpy.bmat([[plE, zero], [zero, plE]]) )
        

    #six-dimensional rotations. Input is an angle
    def rotX(self, theta, unit='rad'):
        return self.rot(self.rx(theta, unit))

    #six-dimensional rotations. Input is an angle
    def rotY(self, theta, unit='rad'):
        return self.rot(self.ry(theta, unit))

    #six-dimensional rotations. Input is an angle
    def rotZ(self, theta, unit='rad'):
        return self.rot(self.rz(theta, unit))

    #Derivate of a rotation matrix. rot must be a sympy.Matrix object, and variable a sympy.Symbol
    def derRot(self, rot, variable):
        return rot.diff( variable )
        
    '''
    Cross product operator (rx pronounced r-cross).
    Input 'r' is either a 3D (point) vector or a skew-symmetric (3x3) matrix.
    A symbolic array can be created as r = sympy.symarray( 'r', 3 ) or r = sympy.symarray( 'r', (3,3) )
    '''
    @staticmethod
    def skew(r):
    
        #change list into sympy.matrix, if necessary
        if type(r) == list:
            r = sympy.Matrix(r)
        elif type(r) == numpy.ndarray:
            r = sympy.Matrix(r)

        if r.shape[1] == 1:#Change from vector to matrix
            return sympy.Matrix([[0.0, -r[2], r[1]],[r[2], 0.0, -r[0]],[-r[1], r[0], 0.0]])
        else:#Change from matrix to vector
            return 0.5*sympy.Matrix([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])

    #Translation transformation. Input is a three-dimensional vector
    def xlt(self, r):

        zero = sympy.zeros(3)
        identity = sympy.eye(3)

        return sympy.Matrix( numpy.bmat( [[identity, zero],[-(self.skew(r)), identity]] ) )
        

    #General Plücker transformation for MOTION vectors.
    #Inputs are a general 3D (or 6D) rotation matrix and a 3D traslation
    def pluX(self, plE, r):

        if type(plE) == list:
            plE = sympy.Matrix(plE)

        #If we received a 3D rotation matrix, change into 6D
        if plE.shape[0] == 3:
            plE = self.rot(plE)
            
        return plE*(self.xlt(r))

    '''
    Plücker transformation for FORCE vectors.
    'pluX' is a 6x6 Plucker transformation or 'pluX' is a 3D rotation and 'r' is a translation vector
    '''    
    def pluXf(self, pluX, r=None):

        #change list into numpy.ndarray, if necessary
        if type(pluX) == list:
            pluX = sympy.Matrix( pluX )
        if r is not None and type(r) == list:
            r = sympy.Matrix( r )

        #If a translation vector is not received,
        #the input pluX is a 6D transformation, just manipulate as necessary
        if r is None:
            out11 = pluX[0:3,0:3]
            out12 = pluX[3:6,0:3]
            out21 = sympy.zeros( 3 )
            out22 = pluX[0:3,0:3]

            return sympy.Matrix( numpy.bmat([[out11, out12],[out21, out22]]) )
        else:
            invTrans = ( self.xlt(r) ).inv()
            return (self.rot(pluX))*(invTrans.T)

    #Inverse for pluX. Inputs are a general 3D (or 6D) rotation matrix and a 3D traslation
    #If receiving a 6 by 6 matrix, only shift its blocks around
    #TODO: Review
    def invPluX(self, plE, r=None):
        
        if plE.shape[0] == 6:
            B_X_A = plE
        else:
            B_X_A = self.pluX(plE, r)

        out11 = (B_X_A[0:3,0:3]).T
        out12 = sympy.zeros( 3 )
        out21 = (B_X_A[3:6,0:3]).T
        out22 = out11

        return sympy.Matrix( numpy.bmat([[out11, out12],[out21, out22]]) )

    #Inverse for pluXf
    #pluX is a 6x6 Plucker transformation or pluX is a 3-D rotation and r is a translation vector
    #TODO: review
    def invPluXf(self, pluX, r=None):
        
        B_Xf_A = self.pluXf(pluX, r)
            
        out11 = ( B_Xf_A[0:3,0:3] ).T
        out12 = ( B_Xf_A[0:3,3:6] ).T
        out21 = sympy.zeros( 3 )
        out22 = out11

        return sympy.Matrix( numpy.bmat([[out11, out12],[out21, out22]]) )

    #Definitions of free space for one-dimensional joints
    @staticmethod
    def freeMotionSpan(typeMotion):
        if typeMotion.lower() == 'rx':
            return sympy.Matrix([1.0,0.0,0.0,0.0,0.0,0.0])
        elif typeMotion.lower() == 'ry':
            return sympy.Matrix([0.0,1.0,0.0,0.0,0.0,0.0])
        elif typeMotion.lower() == 'rz':
            return sympy.Matrix([0.0,0.0,1.0,0.0,0.0,0.0])
        elif typeMotion.lower() == 'px':
            return sympy.Matrix([0.0,0.0,0.0,1.0,0.0,0.0])
        elif typeMotion.lower() == 'py':
            return sympy.Matrix([0.0,0.0,0.0,0.0,1.0,0.0])
        elif typeMotion.lower() == 'pz':
            return sympy.Matrix([0.0,0.0,0.0,0.0,0.0,1.0])
        else:
            return sympy.zeros(6,1)

    #Coordinate transformation. Similarity or congruence. All 6D Matrices. X is always motion transformation
    def coordinateTransform(self, X, A, transType, inputType, simplification=True):
        #A.dot(B).dot(C)
        #reduce(numpy.dot, [A1, A2, ..., An])
        #multi_dot([A1d, B, C, D])#

        if transType.lower()=='similarity':
            if inputType.lower()=='motion':
                #return reduce(numpy.dot, [X, A, numpy.linalg.inv(X)])
                output = X*A*(X.inv())
            elif inputType.lower()=='force':
                #return reduce(numpy.dot, [self.pluXf(X), A, self.pluXf( numpy.linalg.inv(X))])
                output = ( self.pluXf(X) )*A*( self.pluXf(X.inv()) )
            else:
                print('Incorrect input type')
                output = sympy.zeros(6)
        elif transType.lower()=='congruence':
            if inputType.lower()=='motion':
                #return reduce(numpy.dot, [self.pluXf(X), A, numpy.linalg.inv(X)])
                output = ( self.pluXf(X) )*A*( X.inv() )
            elif inputType.lower()=='force':
                #return reduce(numpy.dot, [X, A, self.pluXf( numpy.linalg.inv(X))])
                output = X*A*( self.pluXf(X.inv()) )
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

    #6D rotation matrix corrresponding to a spherical joint
    def eulerRot(self, angles, typeRot, outputFrame ='local', unit='rad'):

        #Change type if necessary
        if type(angles) == list:
            angles = numpy.array(angles)

        #Change units
        if unit.lower() == 'deg':
            angles = numpy.radians(angles)
            
        #Change to sympy object
        angles = sympy.Matrix(angles)

        if typeRot.lower() == 'zyx':
            #rotation = reduce(numpy.dot, [self.rotX(angles[2]), self.rotY(angles[1]), self.rotZ(angles[0])])
            rotation = (self.rotX(angles[2]))*(self.rotY(angles[1]))*(self.rotZ(angles[0]))
        else:
            #rotation = numpy.zeros((6,6))
            rotation = sympy.zeros(6)

        if outputFrame.lower() == 'local':
            return rotation
        elif outputFrame.lower() == 'global':
            #return numpy.transpose(rotation)
            return rotation.T
        else:
            print('Desired output frame not recognized. Returning local')
            return rotation

    #Jacobian corresponding to a rotation matrix from a spherical joint
    #TODO Review
    def eulerJacobian(self, angles, typeRot, outputFrame = 'local', unit='rad'):

        #Change type if necessary
        if type(angles) == list:
            angles = numpy.array(angles)

        #Change units
        if unit.lower() == 'deg':
            angles = numpy.radians(angles)
            
        #Change to sympy
        angles = sympy.Matrix(angles)

        if typeRot.lower() == 'zyx':
            spanS = sympy.zeros(3,6)
            spanS[0,:] = self.freeMotionSpan('rz').reshape(1,6)
            spanS[1,:] = self.freeMotionSpan('ry').reshape(1,6)
            spanS[2,:] = self.freeMotionSpan('rx').reshape(1,6)
            #spanS = [self.freeMotionSpan('rz'), self.freeMotionSpan('ry'), self.freeMotionSpan('rx')]
            rots = [self.rotZ(angles[0]), self.rotY(angles[1]), self.rotX(angles[2])]
            rot = self.rotZ(angles[0])
                    
            #acumulate rotations
            for i in range(1,3):
                #rot = numpy.dot( rots[i], rot ) #total rotation matrix
                rot = rots[i]*rot #total rotation matrix

                #propagate them to each matrix spanS
                for j in range(i):
                    #spanS[j] =  numpy.dot( rots[i], spanS[j])
                    spanS[j,:] = ( rots[i]*(spanS[j,:].reshape(6,1)) ).reshape(1,6)

            #return a 6x3 matrix
            #spanS = numpy.array(spanS).transpose()
            spanS = spanS.T
        else:
            #spanS = numpy.zeros( (6,3) )
            spanS = numpy.zeros(6,3)

        #if the frame is the global, multiply by the corresponding matrix
        if outputFrame.lower() == 'local':
            return spanS
        elif outputFrame.lower() == 'global':
            #return numpy.dot( numpy.transpose(rot), spanS )
            return (rot.T)*spanS
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
    def jcalc(self, typeJoint, q, unit='rad'):
        
        #Change units
        if isinstance(math.pi, numbers.Number):
            if unit.lower() == 'deg':
                q = math.radians(q)

        if typeJoint.lower() == 'rx':
            XJ = self.rotX(q)
            S = self.freeMotionSpan(typeJoint) 
        elif typeJoint.lower() == 'ry':
            XJ = self.rotY(q)
            S = self.freeMotionSpan(typeJoint)
        elif typeJoint.lower() == 'rz':
            XJ = self.rotZ(q)
            S = self.freeMotionSpan(typeJoint)  
        elif typeJoint.lower() == 'px':
            XJ = self.xlt([q,0,0])
            S = self.freeMotionSpan(typeJoint)  
        elif typeJoint.lower() == 'py':
            XJ = self.xlt([0,q,0])
            S = self.freeMotionSpan(typeJoint)  
        elif typeJoint.lower() == 'pz':
            XJ = self.xlt([0,0,q])
            S = self.freeMotionSpan(typeJoint)
        else:
            print("Joint type not recognized")
            XJ = sympy.eye(6)
            S = sympy.zeros(6)

        return XJ,S

    '''
    Xpts(pluX, pts): Transform points between coordinate frames.
    'pts' can be a vector or a list of vectors.
    'pluX' is a 6x6 Plucker transformation.
    points 'pts' are expressed in reference frame A, and pluX is a transformation from frame A to B.
    Output is a list of points w.r.t. frame B.
    '''
    def Xpts(self, pluX, pts):

        #Change type if necessary
        if type(pluX) == list:
            #pluX = numpy.array(pluX).astype(float)
            pluX = sympy.Matrix(pluX)
        if type(pts) == list:
            pts = numpy.array(pts)#Keep this as an array
            
        E = pluX[0:3,0:3]#Rotation component of pluX
        #r = -self.skew( numpy.dot( numpy.transpose(E), pluX[3:6,0:3] ) )#Translation component of pluX
        r = -self.skew( ( E.T )*( pluX[3:6,0:3] ) )#Translation component of pluX
    
        if pts.ndim == 1:
            #newPoints = numpy.dot( E, pts-r )
            newPoints = E*sympy.Matrix(pts-r)
        else:
            newPoints = []
            #newPoints = [(numpy.dot(E,point-r)) for point in pts]
            for i,point in enumerate(pts):
                #newPoints[i] = numpy.dot(E, point-r)
                point = sympy.Matrix(point)
                #newPoints[i] = E*(point-r)
                newPoints.append( E*(point-r) )
        
        return newPoints
        
    '''
    forwardKinematics(): Automatic process to obtain all transformations from the inertial \
    frame {0} to Body-i. The geometric Jacobians of Body i represents the overall motion and \
    not only the CoM (w.r.t. local coordinates). Use createModel() first.
    'q' is the vector of generalized coordinates (including floating base)
    '''
    def forwardKinematics(self, q, unit='rad'):
        
        #Change types and unit if necessary
        if type(q) == list:
            q = numpy.array(q)
        if unit.lower() == 'deg':
            q = numpy.radians(q)
            
        #Change to sympy
        q = sympy.Matrix(q)
        
        if self._modelCreated:
            
#            parentArray = self.model['parents']
#            Xup = numpy.zeros( (self.model['nB'], 6, 6) )
#            S = numpy.zeros( (self.model['nB'], 6) )
#            X0 = numpy.zeros( (self.model['nB'], 6, 6) )
#            invX0 = numpy.zeros( (self.model['nB'], 6, 6) )
#            jacobian = numpy.zeros( (self.model['nB'], 6, self.model['nB']) )
            parentArray = self.model['parents']
            Xup = [sympy.zeros( 6, 6 ) for i in range(self.model['nB'])]
            S = [sympy.zeros( 6, 1 ) for i in range(self.model['nB'])]
            X0 = [sympy.zeros( 6, 6 ) for i in range(self.model['nB'])]
            invX0 = [sympy.zeros( 6, 6 ) for i in range(self.model['nB'])]
            jacobian = [sympy.zeros( 6, self.model['nB'] ) for i in range(self.model['nB'])]
            
            for i in range(self.model['nB']):
                
                #Obtain JX and S
                XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i])
                
                S[i] = tempS
                #Xup[i] = numpy.dot( XJ, numpy.array( (self.model['XT'])[i] ) )
                Xup[i] = XJ*( numpy.array( (self.model['XT'])[i] ) )
                
                #If the parent is the base
                if parentArray[i] == -1:
                    X0[i] = Xup[i]
                else:
                    #X0[i] = numpy.dot( Xup[i], X0[i-1] )
                    X0[i] =  Xup[i]*X0[i-1]
                    
                #Obtain inverse mappings
                #invX0[i] = numpy.linalg.inv(X0[i])
                #invX0[i] = (X0[i]).inv()
                invX0[i] = self.invPluX(X0[i], None)
                
                #We change the previous S into local coordinates of Body-i
                for j in range(i):
                    #S[j] = numpy.dot( Xup[i], S[j] )
                    S[j] = Xup[i]*S[j]
                    
                #jacobian[i] = numpy.transpose(S)
                jacobian[i] = sympy.Matrix( numpy.transpose(S) )
                pass
                
        else:
            print("Model not yet created. Use createModel() first.")
            return
        
        #Return as numpy arrays. Convert to sympy Matrices outside if necessary
        return Xup,X0,invX0,jacobian
        
    '''
    forwardKinematics_sym(): (Symbolic version) Automatic process to obtain all transformations from the inertial \
    frame {0} to Body-i. The geometric Jacobians of Body i represents the overall motion and \
    not only the CoM (w.r.t. local coordinates). Use createModel() first.
    'q' is the list of symbols (or dynamicsymbols) representing the generalized coordinates (including floating base)
    '''
    def forwardKinematics_sym(self, q, unit='rad'):
        
        #Change types and unit if necessary
        if type(q) == list:
            q = numpy.array(q)
        if unit.lower() == 'deg':
            q = numpy.radians(q)
        
        if self._modelCreated:
            
            parentArray = self.model['parents']
            Xup = [sympy.zeros( 6,6 ) for i in range(self.model['nB'])]
            S = numpy.zeros( (self.model['nB'], 6) ).astype(numpy.object)
            X0 = [sympy.zeros( 6,6 ) for i in range(self.model['nB'])]
            invX0 = [sympy.zeros( 6,6 ) for i in range(self.model['nB'])]
            jacobian = [sympy.zeros( 6,self.model['nB'] ) for i in range(self.model['nB'])]
            
            for i in range(self.model['nB']):
                XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i])
                S[i] = tempS

                XJ = sympy.Matrix(XJ)
                XT = sympy.Matrix( (self.model['XT'])[i] )
                Xup[i] = XJ*XT
                
                #If the parent is the base
                if parentArray[i] == -1:
                    X0[i] = Xup[i]
                else:
                    X0[i] = Xup[i]*X0[i-1]
                    
                #Obtain inverse mappings
                invX0[i] = X0[i].inv()
                
                #We change the previous S into local coordinates of Body-i
                for j in range(i):
                    S[j] = numpy.transpose( numpy.array( (Xup[i])*(sympy.Matrix(S[j])) ) )

                jacobian[i] = sympy.Matrix( numpy.transpose(S) )
                
        else:
            print("Model not yet created. Use createModel() first.")
            return
        
        return Xup,X0,invX0,jacobian

    #TODO: DHParamsTranslator
        
    '''
    crossM[v]: Spatial cross product for MOTION vectors. 
    The input ' v' is a twist (either 3 D or 6 D).*)
    '''
    def crossM(self, v):
        
        v = numpy.array(v)
            
        if v.size == 6:
            out11 = self.skew(v[0:3])
            out12 = numpy.zeros( (3,3) )
            out21 = self.skew(v[3:6])
            out22 = out11
            
            out =  numpy.bmat([[out11, out12],[out21, out22]])
        else:
            out = numpy.array( [[0,0,0],[v[2],0,-v[0]],[-v[1],v[0],0]] )
        
        #return out
        return sympy.Matrix(out)
        
    '''
    crossF[v]. Spatial cross product for FORCE vectors. 
    The input ' v' is a twist (either 3 D or 6 D).
    '''
    def crossF(self, v):
        #return -numpy.transpose(self.crossM(v))
        return -(self.crossM(v)).T

    '''
    inertiaTensor[params, type, connectivity] : Calculate inertia tensor of a body
    Assume input is an array of parameters in the form:
    params={mass, l, w, h, r} and typeObj is a string (e.g., "SolidCylinder")
    l:= length along x     w:= length along y     h:= length along z      r:=radius
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
        #inertiaT = sympy.Matrix( [[Ixx,0.0,0.0],[0.0,Iyy,0.0],[0.0,0.0,Izz]] )
        
        return mass*inertiaT

    '''
    rbi(m,c, I): RigidBodyInertia of a body.
    'mass' is the mass, 'center' is the position of the CoM (2D or 3D),
    inertiaT is the rotational inertia around CoM (can be obtained with intertiaTensor())
    '''
    def rbi(self, mass, center, inertiaT):
        
        #If the input is a 3D vector, obtain the 6x6 rbi
        if len(center) == 3:
            skewC = self.skew(center)
            skewC = numpy.array(skewC)
            tranSkewC = numpy.transpose(skewC)
            
            out11 = inertiaT + mass*numpy.dot( skewC,tranSkewC )
            out12 = mass*skewC
            out21 = mass*tranSkewC
            out22 = mass*numpy.identity(3)
            
            out =  numpy.bmat([[out11, out12],[out21, out22]])
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
            print("Wrong dimenstions")
            out = numpy.zeros((3,3))
            
        return sympy.Matrix(out)
        
    '''
    contactConstraints(): Create spanning matrix T for forces and for free motions S. 
    Assumes contact is in the y-direction of the local frame.
    '''
    def contactConstraints(self, contactType, normalAxis='y'):
        
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
            
        return T,S
        
    #def constrainedSubspace(self, constraintsInformation, Jacobians, beta, velocities, unit='rad'):
    #TODO: complete
    def constrainedSubspace(self, constraintsInformation, Jacobians, unit='rad'):
        
        if self._modelCreated:
            
            nc = constraintsInformation['nc']
            dof = self.model['DoF']
            
            #Constraint matrix (nc times nDoF)
            A = sympy.zeros( nc,dof )
    
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
                    T,S = self.contactConstraints(contactModel)
                    #A[i] = reduce(numpy.dot, [numpy.transpose(T), self.xlt(contactPoint), Jacobians[constrainedBody]])
                    A[i,:] = ( T.T )*( self.xlt(contactPoint) )*( Jacobians[constrainedBody] )
                elif constraintType.lower() == 'non-slippagewithfriction':
                    T,S = self.contactConstraints(contactModel)
                    #A[i] = reduce(numpy.dot, [numpy.transpose(T), self.xlt(contactPoint), Jacobians[constrainedBody]])
                    A[i,:] = ( T.T )*( self.xlt(contactPoint) )*( Jacobians[constrainedBody] )
                elif constraintType.lower() == 'bodycontact':
                    T,S = self.contactConstraints(contactModel)
                    if contactingSide.lower() == 'left':
                        #A[i] = reduce(numpy.dot, [numpy.transpose(T), self.pluX(self.rz(contactAngle), contactPoint), Jacobians[constrainedBody]])
                        A[i,:] = ( T.T )*( self.pluX(self.rz(contactAngle), contactPoint) )*( Jacobians[constrainedBody] )
                    elif contactingSide.lower() == 'right':
                        contactAngle = contactAngle + math.pi
                        #A[i] = reduce(numpy.dot, [numpy.transpose(T), self.pluX(self.rz(contactAngle), contactPoint), Jacobians[constrainedBody]])
                        A[i,:] = ( T.T )*( self.pluX(self.rz(contactAngle), contactPoint) )*( Jacobians[constrainedBody] )
                    else:
                        print('Wrong side')
                else:
                    print('Wrong contraint type')
            
            # How to get derivative of the constraint matrix? derConstraintMatrixA = d A/dt
            derA = A.diff(sympy.symbols('t'))
            
            # kappa = numpy.dot( derConstraintMatrixA, velocities )
            kappa = derA*(sympy.Matrix(self.dq))
            
            # kappa_stab = beta*numpy.dot( constraintMatrix, velocities )
            beta = 0.1
            kappa_stab = beta*kappa
                   
            return A, kappa
        else:
            print("Model not yet created. Use createModel() first.")
            return
            
#    def ID(self, q, qd, qdd, fext = []):
#        
#        #change list into numpy.ndarray, if necessary
#        if type(q) == list:
#            q = numpy.array( q )
#        if type(qd) == list:
#            qd = numpy.array( qd )
#        if type(q) == list:
#            qdd = numpy.array( qdd )
#        if type(fext) == list:
#            fext = numpy.array( fext )
#
#        # Only continue if createModel has been called                
#        if self._modelCreated:
#            
#            dof = self.model['DoF']
#            nBodies = self.model['nB']
#         
#            Xup = numpy.zeros( (nBodies, 6, 6) )
#            S = numpy.zeros( (nBodies, 6) )
#            X0 = numpy.zeros( (nBodies, 6, 6) )
#            parentArray = self.model['parents']
#            
#            v = numpy.zeros( (nBodies, 6) )
#            a = numpy.zeros( (nBodies, 6) )
#            f = numpy.zeros( (nBodies, 6) )
#            
#            tau = numpy.zeros( dof )
#            aGrav = self.model['inertialGrav']
#            
#            if fext.size == 0:
#                fext = numpy.zeros( (nBodies, 6) )
#                
#            for i in range(nBodies):
#                
#                XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i])
#                #S[i] = numpy.reshape(tempS, (6,1))
#                S[i] = tempS
#                vJ = S[i]*qd[i]
#                Xup[i] = numpy.dot( XJ, (self.model['XT'])[i] )
#                
#                #If the parent is the base
#                if parentArray[i] == -1:
#                    X0[i] = Xup[i]
#                    v[i] = vJ
#                    a[i] = numpy.dot( Xup[i], -aGrav ) + S[i]*qdd[i]
#                else:
#                    X0[i] = numpy.dot( Xup[i], X0[i-1] )
#                    v[i] = numpy.dot( Xup[i], v[parentArray[i]] ) + vJ
#                    a[i] = numpy.dot( Xup[i], a[parentArray[i]] ) + S[i]*qdd[i] + numpy.dot( self.crossM(v[i]), vJ )
#                
#                RBInertia = (self.model['rbInertia'])[i]
#                
#                f1 = numpy.dot( RBInertia, a[i] )
#                f2 = reduce(numpy.dot, [self.crossF(v[i]), RBInertia, v[i]])
#                f3 = numpy.dot( numpy.transpose( numpy.linalg.inv( X0[i] ) ), fext[i] )
#                
#                f[i] = f1 + f2 - f3
#                #f[i] = numpy.dot( RBInertia, a[i] ) + reduce(numpy.dot, [self.crossF(v[i]), RBInertia, v[i]]) - numpy.dot( numpy.transpose( numpy.linalg.inv( X0[i] ) ), fext[i] )
#
#            for i in range(nBodies-1,-1,-1):
#                tau[i] = numpy.dot( S[i], f[i] )
#                
#                #If the parent is not the base
#                if parentArray[i] != -1:
#                    f[parentArray[i]] = f[parentArray[i]] + numpy.dot( numpy.transpose(Xup[i]), f[i] )
#                    
#            return tau
#                
#            
#        else:
#            print("Model not yet created. Use createModel() first.")
#            return
    
    def ID(self, q, qd, qdd, fext = [], gravTerms=True):
        
        #change list into sympy.Matrix
        q = sympy.Matrix( q )
        qd = sympy.Matrix( qd )
        qdd = sympy.Matrix( qdd )
        fext = sympy.Matrix( fext ) #Each row represents a wrench applied to a body

        # Only continue if createModel has been called                
        if self._modelCreated:
            
            dof = self.model['DoF']
            nBodies = self.model['nB']
         
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
                
                XJ,tempS = self.jcalc((self.model['jointType'])[i], q[i])
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
                    a[i] = Xup[i]*a[parentArray[i]] + S[i]*qdd[i] + self.crossM(v[i])*vJ
                    
                    
                RBInertia = (self.model['rbInertia'])[i]
                
                #f[i] = numpy.dot( RBInertia, a[i] ) + reduce(numpy.dot, [self.crossF(v[i]), RBInertia, v[i]]) - numpy.dot( numpy.transpose( numpy.linalg.inv( X0[i] ) ), fext[i] )
                f1 = RBInertia*a[i]
                f2 = ( self.crossF(v[i]) )*( RBInertia )*( v[i] )
                f3 = ( (self.invPluX(X0[i])).T )*( fext[i] )
                
                f[i] = f1 + f2 - f3

            for i in range(nBodies-1,-1,-1):
                tau[i] = (S[i].T)*f[i]
                
                #If the parent is not the base
                if parentArray[i] != -1:
                    f[parentArray[i]] = f[parentArray[i]] + ( (Xup[i]).T )*(f[i] )
                    
            return tau
                
            
        else:
            print("Model not yet created. Use createModel() first.")
            return
    
#---------------Tested until here [2017-09-13]
    
    '''
    HandC(q,qd,fext,gravityTerms): Coefficients of the eqns. of motion.
    gravityTerms is a boolean variable to decide if gravitational terms should be included in the output.
    Set as False if only Coriolis/centripetal effects are desired.
    '''
    # FIXME: Symbolic values raise an error in Xup[i] = numpy.dot( XJ, XT ), but works outside of the function. why?
    def HandC(self, q, qd, fext = [], gravityTerms = True):
        
        #change list into numpy.ndarray, if necessary
        if type(q) == list:
            q = numpy.array( q )
        if type(qd) == list:
            qd = numpy.array( qd )
        if type(fext) == list:
            fext = numpy.array( fext )
            
        # Check if we received a symbolic vector
        _symbolic = False
        for quantity in q:
            if isinstance(quantity, tuple(sympy.core.all_classes)):
                _symbolic = True
                break

        # Only continue if createModel has been called                
        if self._modelCreated:
            
            dof = self.model['DoF']
            nBodies = self.model['nB']
         
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
                S[i] = tempS
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
                    
            # If the values received where symbolic, save H and C
            if _symbolic:
                self.H = H
                self.Cor = Cor
                    
            return H,Cor
            
        else:
            print("Model not yet created. Use createModel() first.")
            return
    
    '''
    FDcrb(q, qd, tau, fext): Composite-rigid-Body algorithm. Only works with numerical values
    '''    
    def FDcrb(self, q, qd, tau, fext = []):
        
        localH, localCor = self.HandC(q, qd, fext, gravityTerms=True)
        
        RHS = tau -localCor
        ddq = numpy.linalg.solve( localH, RHS )
        
        return ddq
        
    '''
    FDab(q, qd, tau, fext, gravityTerms):
    '''
    def FDab(self, q, qd, tau, fext = [], gravityTerms=True):
        
        #change list into numpy.ndarray, if necessary
        if type(q) == list:
            q = numpy.array( q )
        if type(qd) == list:
            qd = numpy.array( qd )
        if type(fext) == list:
            fext = numpy.array( fext )

        # Only continue if createModel has been called                
        if self._modelCreated:
            
            dof = self.model['DoF']
            nBodies = self.model['nB']
         
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
                S[i] = tempS
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
            
            return qdd
            
        else:
            print("Model not yet created. Use createModel() first.")
            return
    
    #-------------------------------------------------------------------
            
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
