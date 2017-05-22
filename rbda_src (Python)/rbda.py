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

from rbda import rbdaClass
plu = rbdaClass()

#Start using the functions. Example:
plu.rx(45, unit='deg')

----------------------------------------------------------------------------
Log:

[2017-05-22]:   Created v0.2
                -clean code
                -created and tested: jcalc(), Xpts(), inertiaTensor(), rbi()
[2017-05-15]:   Created v0.1
                -Start version control
                -Review and merge results from codenvy (ROBOMECH)

@author: Reyes Fabian
"""

import numpy, math, copy

class rbdaClass():

    #Constructor
    def __init__(self):
        self.__version__ = '0.2.0'
        
    #Created model TODO: Add XT
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
    'jointType':['Rz','Rz','Rz','Rz','Rz']
    }
    
    plu.createModel(False, 5, '[kg m s]', bodiesParams, None, None)
    
    or
    
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
    'jointType':['Rz','Rz']
    }
    
    plu.createModel(True, 2, '[kg m s]', bodiesParams, None, None)
    '''        
    def createModel(self, floatingBaseBool, noJoints, units, bodiesParams, DHParameters, conInformation):
        
        #Create deep copy of the parameters
        self.model = copy.deepcopy(bodiesParams)
        
        self.model['floatingBase'] = floatingBaseBool
        
        # nR: degrees of freedom (DoF) of the robot including the floating base (nFB). 
        # We have n-2 links (bodies) and n-nFB actuated joints*)
        
        #non-actuated DoF of the floating base
        if floatingBaseBool is True:
            self.model['nFB'] = 3
        else:
            self.model['nFB'] = 0
            
        self.model['DoF'] = self.model['nFB'] + noJoints
        self.model['nR'] = self.model['DoF']
        
        #How many bodies are in the system
        if floatingBaseBool is True:
            bodies = noJoints+3#I am considering the virtual bodies too
        else:
            bodies = noJoints
        self.model['nB'] = bodies
        self.model['nA'] = noJoints

        #Decide gravity acceleration based on units
        if units.lower() == '[kg m s]':
            self.model['g'] = 9.8 #gravity acceleration [m/s^2]
        elif units.lower() == '[kg cm s]':
            self.model['g'] = 980 #gravity acceleration [cm/s^2]
        else:
            print("Units not recognized. Using [kg m s]")
            self.model['g'] = 9.8
            
        #Gravity spatial force expressed in inertial frame. 
        #Assume it is in the y-axis, negative direction
        self.model['inertialGrav'] = numpy.array( [0,0,0,0,-(self.model['g']),0] )
        
        #assign parameters. If no floating base, we should have received only parameters of real links
        rbInertia = []
        if floatingBaseBool is False:

            for i in range(bodies):
                
                m = (self.model['mass'])[i]
                center = [(self.model['Lx'])[i],(self.model['Ly'])[i],(self.model['Lz'])[i]]
                inertia = self.inertiaTensor(\
                [m, (self.model['lengths'])[i], (self.model['widths'])[i], (self.model['heights'])[i],(self.model['radius'])[i]], (self.model['shapes'])[i])
                
                rbInertia.append( self.rbi(m, center, inertia) )
        else:
            #If there is a floating base, then there should be two virtual (massless) links
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
            
            for i in range(bodies):
                
                m = (self.model['mass'])[i]
                center = [(self.model['Lx'])[i],(self.model['Ly'])[i],(self.model['Lz'])[i]]
                inertia = self.inertiaTensor(\
                [m, (self.model['lengths'])[i], (self.model['widths'])[i], (self.model['heights'])[i],(self.model['radius'])[i]], (self.model['shapes'])[i])
                
                rbInertia.append( self.rbi(m, center, inertia) )
                
        self.model['rbInertia'] = rbInertia
        self.model['parents'] = range(bodies)#This represents a serial chain
        
        if floatingBaseBool is True:
            self.model['bodiesRealQ'] = [False for i in range(self.model['nFB'] - 1)] + \
            [True for i in range(noJoints+1)]
        else:
            self.model['bodiesRealQ'] =  [True for i in range(noJoints)]
            
        print('Model created')
    
    #three-dimensional rotations
    @staticmethod
    def rx(theta, unit='rad'):

        if unit.lower() == 'deg':
            theta=math.radians(theta)

        c_theta = math.cos(theta)
        s_theta = math.sin(theta)
        
        return numpy.array([[1,0,0],[0,c_theta,s_theta],[0,-s_theta,c_theta]])

    @staticmethod
    def ry(theta, unit='rad'):

        if unit.lower() == 'deg':
            theta=math.radians(theta)

        c_theta = math.cos(theta)
        s_theta = math.sin(theta)
        
        return numpy.array([[c_theta,0,-s_theta],[0,1,0],[s_theta,0,c_theta]])

    @staticmethod
    def rz(theta, unit='rad'):

        if unit.lower() == 'deg':
            theta=math.radians(theta)

        c_theta = math.cos(theta)
        s_theta = math.sin(theta)
        
        return numpy.array([[c_theta,s_theta,0],[-s_theta,c_theta,0],[0,0,1]])

    #general container for six-dimensional rotations. Input 'plE' is a three dimensional rotation matrix
    @staticmethod
    def rot(plE):
         #change list into numpy.ndarray, if necessary
        if type(plE) == list:
            plE = numpy.array( plE )

        #alternative method
        # output = numpy.zeros((6,6))
        # output[0:3,0:3] = output[3:6,3:6] = plE
        # return output

        #obtain output
        zero = numpy.zeros( (3,3) )
        return numpy.bmat([[plE, zero], [zero, plE]])

    #six-dimensional rotations. Input is an angle
    def rotX(self, theta, unit='rad'):
        return self.rot(self.rx(theta, unit))

    #six-dimensional rotations. Input is an angle
    def rotY(self, theta, unit='rad'):
        return self.rot(self.ry(theta, unit))

    #six-dimensional rotations. Input is an angle
    def rotZ(self, theta, unit='rad'):
        return self.rot(self.rz(theta, unit))

    '''
    Cross product operator (rx pronounced r-cross).
    Input 'r' is either a 3D (point) vector or a skew-symmetric (3x3) matrix.
    '''
    @staticmethod
    def skew(r):
        #change list into numpy.ndarray, if necessary
        if type(r) == list:
            r = numpy.array( r )

        if r.ndim == 1:#Change from vector to matrix
            return numpy.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
        elif r.ndim == 2:#Change from matrix to vector
            return 0.5*numpy.array([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])
        else:
            print('Wrong input')
            return [0]

    #Translation transformation. Input is a three-dimensional vector
    def xlt(self, r):
        #change list into numpy.ndarray, if necessary
        if type(r) == list:
            r = numpy.array( r )

        #alternative method
        # output = numpy.identity(6)
        # output[3:6,0:3] = -(self.skew(r))
        # return output

        zero = numpy.zeros( (3,3) )
        identity = numpy.identity(3)

        return numpy.bmat( [[identity, zero],[-(self.skew(r)), identity]] )

    #General Plücker transformation for MOTION vectors.
    #Inputs are a general 3D (or 6D) rotation matrix and a 3D traslation
    def pluX(self, plE, r):

        if type(plE) == list:
            plE = numpy.array(plE)

        #If we received a 3D rotation matrix, change into 6D
        if plE.shape[0] == 3:
            plE = self.rot(plE)

        return numpy.dot( plE, self.xlt(r) )

    '''
    Plücker transformation for FORCE vectors.
    'pluX' is a 6x6 Plucker transformation or 'pluX' is a 3D rotation and 'r' is a translation vector
    '''    
    def pluXf(self, pluX, r=None):

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

            return numpy.bmat([[out11, out12],[out21, out22]])
        else:
            invTrans = numpy.linalg.inv( self.xlt(r) )
            return numpy.dot( self.rot(pluX), numpy.transpose(invTrans) )

    
    #Inverse for pluX
    #TODO: Review
    def invPluX(self, plE, r):
        B_X_A = self.pluX(plE, r)

        out11 = numpy.transpose( B_X_A[0:3,0:3] )
        out12 = numpy.zeros( (3,3) )
        out21 = numpy.transpose( B_X_A[3:6,0:3] )
        out22 = out11

        #return A_X_B
        return numpy.bmat([[out11, out12],[out21, out22]])

    #Inverse for pluXf
    #pluX is a 6x6 Plucker transformation or
    #pluX is a 3-D rotation and r is a translation vector
    #TODO: review
    def invPluXf(self, pluX, r=None):
        B_Xf_A = self.pluXf(pluX, r)

        out11 = numpy.transpose( B_Xf_A[0:3,0:3] )
        out12 = numpy.transpose( B_Xf_A[0:3,3:6] )
        out21 = numpy.zeros( (3,3) )
        out22 = out11

        #return A_Xf_B
        return numpy.bmat([[out11, out12],[out21, out22]])

    #Definitions of free space for one-dimensional joints
    @staticmethod
    def freeMotionSpan(typeMotion):
        if typeMotion.lower() == 'revx':
            return numpy.array([1,0,0,0,0,0]).reshape( (6,1) )
        elif typeMotion.lower() == 'revy':
            return numpy.array([0,1,0,0,0,0]).reshape( (6,1) )
        elif typeMotion.lower() == 'revz':
            return numpy.array([0,0,1,0,0,0]).reshape( (6,1) )
        elif typeMotion.lower() == 'tranx':
            return numpy.array([0,0,0,1,0,0]).reshape( (6,1) )
        elif typeMotion.lower() == 'trany':
            return numpy.array([0,0,0,0,1,0]).reshape( (6,1) )
        elif typeMotion.lower() == 'tranz':
            return numpy.array([0,0,0,0,0,1]).reshape( (6,1) )
        else:
            return numpy.zeros(6)

    #Coordinate transformation. Similarity or congruence
    def coordinateTransform(self, X, A, transType, inputType):
        #A.dot(B).dot(C)
        #reduce(numpy.dot, [A1, A2, ..., An])
        #multi_dot([A1d, B, C, D])#

        if transType.lower()=='similarity':
            if inputType.lower()=='motion':
                return reduce(numpy.dot, [X, A, numpy.linalg.inv(X)])
            elif inputType.lower()=='force':
                return reduce(numpy.dot, [self.pluXf(X), A, self.pluXf( numpy.linalg.inv(X))])
            else:
                print('Incorrect input type')
                return numpy.zeros((6,6))
        elif transType.lower()=='congruence':
            if inputType.lower()=='motion':
                return reduce(numpy.dot, [self.pluXf(X), A, numpy.linalg.inv(X)])
            elif inputType.lower()=='force':
                return reduce(numpy.dot, [X, A, self.pluXf( numpy.linalg.inv(X))])
            else:
                print('Incorrect input type')
                return numpy.zeros((6,6))
        else:
            print('Incorrect transformation type')
            return numpy.zeros((6,6))

    #6D rotation matrix corrresponding to a spherical joint
    def eulerRot(self, angles, typeRot, outputFrame ='local', unit='rad'):

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

        if outputFrame.lower() == 'local':
            return rotation
        elif outputFrame.lower() == 'global':
            return numpy.transpose(rotation)
        else:
            print('Desired output frame not recognized. Returning local')
            return rotation

    #Jacobian corresponding to a rotation matrix from a spherical joint
    def eulerJacobian(self, angles, typeRot, outputFrame = 'local', unit='rad'):

        #Change type if necessary
        if type(angles) == list:
            angles = numpy.array(angles)

        #Change units
        if unit.lower() == 'deg':
            angles = numpy.radians(angles)

        if typeRot.lower() == 'zyx':
            spanS = [self.freeMotionSpan('revz'), self.freeMotionSpan('revy'), self.freeMotionSpan('revx')]
            rots = [self.rotZ(angles[0]), self.rotY(angles[1]), self.rotX(angles[2])]
            rot = self.rotZ(angles[0])

            #acumulate rotations
            for i in range(1,3):
                rot = numpy.dot( rots[i], rot ) #total rotation matrix

                #propagate them to each matrix spanS
                for j in range(i):
                    spanS[j] =  numpy.dot( rots[i], spanS[j])

            #return a 6x3 matrix
            spanS = numpy.array(spanS).transpose()
        else:
            spanS = numpy.zeros( (6,3) )

        #if the frame is the global, multiply by the corresponding matrix
        if outputFrame.lower() == 'local':
            return spanS
        elif outputFrame.lower() == 'global':
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
    def jcalc(self,typeJoint,q,unit='rad'):
        
        #Change units
        if unit.lower() == 'deg':
            q = math.radians(q)

        if typeJoint.lower() == 'rx':
            XJ = self.rotX(q)
            S = numpy.array([1,0,0,0,0,0])           
        elif typeJoint.lower() == 'ry':
            XJ = self.rotY(q)
            S = numpy.array([0,1,0,0,0,0])
        elif typeJoint.lower() == 'rz':
            XJ = self.rotZ(q)
            S = numpy.array([0,0,1,0,0,0])  
        elif typeJoint.lower() == 'px':
            XJ = self.xlt([q,0,0])
            S = numpy.array([0,0,0,1,0,0])  
        elif typeJoint.lower() == 'py':
            XJ = self.xlt([0,q,0])
            S = numpy.array([0,0,0,0,1,0])  
        elif typeJoint.lower() == 'pz':
            XJ = self.xlt([0,0,q])
            S = numpy.array([0,0,0,0,0,1])
        else:
            print("Joint type not recognized")
            XJ = numpy.identity(6)
            S = numpy.zeros(6)

        return XJ,S            
        
    '''
    Xpts(pluX, pts): Transform points between coordinate frames.
    'pts' can be a vector or a list of vectors.
    'pluX' is a 6x6 Plucker transformation.
    points 'pts' are expressed in reference frame A, and pluX is a transformation \
    from frame A to B. Output is a list of points w.r.t. frame B.
    '''
    def Xpts(self, pluX, pts):

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
        
    #forwardKinematics
    #DHParamsTranslator
        
    '''
    crossM[v]: Spatial cross product for MOTION vectors. 
    The input ' v' is a twist (either 3 D or 6 D).*)
    '''
    def crossM(self, v):
        #Change type if necessary
        if type(v) == list:
            v = numpy.array(v).astype(float)
            
        if v.size == 6:
            out11 = self.skew(v[0:3])
            out12 = numpy.zeros( (3,3) )
            out21 = self.skew(v[3:6])
            out22 = out11
            
            out =  numpy.bmat([[out11, out12],[out21, out22]])
        else:
            out = numpy.array( [[0,0,0],[v[2],0,-v[0]],[-v[1],v[0],0]] )
            
        return out
        
    '''
    crossF[v]. Spatial cross product for FORCE vectors. 
    The input ' v' is a twist (either 3 D or 6 D).
    '''
    def crossF(self, v):
        return -numpy.transpose(self.crossM(v))

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
            
        return out
        
    #-------------------------------------------------------------------
            
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
