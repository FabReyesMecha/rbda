# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 03:09:34 2017

Small module containing Algorithms for calculating twistes and wrenches.
This module is just a small portion of algorithms available on:
    Rigid Body Dynamic Algorithms, by Featherstone, Roy (2008)

Log:
[2017-05-15]:   Created v0.1
                -Start version control
                -Review and merge results from codenvy (ROBOMECH)

@author: Reyes Fabian
"""

import numpy, math

class rbdaClass():

    #Constructor
    def __init__(self):
        self.__version__ = '0.1.0'

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

    #general container for six-dimensional rotations. Input plE is a three dimensional rotation matrix
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

    #General Plucker transformation for MOTION vectors.
    #Inputs are a general 3-D (or 6D) rotation matrix and a 3-D traslation
    def pluX(self, plE, r):

        if type(plE) == list:
            plE = numpy.array(plE)

        #If we received a 3D rotation matrix, change into 6D
        if plE.shape[0] == 3:
            plE = self.rot(plE)

        return numpy.dot( plE, self.xlt(r) )

    #Plucker transformation for FORCE vectors. Inputs are assumed to be numpy.arrays
    #pluX is a 6x6 Plucker transformation or
    #pluX is a 3-D rotation and r is a translation vector
    def pluXf(self, pluX, r=None):

        #change list into numpy.ndarray, if necessary
        if type(pluX) == list:
            pluX = numpy.array( pluX )
        if r is not None and type(r) == list:
            r = numpy.array( r )

        #If the input is a 6D transformation, just manipulate as necessary
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
    #TODO: review
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
