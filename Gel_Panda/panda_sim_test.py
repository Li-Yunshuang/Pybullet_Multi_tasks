import time
import numpy as np
import math

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 10
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
# restposes for null space
jointPositions=(0.8045609285966308, 0.525471701354679, -0.02519566900946519, -1.3925086098003587, 0.013443782914225877, 1.9178323512245277, -0.007207024243406651, 0.01999436579245478, 0.019977024051412193)
            # [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

class PandaSim(object):
    def __init__(self, bullet_client, offset):
        self.p = bullet_client
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)
        
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        orn=[0, 0, 0, 1]
        self.panda = self.p.loadURDF("franka_panda/panda_1.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)
        
        index = 0
        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.gripper_height = 0.2
        #create a constraint to keep the fingers centered
        c = self.p.createConstraint(self.panda,
                          9,
                          self.panda,
                          11,
                          jointType=self.p.JOINT_GEAR,
                          jointAxis=[1, 0, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
        self.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
    
        for j in range(self.p.getNumJoints(self.panda)):
            self.p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.p.getJointInfo(self.panda, j)
            # info = self.p.getLinkState(self.panda, j)
            print("info=",info)

            jointName = info[1]
            jointType = info[2]
            if (jointType == self.p.JOINT_PRISMATIC):
                self.p.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1

            if (jointType == self.p.JOINT_REVOLUTE):
                self.p.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1

    def calcJointLocation(self, pos, orn):
        """
        根据 pos 和 orn 计算机械臂的关节位置 
        """
        jointPoses = self.p.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll, ul, jr, rp, maxNumIterations=20)
        # ##############################DEBUG#############################3
        #print(jointPoses)
        return jointPoses


    def setArm(self, jointPoses):
        """
        设置机械臂位置
        """
        for i in range(pandaNumDofs):
            self.p.setJointMotorControl2(self.panda, i, self.p.POSITION_CONTROL, jointPoses[i], force=5 * 240.)
    
    def setGripper(self, finger_target):
        """
        设置机械手位置
        """
        for i in [9,11]:
            self.p.setJointMotorControl2(self.panda, i, self.p.POSITION_CONTROL, finger_target, force= 30)



    def reset(self):
        """
        重置状态
        """
        self.state = 0
        self.state_t = 0
        self.cur_state = 0


class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, offset):
        
        PandaSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        #self.states=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #self.state_durations=[0.5, 0.5, 1, 1, 1, 0.5, 0.1, 0.1, 0.1, 0.5, 0.5]
        self.states=[0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.state_durations=[1, 1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30, 1]
    
    def update_state(self):
        self.state_t += self.control_dt
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
            self.state_t = 0
            self.state=self.states[self.cur_state]
            #print("self.state=",self.state)