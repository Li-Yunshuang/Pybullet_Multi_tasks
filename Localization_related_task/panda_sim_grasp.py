#from msvcrt import LK_LOCK
import time
import numpy as np
import math

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

# ll = [-7]*pandaNumDofs
# #upper limits for null space (todo: set them to proper range)
# ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
#ll = [-2.8972,-1.7627,-2.8972,-3.0718,-2.8972,0.0175,-2.8972]
#ul = [2.8972,1.7627,2.8972,0.06981,2.8972,3.7525,2.8972]
ll = [-2.5,-1.5,-2.5,-3.0,-2.5,0.0175,-2.5]
ul = [2.5,1.5,2.5,0.06981,2.5,3.5,2.5]

#range = [5, 3, 5, 3.006981, 5, 3.4825,5]

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
        orn=[0, 0, 0, 1]  #[0,0,0,1]
        self.panda = self.p.loadURDF("franka_panda/panda_1.urdf", np.array([0,0,0])+self.offset, self.p.getQuaternionFromEuler([0, 0, math.pi/2]), useFixedBase=True, flags=flags)
        index = 0
        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.gripper_height = 0.2
        # Close gripper for initialization
        self.setGripper(0)
        #create a constraint to keep the fingers centered
        c = self.p.createConstraint(self.panda,
                          9,
                          self.panda,
                          10,
                          jointType=self.p.JOINT_GEAR,
                          jointAxis=[1, 0, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
        self.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
    
        for j in range(self.p.getNumJoints(self.panda)):
            self.p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.p.getJointInfo(self.panda, j)
            #print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.p.JOINT_PRISMATIC):
                self.p.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1

            if (jointType == self.p.JOINT_REVOLUTE):
                self.p.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1
        self.t = 0.


    def calcJointLocation(self, pos, orn):
        """
        ?????? pos ??? orn ?????????????????????????????? 
        """
        jointPoses = self.p.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll, ul, jr, rp, maxNumIterations=20)
       
        return jointPoses


    def setArm(self, jointPoses):
        """
        ?????????????????????
        """
        for i in range(pandaNumDofs):
            self.p.setJointMotorControl2(self.panda, i, self.p.POSITION_CONTROL, jointPoses[i], force=5 * 240.)
    
    def setGripper(self, finger_target):
        """
        ?????????????????????
        """
        for i in [9,10]:
            self.p.setJointMotorControl2(self.panda, i, self.p.POSITION_CONTROL, finger_target, force= 30)


    def step(self, pos, angle, gripper_w):
        """
        pos: [x, y, z]
        angle: ??????
        gripper_w: ?????????????????????
        """
        
        # ?????????
        # pos = [0.5, 0, 0.3] # ???????????????
        # orn = self.p.getQuaternionFromEuler([math.pi, 0., math.pi / 2])   # ???????????????
        # jointPoses = self.calcJointLocation(pos, orn)
        # print('jointPoses = ', jointPoses)
        # self.setArm(jointPoses)
        # return False

        # ????????????
        self.update_state()
        
        if self.state == 0:
            print('Prepare for PUSHING')
            pos = [pos[0], pos[1], 0.1] # ???????????????
            #orn = self.p.getQuaternionFromEuler([math.pi,0., math.pi / 2])   # ???????????????
            orn = self.p.getQuaternionFromEuler([-math.pi/2,0., math.pi / 2])   # ???????????????
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            return False

        elif self.state == 1:
            print('PUSHING')
            #pos[2] += 0.4
            pos = [pos[0]-0.001, pos[1], 0.1]
            #orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # ???????????????
            orn = self.p.getQuaternionFromEuler([-math.pi/2,0., math.pi / 2])   # ???????????????
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            return False

        elif self.state == 2:
            print('RESET')
            pos = [0.5, 0, 0.4]
            orn = self.p.getQuaternionFromEuler([math.pi,0., math.pi / 2])   # ???????????????
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            self.reset()    # ????????????
            return True

    def localize(self, pos, angle, gripper_w):
        """
        pos: [x, y, z]
        angle: ??????
        gripper_w: ?????????????????????
        """
        # ????????????
        self.update_state()
        
        if self.state == 0:
            #print('Contact')
            #pos = [0.5, 0.5, 0.5]
            pos = [pos[0], pos[1], pos[2]] # ???????????????
            #orn = self.p.getQuaternionFromEuler([math.pi,0., math.pi / 2])   # ???????????????
            #orn = self.p.getQuaternionFromEuler([-math.pi,0, math.pi / 2])   # ??????????????? #-math.pi,0, math.pi / 2  #IN world orn
            orn = self.p.getQuaternionFromEuler([angle[0],angle[1], angle[2]])  
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            return False

        elif self.state == 1:
            #print('RESET')
            pos = [0, 0, 0.5]
            orn = self.p.getQuaternionFromEuler([math.pi,0., math.pi / 2])   # ???????????????
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            self.reset()    # ????????????
            return True

     

    def reset(self):
        """
        ????????????
        """
        self.state = 0
        self.state_t = 0
        self.cur_state = 0


class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, offset):
        """
        0: ????????????
        1: ????????????(???????????????)??????????????????
        2: ????????????
        3: ???????????????
        4: ????????????(???????????????)
        
        5: x???????????????
        6: ????????????(???????????????)
        7: x???????????????
        8: ????????????(???????????????)

        9: ????????????
        10: ???????????????
        """
        PandaSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        # State sequence
        self.states=[0, 1, 2, 3]
        self.state_durations=[2, 3, 3, 3]
    
    def update_state(self):
        self.state_t += self.control_dt
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
            self.state_t = 0
            self.state=self.states[self.cur_state]
            #print("self.state=",self.state)
