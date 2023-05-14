# encoding: utf8
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

from turtlesim_env_base import TurtlesimEnvBase


class TurtlesimEnvSingle(TurtlesimEnvBase):
    def __init__(self):
        super().__init__()

    # TODO-STUDENCI przejechać 1/2 okresu, skręcić, przejechać pozostałą 1/2
    def students_step(self, action, tname):

        twist = Twist()
        twist.linear.x = action[0]
        twist.linear.y = 0
        twist.angular.z = action[1]

        self.set_twist_velocity(tname, twist)
        self.wait_after_move()
        self.set_twist_velocity(tname, twist)
        self.wait_after_move()

    def set_twist_velocity(self, tname, twist):
        self.tapi.setVel(tname, twist)
    
    def wait_after_move(self):
        rospy.sleep(self.WAIT_AFTER_MOVE)

    def step(self, actions, realtime=False):
        self.step_sum += 1

        action = list(actions.values())[0]  # uwzględniamy 1. (i jedyną akcję w słowniku)
        tname = list(self.agents.keys())[0]  # sterujemy 1. (i jedynym) żółwiem

        # pozycja PRZED krokiem sterowania
        pose = self.tapi.getPose(tname)
        _, _, _, fd, _, _ = self.get_road(tname)  # odległość do celu (mogła ulec zmianie)

        # action: [prędkość,skręt]
        if realtime:  # jazda+skręt+jazda+skręt
            self.students_step(action=action, tname=tname)

        else:  # skok+obrót
            # obliczenie i wykonanie przesunięcia
            vx = np.cos(pose.theta + action[1]) * action[0] * self.SEC_PER_STEP
            vy = np.sin(pose.theta + action[1]) * action[0] * self.SEC_PER_STEP
            p = Pose(x=pose.x + vx, y=pose.y + vy, theta=pose.theta + action[1])
            self.tapi.setPose(tname, p, mode='absolute')
            rospy.sleep(self.WAIT_AFTER_MOVE)

        # pozycja PO kroku sterowania
        done = False  # flaga wykrytego końca scenariusza symulacji
        pose1 = self.tapi.getPose(tname)
        self.agents[tname].pose = pose1

        fx1, fy1, fa1, fd1, _, _ = self.get_road(tname)  # warunki drogowe po przemieszczeniu
        vx1 = (pose1.x - pose.x) / self.SEC_PER_STEP  # aktualna prędkość - składowa x
        vy1 = (pose1.y - pose.y) / self.SEC_PER_STEP  # aktualna prędkość - składowa y
        v1 = np.sqrt(vx1 ** 2 + vy1 ** 2)  # aktualny moduł prędkości
        fv1 = np.sqrt(fx1 ** 2 + fy1 ** 2)  # zalecany moduł prędkości

        # wyznaczenie składników funkcji celu
        r1 = min(0, self.SPEED_FINE_RATE * (v1 - fv1))  # kara za przekroczenie prędkości
        r2 = 0
        if fv1 > .001:
            vf1 = (vx1 * fx1 + vy1 * fy1) / fv1  # rzut prędkości faktycznej na zalecaną
            if vf1 > 0:
                r2 = self.SPEED_RWRD_RATE * vf1  # nagroda za jazdę z prądem
            else:
                r2 = -self.SPEED_RVRS_RATE * vf1  # kara za jazdę pod prąd
        r3 = self.DIST_RWRD_RATE * (fd - fd1)  # nagroda za zbliżenie się do celu
        r4 = 0

        if abs(fx1) + abs(fy1) < .01 and fa1 == 1:  # wylądowaliśmy poza trasą
            r4 = self.OUT_OF_TRACK_FINE
            done = True
            # print(done)
        reward = fa1 * (r1 + r2) + r3 + r4
        # print(r3)
        # sp=speed, fl=flow, cl=closing, tr=track
        # print(f'RWD: {reward:.2f} = {fa1:.2f}*(sp{r1:.2f} fl{r2:.2f}) cl{r3:.2f} tr{r4:.2f}')
        if self.step_sum > self.MAX_STEPS:
            done = True
        
        return self.get_map(tname), reward, done


def provide_env():
    """Przygotowanie środowiska dla symulacji jednoagentowej"""
    return TurtlesimEnvSingle()
