"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._next_x             = 0
        self._next_y             = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi
        self._Kv                 = .85
        self._Length             = 3
        self._eps_lookahead      = 10e-3

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        x,y = 0,0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
            x = self._waypoints[min_idx-1][0]
            y = self._waypoints[min_idx-1][1]
        else:
            desired_speed = self._waypoints[-1][2]
            x = self._waypoints[-1][0]
            y = self._waypoints[-1][1]
        self._desired_speed = desired_speed
        self._next_x = x
        self._next_y = y
    
    def goal_waypoint_index(self, x, y, waypoints, lookahead_dis):
        for i in range(len(waypoints)-1,0,-1):
            dis = np.sqrt((x - waypoints[i][0])**2 + (y - waypoints[i][1])**2)
            diff = dis - lookahead_dis
            if (abs(diff) <= self._eps_lookahead) :
                return i
        return len(waypoints)-1
    
    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake
    
    def get_goal_waypoint_index(self, x, y, waypoints, lookahead_dis):
        for i in range(len(waypoints)):
            dis = np.sqrt((x - waypoints[i][0])**2 + (y - waypoints[i][1])**2)
            if abs(dis - lookahead_dis) <= self._eps_lookahead:
                return i
        return len(waypoints)-1

    def get_steering_direction(self, v1, v2):
        corss_prod = v1[0]*v2[1] - v1[1]*v2[0]
        if corss_prod >= 0 :
            return -1
        return 1

    def calculate_steering(self, x, y, yaw, waypoints, v):
        lookahead_dis = min(25,max(3,self._Kv * v))
        idx = self.goal_waypoint_index(x, y, waypoints, lookahead_dis)
        v1 = [waypoints[idx][0] - x, waypoints[idx][1] - y]
        v2 = [np.cos(yaw), np.sin(yaw)]
        inner_prod = v1[0]*v2[0] + v1[1]*v2[1]
        alpha = np.arccos(inner_prod/lookahead_dis)
        if np.isnan(alpha):
            alpha = self.vars.alpha_previous
        if not np.isnan(alpha):
            self.vars.alpha_previous = alpha

        steering = self.get_steering_direction(v1, v2)*np.arctan((2*self._Length*np.sin(alpha))/(self._Kv*v))
        if np.isnan(steering):
            steering = self.vars.steering_previous
        else:
            self.vars.steering_previous = steering
        return steering
    
    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        ######################################################
        # DECLARE USAGE VARIABLES HERE
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('itreation', 0)
        self.vars.create_var('xp', x)
        self.vars.create_var('yp', y)
        self.vars.create_var('stre', 0)
        self.vars.create_var('kp', 0)
        self.vars.create_var('steering_previous', 0)
        self.vars.create_var('alpha_previous', 0)

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            # IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            #curretn_waypoint = waypoints[self.vars.itreation]
            vp = self.vars.v_previous
            kp = self.vars.kp
            error = v_desired - v
            if(error > 0):
                throttle_output = error + .2    
                brake_output    = 0
            else:
                throttle_output = 0 
                brake_output    = 0.2 *  error
                
            ######################################################
            # IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            
            # Change the steer output with the lateral controller. 
            xp = waypoints[0][0]
            yp = waypoints[0][1]
            ld = min(25,max(3,self._Kv * v)) #Kv =2.5
            #ld = 2
            idx = self.goal_waypoint_index( x, y, waypoints, ld)
            xn = waypoints[idx][0]
            yn = waypoints[idx][1]
            ld_real = np.sqrt(np.square(yn-y)+np.square(xn-x)) +1.5 #-1.5Reff Error
            
            #dirc = -1 if (xn-x)*np.cos(yaw)-(yn-y)*np.sin(yaw) >=0 else 1
            #alpha = np.arccos(error/ld_real)      
            #thetaR = np.arccos(((xn-x)*np.cos(yaw))/ld)
            theta = np.arctan((yn-y)/(xn-x))
            thetaR = theta
            if theta < 0:
                alpha =  -yaw + theta 
            else:
                alpha =  yaw + theta
            #sign = -1 if (xp-x)<0 else 1  
            stre = np.arctan(2*self._Length*np.sin(alpha)/(self._Kv * v))
            #print(ld_real,xn-x,yn-y,np.rad2deg(alpha),stre,np.rad2deg(yaw) )
            
            #Stanley Contollrt
            #error_lateral = -(xn - x) * np.sin(yaw) + (yn - y)*np.cos(yaw)
            #error_lateral = np.sqrt(np.square(yp-y)+np.square(xp-x))
            #error = (xn-x)*np.cos(yaw)+ (yn-y)*np.sin(yaw)
            #theta = np.arctan((yn-yp)/(xn-xp)) 
            #thetaR = theta
            #if theta > 0:
            #    alpha =  yaw + theta 
            #else:
            #    alpha =  yaw - theta
            #stre = alpha + np.arctan(4*error_lateral/v)
            #print("X%.4F Y%.4F W%.4F T%.4F TR%.4F A%.4F S%.4F V%.4F D%.4F X%.4F Y%.4F X%.4F Y%.4F" %(xn-x,yn-y,np.rad2deg(yaw),np.rad2deg(theta),np.rad2deg(thetaR),np.rad2deg(alpha),stre,v,ld_real,xp-x,yp-y,xp,yp))
            #print( xn,yn,x,y,np.rad2deg(yaw),np.rad2deg(alpha),np.rad2deg(theta),stre,ld_real)
            steer_output    = self.calculate_steering(x, y, yaw, waypoints, v)

            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        ######################################################
        # STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.v_previous = v  # Store forward speed to be used in next step
        self.vars.itreation += 1
        self.vars.stre = stre
        self.vars.xp = x
        self.vars.yp = y