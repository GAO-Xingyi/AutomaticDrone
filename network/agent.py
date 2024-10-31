import cv2
from network.network import *
import airsim, time
import random
import matplotlib.pyplot as plt
from util.transformations import euler_from_quaternion
import importlib

from aux_functions import get_CustomImage, get_MonocularImageRGB, get_StereoImageRGB


class PedraAgent():
    def __init__(self, cfg, client, name, vehicle_name):
        self.env_type = cfg.env_type
        self.input_size = cfg.input_size
        self.num_actions = cfg.num_actions
        self.iter = 0
        self.vehicle_name = vehicle_name
        self.client = client
        half_name = name.replace(vehicle_name, '')
        print('Initializing ', half_name)

        ###########################################################################
        # Network related modules: Class
        ###########################################################################
        network = importlib.import_module('network.network_models')
        net_mod = 'network.' + 'initialize_network_' + cfg.algorithm + '(cfg, name, vehicle_name)'

        self.network_model = eval(net_mod)

    ###########################################################################
    # Drone related modules
    ###########################################################################

    def take_action(self, action, num_actions, Mode):
        # Mode
        # static: The drone moves by position. The position corresponding to the action
        # is calculated and the drone moves to the position and remains still
        # dynaic: The drone moves by velocity. The velocity corresponding to the action
        # is calculated and teh drone executes the same velocity command until next velocity
        # command is executed to overwrite it.

        # Set Paramaters
        fov_v = (45 * np.pi / 180) / 1.5
        fov_h = (80 * np.pi / 180) / 1.5
        r = 0.4

        ignore_collision = False
        sqrt_num_actions = np.sqrt(num_actions)

        posit = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        pos = posit.position
        orientation = posit.orientation

        quat = (orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val)
        eulers = euler_from_quaternion(quat)
        alpha = eulers[2]

        theta_ind = int(action[0] / sqrt_num_actions)
        psi_ind = action[0] % sqrt_num_actions

        theta = fov_v / sqrt_num_actions * (theta_ind - (sqrt_num_actions - 1) / 2)
        psi = fov_h / sqrt_num_actions * (psi_ind - (sqrt_num_actions - 1) / 2)

        if Mode == 'static':
            noise_theta = (fov_v / sqrt_num_actions) / 6
            noise_psi = (fov_h / sqrt_num_actions) / 6

            psi = psi + random.uniform(-1, 1) * noise_psi
            theta = theta + random.uniform(-1, 1) * noise_theta

            x = pos.x_val + r * np.cos(alpha + psi)
            y = pos.y_val + r * np.sin(alpha + psi)
            z = pos.z_val + r * np.sin(theta)  # -ve because Unreal has -ve z direction going upwards

            self.client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, alpha + psi)),
                ignore_collison=ignore_collision, vehicle_name=self.vehicle_name)
        elif Mode == 'dynamic':
            r_infer = 0.4
            vx = r_infer * np.cos(alpha + psi)
            vy = r_infer * np.sin(alpha + psi)
            vz = r_infer * np.sin(theta)
            # TODO
            # Take average of previous velocities and current to smoothen out drone movement.
            self.client.moveByVelocityAsync(vx=vx, vy=vy, vz=vz, duration=1,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=False,
                                                                    yaw_or_rate=180 * (alpha + psi) / np.pi),
                                            vehicle_name=self.vehicle_name)
            time.sleep(0.07)
            self.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1, vehicle_name=self.vehicle_name)

    def get_CustomDepth(self, cfg):
        camera_name = 2
        if cfg.env_type == 'indoor' or cfg.env_type == 'Indoor':
            max_tries = 5
            tries = 0
            correct = False
            while not correct and tries < max_tries:
                tries += 1
                responses = self.client.simGetImages(
                    [airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, False, False)],
                    vehicle_name=self.vehicle_name)
                img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
                # AirSim bug: Sometimes it returns invalid depth map with a few 255 and all 0s
                if np.max(img1d) == 255 and np.mean(img1d) < 0.05:
                    correct = False
                else:
                    correct = True
            depth = img1d.reshape(responses[0].height, responses[0].width, 3)[:, :, 0]
            thresh = 50
        elif cfg.env_type == 'outdoor' or cfg.env_type == 'Outdoor':
            responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthPlanner, True)],
                                                 vehicle_name=self.vehicle_name)
            depth = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width,
                                                  responses[0].height)
            thresh = 50

        # To make sure the wall leaks in the unreal environment doesn't mess up with the reward function
        super_threshold_indices = depth > thresh
        depth[super_threshold_indices] = thresh
        depth = depth / thresh

        return depth, thresh

    def get_state(self):

        camera_image = get_MonocularImageRGB(self.client, self.vehicle_name)
        self.iter = self.iter + 1
        state = cv2.resize(camera_image, (self.input_size, self.input_size), cv2.INTER_LINEAR)
        state = cv2.normalize(state, state, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        state_rgb = []
        state_rgb.append(state[:, :, 0:3])
        state_rgb = np.array(state_rgb)
        state_rgb = state_rgb.astype('float32')

        return state_rgb

    def GetAgentState(self):
        return self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)

    ###########################################################################
    # RL related modules
    ###########################################################################

    def avg_depth(self, depth_map1, thresh, debug, cfg):
        # Version 0.3 - NAN issue resolved
        # Thresholded depth map to ignore objects too far and give them a constant value
        # Globally (not locally as in the version 0.1) Normalise the thresholded map between 0 and 1
        # Threshold depends on the environment nature (indoor/ outdoor)
        depth_map = depth_map1
        global_depth = np.mean(depth_map)
        n = max(global_depth * thresh / 3, 1)
        H = np.size(depth_map, 0)
        W = np.size(depth_map, 1)
        grid_size = (np.array([H, W]) / n)

        # scale by 0.9 to select the window towards top from the mid line
        h = max(int(0.9 * H * (n - 1) / (2 * n)), 0)
        w = max(int(W * (n - 1) / (2 * n)), 0)
        grid_location = [h, w]

        x_start = int(round(grid_location[0]))
        y_start_center = int(round(grid_location[1]))
        x_end = int(round(grid_location[0] + grid_size[0]))
        y_start_right = min(int(round(grid_location[1] + grid_size[1])), W)
        y_start_left = max(int(round(grid_location[1] - grid_size[1])), 0)
        y_end_right = min(int(round(grid_location[1] + 2 * grid_size[1])), W)

        fract_min = 0.05

        L_map = depth_map[x_start:x_end, y_start_left:y_start_center]
        C_map = depth_map[x_start:x_end, y_start_center:y_start_right]
        R_map = depth_map[x_start:x_end, y_start_right:y_end_right]

        if not L_map.any():
            L1 = 0
        else:
            L_sort = np.sort(L_map.flatten())
            end_ind = int(np.round(fract_min * len(L_sort)))
            L1 = np.mean(L_sort[0:end_ind])

        if not R_map.any():
            R1 = 0
        else:
            R_sort = np.sort(R_map.flatten())
            end_ind = int(np.round(fract_min * len(R_sort)))
            R1 = np.mean(R_sort[0:end_ind])

        if not C_map.any():
            C1 = 0
        else:
            C_sort = np.sort(C_map.flatten())
            end_ind = int(np.round(fract_min * len(C_sort)))
            C1 = np.mean(C_sort[0:end_ind])

        if debug:
            cv2.rectangle(depth_map1, (y_start_center, x_start), (y_start_right, x_end), (0, 0, 0), 3)
            cv2.rectangle(depth_map1, (y_start_left, x_start), (y_start_center, x_end), (0, 0, 0), 3)
            cv2.rectangle(depth_map1, (y_start_right, x_start), (y_end_right, x_end), (0, 0, 0), 3)

            dispL = str(np.round(L1, 3))
            dispC = str(np.round(C1, 3))
            dispR = str(np.round(R1, 3))
            cv2.putText(depth_map1, dispL, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
            cv2.putText(depth_map1, dispC, (int(W / 2 - 40), 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
            cv2.putText(depth_map1, dispR, (int(W - 80), 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=2)
            cmap = plt.get_cmap('jet')
            depth_map_heat = cmap(depth_map1)
            cv2.imshow('Depth Map: ' + self.vehicle_name, depth_map_heat)
            cv2.waitKey(1)

        return L1, C1, R1

    def reward_gen(self, d_new, action, crash_threshold, thresh, debug, cfg):
        L_new, C_new, R_new = self.avg_depth(d_new, thresh, debug, cfg)
        # For now, lets keep the reward a simple one
        if C_new < crash_threshold:
            done = True
            reward = -1
        else:
            done = False
            if action == 0:
                reward = C_new
            else:
                # reward = C_new/3
                reward = C_new

        return reward, done

    def reward_gen_ActorCritic(self, d_new, action, crash_threshold, thresh, debug, cfg):
        L_new, C_new, R_new = self.avg_depth(d_new, thresh, debug, cfg)
        safe_threshold = crash_threshold * 1.2

        # 判断是否撞墙
        if C_new < crash_threshold:
            done = True
            reward = -100  # 撞墙高惩罚
        else:
            # 初始化奖励和完成标志
            reward = 0
            done = False
            # 鼓励前进，前方空旷时奖励更高
            if action == 0:  # 前进
                if C_new > max(L_new, R_new) and C_new > safe_threshold:
                    reward += C_new * 2  # 鼓励前进到空旷区域
                elif C_new > safe_threshold:
                    reward += C_new  # 前方不完全空旷，仍鼓励前进
                else:
                    reward -= 10
                # elif C_new < safe_threshold:
                #     reward -= C_new

                    # 左右平移（减少使用）给予轻微惩罚
            elif action in [1, 2]:  # 向左或向右平移
                reward -= 10  # 惩罚，因为希望尽量避免左右平移

            # 镜头转动给予少量奖励，鼓励合理调整视角
            elif action in [3, 4]:  # 镜头左/右
                if (action == 3 and R_new > L_new) or (action == 4 and L_new > R_new):
                    reward += 15  # 如果镜头转向较空旷区域，给予奖励
                else:
                    reward -= 15  # 不合理的镜头移动给予惩罚

        # 增加速度奖励（如果速度信息可用）
        if hasattr(self, 'speed'):
            reward += self.speed / 10.0  # 鼓励快速移动

        # 加入少量随机探索奖励，鼓励探索新区域
        reward += np.random.uniform(0, 0.1)

        return reward, done

    def reward_gen_ActorCritic_1(self, d_new, action, crash_threshold, thresh, debug, cfg):
        L_new, C_new, R_new = self.avg_depth(d_new, thresh, debug, cfg)
        safe_threshold = crash_threshold * 20  # 25
        turn_safe_threshold = crash_threshold * 15

        # 初始化奖励和完成标志
        reward = 0
        done = False

        # 判断是否撞墙
        if C_new < crash_threshold:
            done = True
            reward = -50  # 降低撞墙惩罚以避免不稳定
        elif C_new < 12*crash_threshold:  #20
            reward -= 5
        elif C_new < 8*crash_threshold:   #10
            reward -= 10
        else:
            # 鼓励前进：如果前方比左右更空旷，并且超过安全距离，则奖励更高
            if action == 0:  # 前进
                if C_new > 20 * crash_threshold:  # 非常安全 30
                    reward += C_new * 2
                if C_new > max(L_new, R_new) and C_new > safe_threshold:
                    reward += C_new * 1.5 # 鼓励快速通过空旷区域
                elif C_new > safe_threshold:
                    reward += C_new  # 鼓励前进，但稍低奖励
                else:
                    reward -= 5  # 前方不够安全，给予轻微惩罚
                # **新增：保持居中奖励机制**
                if abs(L_new - R_new) < 0.2 * max(L_new, R_new):
                    reward += 10  # 如果无人机在路径中央，给予奖励
                else:
                    reward -= abs(L_new - R_new) * 0.5  # 偏离中央时，增加惩罚

            # 对左右平移进行动态奖励和惩罚
            elif action == 1:  # 左移
                if L_new > C_new and L_new > turn_safe_threshold:
                    reward += 5  # 鼓励向空旷区域移动
                else:
                    reward -= 10  # 惩罚不必要的左右平移

            elif action == 2:  # 右移
                if R_new > C_new and R_new > turn_safe_threshold:
                    reward += 5  # 鼓励向空旷区域移动
                else:
                    reward -= 10  # 同样惩罚无效的左右平移

            # 镜头转动：鼓励合理的视角调整，避免不必要的镜头移动
            elif action in [3, 4]:  # 镜头右/左
                if (action == 3 and R_new > L_new and R_new > 1.2*turn_safe_threshold) :
                    reward += 8  # 鼓励镜头转向更空旷区域
                elif (action == 4 and L_new > R_new and L_new > turn_safe_threshold):
                    reward += 10
                else:
                    reward -= 5  # 不合理的镜头移动轻微惩罚

        # 增加速度奖励（如果速度信息可用）
        if hasattr(self, 'speed'):
            reward += min(self.speed / 10.0, 2.0)  # 限制速度奖励的最大值，避免过快移动

        # 如果有目标位置，增加终点奖励（示例）
        if hasattr(self, 'goal_distance') and self.goal_distance < 1.0:
            reward += 50  # 达到目标位置的高奖励

        # 加入随机探索奖励，适当增加探索激励
        reward += np.random.uniform(0, 0.5)

        return reward, done

    ###########################################################################
    # Network related modules
    ###########################################################################
