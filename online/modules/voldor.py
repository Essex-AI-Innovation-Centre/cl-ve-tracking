import numpy as np
from VOLDOR.slam_py.slam_utils import T44_to_T6, T6_to_T44, polish_T44, eval_covisibility
from online.utils.voldor import Frame
import online.shared.pyvoldor_vo as pyvoldor

class VoldorSetup:
    def __init__(self, config, camera):
        self.fid_cur = 0
        self.fid_cur_tmpkf = -1 # temporal key frame
        self.fid_cur_spakf = -1 # spatial key frame
        self.frames = []
        self.camera = camera
        self.Twc_cur = np.eye(4,4,dtype=np.float32)
        self.vo_win_size = config['vo_win_size']
        self.rescale = config['rescale']
        self.vostep_visibility_thresh = config['thresh']['vostep'] # visibility threshold for vo window step
        self.spakf_visibility_thresh = config['thresh']['spakf'] # visibility threshold for creating a new spatial keyframe
        self.depth_covis_conf_thresh = config['thresh']['depth'] # depth confidence threshold for estimating covisibility
        self.meanshift_kernel_var = str(config['meanshift']['kernel_var'])
        self.delta = str(config['meanshift']['delta'])
        self.max_iters = str(config['meanshift']['max_iters'])
        self.voldor_config = f'--silent --meanshift_kernel_var '+self.meanshift_kernel_var+' --delta '+self.delta+' --max_iters '+self.max_iters+' '
        self.voldor_config += f"--pose_sample_min_depth {camera.basefocal/config['disp']['max']} "
        self.voldor_config += f"--pose_sample_max_depth {camera.basefocal/config['disp']['min']} "
        self.voldor_user_config = f"--abs_resize_factor {self.rescale}"
        self.n_frames_registered = self.vo_win_size
        self.test_mode = config['test_mode']

    def voldor_step(self, of_batch, disp=None):
        depth_priors = []
        depth_prior_pconfs = []
        depth_prior_poses = []
        dpkf_list = []
        
        if self.fid_cur_tmpkf >= 0:
            dpkf_list.append(self.fid_cur_tmpkf)
        if self.fid_cur_spakf >= 0 and self.fid_cur_spakf != self.fid_cur_tmpkf:
            dpkf_list.append(self.fid_cur_spakf)
        for fid in dpkf_list:
            if fid >= 0:
                depth_priors.append(self.frames[fid].get_scaled_depth())
                depth_prior_pconfs.append(self.frames[fid].depth_conf)
                depth_prior_poses.append(T44_to_T6(np.linalg.inv(self.Twc_cur @ self.frames[fid].Tcw)))

        # run voldor C++ code
        py_voldor_kwargs = {
            'flows': np.stack(of_batch, axis=0),
            'fx': self.camera.fx, 'fy': self.camera.fy,
            'cx': self.camera.cx, 'cy': self.camera.cy,
            'basefocal': self.camera.basefocal,
            'disparity' : disp,
            'depth_priors' : np.stack(depth_priors, axis=0) if len(depth_priors)>0 else None,
            'depth_prior_pconfs' : np.stack(depth_prior_pconfs, axis=0) if len(depth_prior_pconfs)>0 else None,
            'depth_prior_poses' : np.stack(depth_prior_poses, axis=0) if len(depth_prior_poses)>0 else None,
            'config' : self.voldor_config + ' ' + self.voldor_user_config}
        vo_ret = pyvoldor.voldor(**py_voldor_kwargs)
        
        if vo_ret['n_registered'] == 0 or vo_ret['n_registered'] == 1:
            print(f'Tracking lost at {self.fid_cur}')
            self.fid_cur_tmpkf = -1
            self.fid_cur_spakf = -1
            self.n_frames_registered = 1
            self.lost = True
        else:
            vo_ret['Tc1c2'] = T6_to_T44(vo_ret['poses'])

            # based on covisibility, figure out how many steps to move
            vo_step = 0
            T_tmp = np.eye(4,4,dtype=np.float32)
            for i in range(vo_ret['n_registered']):
                vo_step = vo_step + 1
                T_tmp = vo_ret['Tc1c2'][i] @ T_tmp
                covis = eval_covisibility(vo_ret['depth'], T_tmp, self.camera.K, vo_ret['depth_conf']>self.depth_covis_conf_thresh)
                if covis < self.vostep_visibility_thresh:
                    break
            self.n_frames_registered = vo_step
            self.lost = False

            for i in range(vo_step):
                if i==0:
                    self.frames.append(Frame(np.linalg.inv(self.Twc_cur), vo_ret['depth'], vo_ret['depth_conf']))
                else:
                    self.frames.append(Frame(np.linalg.inv(self.Twc_cur)))

                self.Twc_cur = vo_ret['Tc1c2'][i] @ self.Twc_cur
                polish_T44(self.Twc_cur)

            # based on covisibility, to see if need let current frame be a new spatial keyframe
            if self.fid_cur_spakf >= 0:
                T_spa2cur = self.Twc_cur @ self.frames[self.fid_cur_spakf].Tcw
                covis = eval_covisibility(self.frames[self.fid_cur_spakf].get_scaled_depth(), T_spa2cur, self.camera.K, self.frames[self.fid_cur_spakf].depth_conf > self.depth_covis_conf_thresh)
                if covis < self.spakf_visibility_thresh:
                    self.fid_cur_spakf = self.fid_cur
            else:
                self.fid_cur_spakf = self.fid_cur

            # set temporal kf to current frame, move fid_cur pt
            self.fid_cur_tmpkf = self.fid_cur
            self.fid_cur = self.fid_cur + vo_step
    
        return self.lost
    