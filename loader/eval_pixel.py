#!/usr/bin/env python
import sys
import os
import os.path
import numpy as np
from scipy.io import loadmat,savemat
from sklearn.neighbors import NearestNeighbors
import random
import warnings
import argparse
import math

#set random seed 
random.seed(0)

#########################################################
# meshObj: Data structure to parse the obj file format
# contains: vertices & faces of the mesh 
#          + 51 3D landmarks indicies on the mesh
#####################################################
class meshObj(object):
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.uv = []
        self.landmarks_3D = []

    def write_obj_with_colors(obj_name, vertices, triangles, colors):
        ''' Save 3D face model with texture represented by colors.
        Args:
            obj_name: str
            vertices: shape = (nver, 3)
            triangles: shape = (ntri, 3)
            colors: shape = (nver, 3)
        '''
        triangles = triangles.copy()
        triangles += 1 # meshlab start with 1
    
        if obj_name.split('.')[-1] != 'obj':
            obj_name = obj_name + '.obj'
            
        # write obj
        with open(obj_name, 'w') as f:
            
            # write vertices & colors
            for i in range(vertices.shape[0]):
                # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
                s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
                f.write(s)
    
            # write f: ver ind/ uv ind
            [k, ntri] = triangles.shape
            for i in range(triangles.shape[0]):
                # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
                s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
                f.write(s)

    def load_from_obj(self, path, MeshNum):
        print(path)
        #load the vertices 
        regexp = r"v\s([+-]?[0-9]*[.]?[0-9]+[e]?[+-]?[0-9]*)\s([+-]?[0-9]*[.]?[0-9]+[e]?[+-]?[0-9]*)\s([+-]?[0-9]*[.]?[0-9]+[e]?[+-]?[0-9]*)*"

        # Error handle and check for corrupt formating of file
        try:
                self.vertices = np.fromregex(path + MeshNum+".obj", regexp, ('f'))
        except IOError:
                print ("I/O error: "+path + MeshNum+".obj" +" not found!")
                quit()
        if self.vertices.shape[0] == 0:
                print ("File Format Error: vertices cannot be found in file, or the file is corrupted. Please check that "+\
                                        path + MeshNum+".obj"+ " is in valid wavefront OBJ format.")
                quit()
        #load faces
        regexp = r"f\s(\d+)\/*\d*\/*\d*\s(\d+)\/*\d*\/*\d*\s(\d+)\/*\d*\/*\d*"

        # Error handle and check for corrupt formating of file
        try:
                self.faces = np.fromregex(path + MeshNum+".obj", regexp, ('i')) -1
        except IOError:
                print ("I/O error: "+path + MeshNum+".obj" +" not found!")
                quit()
        if self.faces.shape[0] == 0:
                print ("File Format Error: faces cannot be found in file, or the file is corrupted. Please check that "+\
                                        path + MeshNum+".obj"+ " is in valid wavefront OBJ format.")
                quit()

        #load the 3D landmark indices
        self.landmarks_3D  = np.loadtxt("/".join(path.split("/")[:-1])+"/"+ 'VertexLandmarks'+str(MeshNum)+'.txt', dtype='int32').flatten()

        #check for the number of  landmarks to be 51
        # if self.landmarks_3D.shape[0] != 51:
        #         print ("File Format Error: "+'VertexLandmarks'+str(MeshNum)+'.txt' + ' file does not have exactly 51 landmarks.')
        #         quit()

        # check to make sure free vertices are not refered to in the landmark indices file, otherwise remove all free vertices
        uniq = np.unique(self.faces)
        print(uniq)
        landmarks = []
        for idx, i in enumerate(self.landmarks_3D):
                #print('idx: ',idx)
                #print('i: ',i)
                if i == -1: 
                        landmarks.append(-1)
                        if idx == 13:
                                print ("File Value Error: 3D Landmark indices cannot have an unknown value at index 13, nose tip!")
                                quit()
                        continue
                try:
                        landmarks.append(np.where(i == uniq)[0][0])
                except:
                        print ("File Format Error: 3D Landmark indices allude to free vertices which will be removed.\
                        \nPlease fix VertexLandmarks correspondences to be not use free vertices!")
                        quit()
        landmarks = np.array(landmarks)
        self.landmarks_3D = landmarks

        if self.vertices.shape[0] < uniq.shape[0]:
                print ("File Format Error: File contains references to non-existant vertices!")
                quit()

        ## correct the faces of the meshes ##
        if self.vertices.shape[0] > uniq.shape[0]:
                new_idx = range(0,len(uniq))
                idx_map = dict(zip(uniq,new_idx))
                new_faces = []
                for tri in self.faces:
                        new_tri = [idx_map[tri[0]], idx_map[tri[1]],idx_map[tri[2]]]
                        new_faces.append(new_tri)
                new_faces = np.array(new_faces)
                self.faces = new_faces
        #####

        #filter to only include vertices that are part of a face. NO free vertices
        self.vertices = self.vertices[uniq]
        return 


###################################################################################################
# for source point, get its closest point on ground truth
# input: s, shape = [N,3], source 3D vertices
#		 gt, shape = [M,3], ground truth 3D vertices
#	     nbrs, a kd tree structure for ground truth vertices
# output: gt_closest_P, shape = [N,3], closest point on ground truth for each vertex on s
#		  idx, shape = [N], index of closest point
# 		  dist, shape = [N], distance of each closest point pair
###################################################################################################
def get_closest_point(s,gt,nbrs):
	# land, vert, uv
	dist,idx = nbrs.kneighbors(s)
	idx = np.reshape(idx,[s.shape[0]])
	gt_closest_p = gt[idx,:]

	return gt_closest_p,idx,dist

###################################################################################################
# align source mesh to target mesh
# min ||target - (s*source*R + t)||^2
# input: source, shape = [N,3], source mesh to be aligned
# 		 target, shape = [N,3], target mesh
#		 scale, if True, consider scaling factor for alignment; if False, scale will be set to 1
# output: R, shape = [3,3], rotation matrix
#		  t, shape = [1,3], translation vector
# 		  s, scaling factor
###################################################################################################
def align_source_to_target(source,target,scale = False):

	tar = target.copy()
	sou = source.copy()
	center_tar = tar - np.mean(tar,0) # centralized target mesh
	center_sou = sou - np.mean(sou,0) # centralized source mesh
	# print('center_tar:',center_tar.size)
	# print('center_sou:',center_sou.size)
	W = np.matmul(center_tar.transpose(),center_sou)
	U,S,V = np.linalg.svd(W)
	R = np.matmul(np.matmul(V.transpose(),np.diag([1,1,np.linalg.det(np.matmul(V.transpose(),U.transpose()))])),U.transpose()) # calculate rotation matrix (exclude mirror symmetry)

	if scale:
		R_sou = np.matmul(center_sou,R)
		s = np.sum(R_sou*center_tar)/np.sum(R_sou*R_sou)
	else:
		s = 1

	t = np.mean(tar,0) - s*np.matmul(np.expand_dims(np.mean(sou,0),0),R)

	return R,t,s

###################################################################################################
# iterative closest point
# input: source, shape = [N,3], source vertex to be aligned
# 		 target, shape = [N,3], corresponding target vertex
#		 landmark_s, shape = [K,3], facial landmarks of source mesh
#		 landmark_t, shape = [K,3], facial landmakrs of target mesh
# 		 nbrs, a kd tree structure for target mesh
#		 max_iter, iterations for icp
# output: sou, shape = [N,3], aligned source mesh
# 		  RR, shape = [3,3], rotation matrix
# 		  tt, shape = [1,3], translation
#		  ss, scaling factor
###################################################################################################
def icp(source,target,landmark_s,landmark_t,nbrs,max_iter = 100):

	sou = source.copy()
	tar = target
	i = 0

	# initialize rotation, translation and scale
	tt = np.zeros([1,3]).astype(np.float32)
	RR = np.eye(3)
	ss = 1

	while i < max_iter:
		# using landmarks for alignment in the first step
		if i == 0:
			s_match = landmark_s
			t_match = landmark_t
		# using closest point pairs for alignment from the second step
		else:
			s_match = sou
			t_match,_,_ = get_closest_point(sou,tar,nbrs)

		R,t,s = align_source_to_target(s_match,t_match,scale = True) # calculate scale and rigid transformation

		# accumulate rotation, translation and scaling factor
		RR = np.matmul(RR,R)
		tt = t + s*np.matmul(tt,R)
		ss = ss*s
		sou = s*np.matmul(sou,R) + t

		i += 1

	return sou,RR,tt,ss

###################################################################################################
# calculate point to face distance
# input: source, shape = [N,3], aligned source mesh
# 		 target, shape = [M,3], target mesh
# 		 f_t, shape = [F,3], triangles for target mesh
# 		 nbrs, a kd tree structure for target mesh
# output: error, shape = [n], distance to closest point for each vertex on source mesh
###################################################################################################
def point_to_face_distance(source,target,f_t,nbrs):
        with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                _,idx,dist = get_closest_point(source,target,nbrs) # get closest vertex on target mesh for each vertex of source mesh

                face = f_t
                v1 = target[face[:,0],:]
                v2 = target[face[:,1],:]
                v3 = target[face[:,2],:]

                normals = np.cross(v2-v1,v3-v1,axis = 1)
                normals = normals/np.expand_dims(np.linalg.norm(normals,axis = 1),1) # calculate face normal of target mesh

                error = np.zeros([len(idx)]) # initialize error

                # for each vertex on source mesh, calculate the distance to its closest point
                for i in range(len(idx)):

                        # for each closest vertex on target mesh, find its adjacent triangles 
                        row = np.where(face == idx[i])
                        row = row[0]
                        dist_p = dist[i][0]

                        # if the closest vertex is isolated with others, then treat it as the closest point
                        if row.shape[0] == 0:
                                error[i] = dist_p
                                continue

                        # find edges of adjacent triangles
                        edges = np.concatenate([face[row,:2],face[row,::2],face[row,1:]],axis = 0)
                        r1 = target[edges[:,0],:]
                        r2 = target[edges[:,1],:]

                        # calculate the projection on edges
                        t = np.sum((np.expand_dims(source[i,:],0)-r1)*(r2-r1),1)/np.sum((r2-r1)**2,1)
                        # exclude the projection outside the edge
                        t[(t<=0)|(t>=1)] = np.nan

                        P = r1 + (r2-r1)*np.expand_dims(t,1)
                        D = np.expand_dims(source[i,:],0) - P
                        D = np.sqrt(np.sum(D**2,1))
                        dist_e = np.nanmin(D) # closest distance to edges

                        # calculate baricentric coordinates for projections on each adjacent triangles
                        r1 = target[face[row,0],:]
                        r2 = target[face[row,1],:]
                        r3 = target[face[row,2],:]

                        vq = np.expand_dims(source[i,:],0) - r1
                        D = np.sum(vq*normals[row,:],1)
                        rD = normals[row,:]*np.expand_dims(D,1)

                        r31r31 = np.sum((r3-r1)**2,1)
                        r21r21 = np.sum((r2-r1)**2,1)
                        r21r31 = np.sum((r2-r1)*(r3-r1),1)
                        r31vq = np.sum((r3-r1)*vq,1)
                        r21vq = np.sum((r2-r1)*vq,1)

                        d = r31r31*r21r21 - r21r31**2
                        bary1 = (r21r21*r31vq - r21r31*r21vq)/d
                        bary2 = (r31r31*r21vq - r21r31*r31vq)/d
                        bary3 = 1 - bary1 - bary2

                        # exclude projections outside triangles
                        out_idx = ((bary1<=0) | (bary1>=1) | (bary2<=0) | (bary2>=1) | (bary3<=0) | (bary3>=1)|(np.abs(d)<1e-16)) 

                        D[out_idx] = np.nan
                        dist_f = np.nanmin(np.abs(D)) # closest distance to triangles 
                        error[i] = np.nanmin(np.array([dist_p,dist_e,dist_f])) # distance to closest point

                return error

###################################################################################################
# calculate source to target reconstruction error
# input: source, shape = [N,3], aligned source mesh
# 		 target, shape = [M,3], target mesh
# 		 f_t, shape = [F,3], triangles for target mesh 
#		 landmark_s, shape = [K,3], facial landmarks of source mesh
#		 landmark_t, shape = [K,3], facial landmakrs of target mesh
# output: error, shape = [n], distance to closest point for each vertex on source mesh
# 		  RR, shape = [3,3], rotation matrix
# 		  tt, shape = [1,3], translation
#		  ss, scaling factor
###################################################################################################
def reconstruct_distance(source,target,f_s,f_t,landmark_s,landmark_t):
	max_iter = 100
	nbrsT = NearestNeighbors(n_neighbors = 1, algorithm = 'kd_tree').fit(target) # create kd tree for target mesh

	# do icp with scale to align source mesh
	sou,RR,tt,ss = icp(source,target,landmark_s,landmark_t,nbrsT,max_iter = max_iter)

	# calculate point to mesh error, to reduce computational time, we calculate error every 4 sou vertex 
	errorS2T = point_to_face_distance(sou[::1,:],target,f_t,nbrsT)

	nbrsS = NearestNeighbors(n_neighbors = 1, algorithm = 'kd_tree').fit(sou) # create kd tree for sou mesh after ICP
	# calculate point to mesh error, to reduce computational time, we calculate error every 4 tar vertex
	errorT2S = point_to_face_distance(target[::1,:],sou,f_s,nbrsS)

	return np.nan_to_num(errorS2T),np.nan_to_num(errorT2S),RR,tt,ss

def align2axie(mesh_s):
        sum_x = 0
        sum_y = 0
        sum_z = 0
        dist_max = -1
        for i in range(len(mesh_s.vertices)):
                sum_x = sum_x + mesh_s.vertices[i,0]
                sum_y = sum_y + mesh_s.vertices[i,1]
                sum_z = sum_z + mesh_s.vertices[i,2]
        mean_x = sum_x/len(mesh_s.vertices)
        mean_y = sum_y/len(mesh_s.vertices)
        mean_z = sum_z/len(mesh_s.vertices)
        for i in range(len(mesh_s.vertices)):
                mesh_s.vertices[i,0] = mesh_s.vertices[i,0] - mean_x
                mesh_s.vertices[i,1] = mesh_s.vertices[i,1] - mean_y
                mesh_s.vertices[i,2] = mesh_s.vertices[i,2] - mean_z
        # for i in range(len(mesh_s.vertices)):
        #         dist_ = math.sqrt(mesh_s.vertices[i,0]*mesh_s.vertices[i,0] + mesh_s.vertices[i,1]*mesh_s.vertices[i,1] 
        #                 + mesh_s.vertices[i,2]*mesh_s.vertices[i,2])
        #         if dist_ > dist_max:
        #                 dist_max = dist_
        # for i in range(len(mesh_s.vertices)):
        #         mesh_s.vertices[i,0] = mesh_s.vertices[i,0] / dist_max
        #         mesh_s.vertices[i,1] = mesh_s.vertices[i,1] / dist_max
        #         mesh_s.vertices[i,2] = mesh_s.vertices[i,2] / dist_max
        return mesh_s

####################################################################################################
# calculate source <-> target reconstruction error for one mesh pair only 
# input: test_mesh = string: mesh number of subject
#        res_dir = string: directory path to the location of predicted/reconstructed/source mesh
#        red_dir = string: directory path to the location of ground truth/target mesh
#        reffile_prefix = string : the prefix will be added to the ref mesh filename to be found
#                                  e.g. reffile_prefix='mesh' => meshxxx.obj 
#        resfile_prefix = string : the prefix will be added to the predicted mesh filename to be found
#                                  e.g. resfile_prefix='pred' => predxxx.obj 
# output: error = ARMSE error between the ground truth and predicted meshes 
###################################################################################################
def eval_mesh(test_mesh, res_dir, ref_dir, reffile_prefix='mesh', resfile_prefix='pred'):
        #### load the 2 meshes
        # target
        mesh_t = meshObj()
        mesh_t.load_from_obj(os.path.join(ref_dir,reffile_prefix), test_mesh) #origin
        # source
        mesh_s = meshObj()
        mesh_s.load_from_obj(os.path.join(res_dir,resfile_prefix), test_mesh) #construct
        # mesh_s = align2axie(mesh_s)
        # mesh_t = align2axie(mesh_t)
        for i in range(len(mesh_s.vertices)):
                mesh_s.vertices[i,1] = -mesh_s.vertices[i,1]
        for i in range(len(mesh_t.landmarks_3D)):
                mesh_t.landmarks_3D[i] = mesh_t.landmarks_3D[i] - 1
        # eye_bs = np.sqrt(np.sum(np.square(mesh_s.vertices[mesh_s.landmarks_3D[19]] -mesh_s.vertices[mesh_s.landmarks_3D[28]])))
        # eye_bt = np.sqrt(np.sum(np.square(mesh_t.vertices[mesh_t.landmarks_3D[19]] -mesh_t.vertices[mesh_t.landmarks_3D[28]])))
        # print('s: ',eye_bs)
        # print('t: ',eye_bt)
        #mesh_s.vertices = mesh_s.vertices * eye_bt/eye_bs

        #extract only the vertices of the mesh
        gt_shape = mesh_t.vertices
        reconstruct_shape = mesh_s.vertices 
        # reconstruct_file = open('/media/weepies/Seagate Backup Plus Drive/3DMM/3dp_chosse/3dpixel/view_data/'+test_mesh+'rec.obj','w')
        # gt_file = open('/media/weepies/Seagate Backup Plus Drive/3DMM/3dp_chosse/3dpixel/view_data/'+test_mesh+'gt.obj','w')
        # for i in range(len(reconstruct_shape)):
        #         reconstruct_file.write('v '+str(reconstruct_shape[i][0])+' '+str(reconstruct_shape[i][1])+' '+str(reconstruct_shape[i][2])+'\n')

        # for i in range(len(mesh_s.faces)):
        #         reconstruct_file.write('f '+str(mesh_s.faces[i][0]+1)+' '+str(mesh_s.faces[i][1]+1)+' '+str(mesh_s.faces[i][2]+1)+'\n')
        # reconstruct_file.close()

        # for i in range(len(gt_shape)):
        #         gt_file.write('v '+str(gt_shape[i][0])+' '+str(gt_shape[i][1])+' '+str(gt_shape[i][2])+'\n')

        # for i in range(len(mesh_t.faces)):
        #         gt_file.write('f '+str(mesh_t.faces[i][0]+1)+' '+str(mesh_t.faces[i][1]+1)+' '+str(mesh_t.faces[i][2]+1)+'\n')
        # gt_file.close()
        #load the landmarks for rough alignment 
        landmark_s = mesh_s.vertices[mesh_s.landmarks_3D]
        landmark_t = mesh_t.vertices[mesh_t.landmarks_3D]
        selInds = np.union1d(np.where(mesh_s.landmarks_3D ==-1)[0], np.where(mesh_t.landmarks_3D ==-1)[0])

        # ignore landmarks that were -1 in the target
        landmark_s_4crop  = np.delete(landmark_s, selInds, axis=0)
        landmark_t_4crop  = np.delete(landmark_t, selInds, axis=0)
        # Get rough aligned pred mesh with the GT
        nbrsT_4crop = NearestNeighbors(n_neighbors = 1, algorithm = 'kd_tree').fit(gt_shape)
        reconstruct_shape,r ,t ,s = icp(reconstruct_shape,gt_shape,landmark_s_4crop,landmark_t_4crop,nbrsT_4crop,max_iter = 1)
        landmark_s = s*np.matmul(landmark_s,r) + t

        #### ONLY for the reconstructed/pred mesh we
        # crop to 95 mm around the nose tip (landmark 13th/51)
        center_s = landmark_s[13] + 0.3*(landmark_s[10]-landmark_s[13])
        center_t = landmark_t[13] + 0.3*(landmark_t[10]-landmark_t[13])
        nose_dist_s = np.sqrt(np.sum(np.square(landmark_s[10]-landmark_s[13])))
        eye_dist_s = np.sqrt(np.sum(np.square(landmark_s[19]-landmark_s[28])))
        r_s = 1.2*(nose_dist_s+eye_dist_s)/2
        nose_dist_t = np.sqrt(np.sum(np.square(landmark_t[10]-landmark_t[13])))
        eye_dist_t = np.sqrt(np.sum(np.square(landmark_t[19]-landmark_t[28])))
        r_t = 1.2*(nose_dist_t+eye_dist_t)/2
        crop_size_t = r_t
        crop_size_s = r_s
        print('crop_size_s: ',crop_size_s)
        print('crop_size_t: ',crop_size_t)
        dist_s = crop_size_s
        dist_t = crop_size_t

        
        sdist_mm = np.sqrt(np.sum(np.square(reconstruct_shape - center_s),axis=1)) 
        indxSel = np.where(sdist_mm <= dist_s)[0]
        reconstruct_shape_crop = reconstruct_shape[indxSel] 

        sdist_mmg = np.sqrt(np.sum(np.square(gt_shape - center_t),axis=1)) 
        indxSelg = np.where(sdist_mmg <= dist_t)[0]
        reconstruct_shape_cropg = gt_shape[indxSelg] 

        reconFaceCrop = []
        for (a, b, c) in mesh_s.faces:
                if (a in indxSel) and (b in indxSel) and (c in indxSel):
                        reconFaceCrop.append([np.where(indxSel == a)[0][0], np.where(indxSel == b)[0][0], np.where(indxSel == c)[0][0]])
        reconFaceCropg = []
        for (a, b, c) in mesh_t.faces:
                if (a in indxSelg) and (b in indxSelg) and (c in indxSelg):
                        reconFaceCropg.append([np.where(indxSelg == a)[0][0], np.where(indxSelg == b)[0][0], np.where(indxSelg == c)[0][0]])

        # find the faces(triangles from the mesh)
        # needed for the reconstruction dist. calculation
        f_s = np.array(reconFaceCrop) 
        f_t = np.array(reconFaceCropg) 
        #f_t = mesh_t.faces

        # get outer-interoccular distance, oic_dist to normalize 
        # before changing the landmark references
        oic_dist  = np.sqrt(np.sum(np.square(landmark_t[0] - landmark_t[3])))
        #print('dist: ',oic_dist)
        # ignore landmarks that were -1 in the target
        landmark_s  = np.delete(landmark_s, selInds, axis=0)
        landmark_t  = np.delete(landmark_t, selInds, axis=0)
        NME = 100*np.mean(np.sqrt(np.sum(np.square(landmark_s-landmark_t),1)))/oic_dist
        errorS2T,errorT2S,RR,tt,ss = reconstruct_distance(reconstruct_shape_crop,reconstruct_shape_cropg,f_s,f_t,landmark_s,landmark_t)
        print('error:',errorS2T.shape)
        print('max: ',max(errorS2T))
        print('crop size: ',reconstruct_shape_crop.shape)
        reconstruct_file = open('/media/weepies/Seagate Backup Plus Drive/3DMM/3dp_chosse/3dpixel/view_data/'+test_mesh+'rec.obj','w')
        gt_file = open('/media/weepies/Seagate Backup Plus Drive/3DMM/3dp_chosse/3dpixel/view_data/'+test_mesh+'gt.obj','w')
        l_r = list(reconstruct_shape_crop)
        print('l_r: ',len(reconstruct_shape_crop))
        l_g = list(reconstruct_shape_cropg)
        l_r_f = list(f_s)
        print('f size: ',len(l_r_f))
        l_g_f = list(f_t)
        for i in range(len(reconstruct_shape_crop)):
                reconstruct_file.write('v '+str(reconstruct_shape_crop[i][0])+' '+str(reconstruct_shape_crop[i][1])+' '+str(reconstruct_shape_crop[i][2])+' '
                        +'0' + ' ' + '0' + ' ' + str(int(errorS2T[i]/max(errorS2T)*255))+'\n')

        for i in range(len(f_s)):
                reconstruct_file.write('f '+str(f_s[i][0]+1)+' '+str(f_s[i][1]+1)+' '+str(f_s[i][2]+1)+'\n')
        reconstruct_file.close()

        for i in range(len(reconstruct_shape_cropg)):
                gt_file.write('v '+str(reconstruct_shape_cropg[i][0])+' '+str(reconstruct_shape_cropg[i][1])+' '+str(reconstruct_shape_cropg[i][2])+'\n')

        for i in range(len(f_t)):
                gt_file.write('f '+str(f_t[i][0]+1)+' '+str(f_t[i][1]+1)+' '+str(f_t[i][2]+1)+'\n')
        gt_file.close()
        print('write down')

        absolute_rmseS2T = 100*np.sqrt(np.mean(errorS2T**2))/(oic_dist*2)
        absolute_rmseT2S= 100*np.sqrt(np.mean(errorT2S**2))/(oic_dist*2)

	# take the average of the assymetrical point to mesh dists.
        absolute_armse = (absolute_rmseS2T + absolute_rmseT2S)
        return  absolute_armse,NME


###################################################################################################
# main function, needs reconstruction mesh, ground truth mesh and corresponding landmarks to run.
# we do icp alignment with scale to align the ground truth mesh to reconstruction mesh,
# then compute ARMSE of point to face distance. Note that we divide the RMSE with scaling factor ss because 
# errors need to match the scale of ground truth mesh instead of reconstruction mesh.
###################################################################################################
# Modified: Takes in a list of test meshes
##############################################
def eval_meshes(test_list, res_dir, ref_dir):
    n = 0
    ARMSE = [] 
    NME = []
    arm_0 = []
    arm_1 = []
    arm_2 = []
    nme_0 = []
    nme_1 = []
    nme_2 = []
    # (for loop for multiple subjects)
    txtfile = open('/media/weepies/Seagate Backup Plus Drive/3DMM/3dp_chosse/3dpixel/test/result_finetune.txt','w')
    print("Absolute ARMSE scores:")
    for test_mesh in test_list:
        print('step: ',n+1)
        n += 1
        if n<=100:
                pos = test_mesh[-1]
                print('pos: ',pos)
                absolute_armse,ab_NME = eval_mesh(test_mesh, res_dir, ref_dir)
                print("mesh"+" " +test_mesh + " ARMSE:", absolute_armse)
                print("mesh"+" " +test_mesh + " NME:", ab_NME)
                if True:
                        txtfile.write('file: '+test_mesh+'\n')
                        txtfile.write('ARMSE: '+str(absolute_armse) + '\n')
                        txtfile.write('NME: '+str(ab_NME) + '\n')
                        txtfile.write('\n')

                
                if pos == str(0):
                        arm_0.append(absolute_armse)
                if pos == str(1):
                        arm_1.append(absolute_armse)
                if pos == str(2):
                        arm_2.append(absolute_armse)
                if pos == str(0):
                        nme_0.append(ab_NME)
                if pos == str(1):
                        nme_1.append(ab_NME)
                if pos == str(2):
                        nme_2.append(ab_NME)
                ARMSE.append(absolute_armse)
                NME.append(ab_NME)
    # compute mean RMSE across subjects (final score as reported in the paper)
    mean_ARMSE = np.mean(ARMSE)
    mean_NME = np.mean(NME)
    mean_a0 = np.mean(arm_0)
    mean_a1 = np.mean(arm_1)
    mean_a2 = np.mean(arm_2)
    mean_n0 = np.mean(nme_0)
    mean_n1 = np.mean(nme_1)
    mean_n2 = np.mean(nme_2)
    txtfile.write('armse_0: '+str(mean_a0)+'\n')
    txtfile.write('armse_1: '+str(mean_a1)+'\n')
    txtfile.write('armse_2: '+str(mean_a2)+'\n')
    txtfile.write('nme_0: '+str(mean_n0)+'\n')
    txtfile.write('nme_0: '+str(mean_n1)+'\n')
    txtfile.write('nme_0: '+str(mean_n2)+'\n')
    txtfile.write('mean_ARMSE: '+str(mean_ARMSE)+'\n')
    txtfile.write('mean NME: '+str(mean_NME)+'\n')
    txtfile.write('std_ARMSE: '+str(np.std(ARMSE))+'\n')
    txtfile.close()
    print("\nSummary:") 
    print('mean_ARMSE:',mean_ARMSE)
    print('mean NME: ',mean_NME)
    print('std_ARMSE:',np.std(ARMSE))
    return mean_ARMSE


##########################################################
# Main: load and evaluate scripts
##########################################################

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pred-dir", required=True,
	help="path to predicted/reconstructed directory with the Mesh and Vertexlandmark files")
ap.add_argument("-g", "--gt-dir", required=True,
	help="path to ground truth directory with the Mesh and Vertexlandmark files")
ap.add_argument("-t", "--test-file", required=True,
	help="path to test file with the mesh subject numbers to test")
args = vars(ap.parse_args())

# parse the directories for the reference, reconstructed and test_file containing 
# meshes that need to be tested.
submit_dir = args["pred_dir"]
ref_dir = args["gt_dir"]
test_file = args["test_file"]

print("3DFAW-Video evaluation program version 12.9")
print('Pred. Dir:',submit_dir)
print('Ground Truth Dir:', ref_dir)

if not os.path.isdir(submit_dir) or not os.path.isdir(ref_dir) :
    print("Either %s or %s doesn't exist" % (submit_dir,ref_dir))

else:
    #load the test meshes
    tests_meshes = np.loadtxt(test_file, dtype='str')
    #find mean ARMSE for eval
    armse_score = eval_meshes(tests_meshes, submit_dir, ref_dir)
    print('score: ',armse_score)

print("done!")
           
    



