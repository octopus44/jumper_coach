from data_utils import get_dir_list,save_pickle,load_pickle,plot_skeleton_3d,rot_mat_to_euler, load_dict_from_csv
from dtw import *
import matplotlib.pyplot as plt
import numpy as np
import os



def kNN(dist_mat,label_mat,k=5):

    k_sorted_inds = np.argsort(dist_mat,axis=1)[:,0:k]
    k_dist_mat = np.diagonal(dist_mat[:,k_sorted_inds],axis1=0,axis2=1).T
    k_label_mat = np.diagonal(label_mat[:,k_sorted_inds],axis1=0,axis2=1).T
    # compute weights
    w = 1.0/k_dist_mat
    sum_w = np.sum(1.0/k_dist_mat,axis=1)
    y_pred = np.round(np.sum(k_label_mat*w,axis=1)/sum_w)
    return y_pred, k_dist_mat,k_sorted_inds


ann_dict = load_dict_from_csv(os.path.join("data", "annotations.csv"))
joint_dict = load_pickle(os.path.join("data", "joint_trajectories_norm.pkl"))
dtw_alignments = load_pickle(os.path.join("data", "dtw_alignments.pkl"))

num_vids = len(joint_dict.keys())

# Leave one out cross validation

inds = np.arange(num_vids)
np.random.shuffle(inds)

n_folds = num_vids
folds=np.array_split(inds,n_folds)
sum_error = 0
errors = np.zeros((n_folds))
k_dists = []
k_inds= []
k_alignments = []
fold_train_inds = []


y = []

row_inds = list(dtw_alignments.keys())
col_inds = []
dist_mat = np.zeros((len(row_inds),len(row_inds)-1))
label_mat = np.zeros((len(row_inds),len(row_inds)-1))

for i,name1 in enumerate(row_inds):
    # get ground truth
    y.append(int(ann_dict["bar_outcome"][np.argwhere(np.array(ann_dict["name"]) == name1)[0, 0]]))

    # populate distance matrix
    col_inds.append(list(dtw_alignments[name1].keys()))
    for j,name2 in enumerate(col_inds[-1]):
        dist_mat[i,j] = dtw_alignments[name1][name2]["dist"]
        label_mat[i,j] = int(ann_dict["bar_outcome"][np.argwhere(np.array(ann_dict["name"]) == name2)[0, 0]])


# do kNN clustering
best_k = 0
best_acc= 0
for k in range(1,11):

    y_pred, k_dists,k_sorted_inds = kNN(dist_mat,label_mat,k=k)

    error =  np.sum(np.abs(np.array(y)-np.array(y_pred)))/len(y)
    print(f"kNN accuracy (k={k}): {1-error}")
    if 1-error > best_acc:
        best_k = k
        best_acc = 1-error

y_pred, k_dists,k_sorted_inds = kNN(dist_mat,label_mat,best_k)

print(f"\nBest kNN accuracy (k={best_k}): {best_acc}")



# visualize the failures

dist_sums = np.sum(k_dists,axis=1)
worst_i = np.argsort(dist_sums)[-1]

worst_vid = row_inds[worst_i]

k_vids = [col_inds[worst_i][k_sorted_inds[worst_i,j]] for j in range(best_k)]

print(f"worst video - {worst_vid}")


joints_worst = joint_dict[worst_vid]


figs = []
axes = []
for vid in k_vids:
    figs.append(plt.figure(figsize=(16,10),dpi=300))
    axes.append( figs[-1].add_subplot(1, 1, 1, projection="3d")) #
    joints_k = joint_dict[vid]
    joints_worst_warp = joints_worst[dtw_alignments[worst_vid][vid]["index1"],...]
    joints_k_warp = joints_k[dtw_alignments[worst_vid][vid]["index2"],...]

    step = joints_k_warp.shape[0]//4
    ts = np.linspace(0,joints_k_warp.shape[0]-1,5)

    for i,t in enumerate(ts):
        joints_worst_warp[int(t),:,0] += i
        joints_k_warp[int(t),:,0] += i
        plot_skeleton_3d(joints_k_warp[int(t),...],axes[-1],xrange=[-0.5,0.5+len(ts)],marker_size = 0.5,linewidth=0.5,limb_color='#1f77b490',joint_color='#d6272890')
        plot_skeleton_3d(joints_worst_warp[int(t),...],axes[-1],xrange=[-0.5,0.5+len(ts)],marker_size = 0.5,linewidth=0.5)

    axes[-1].view_init(elev=25., azim=90)
    axes[-1].xaxis.set_ticklabels([])
    axes[-1].yaxis.set_ticklabels([])
    axes[-1].zaxis.set_ticklabels([])
    figs[-1].suptitle(f"{worst_vid} vs. {vid}")
    figs[-1].subplots_adjust(wspace=0, hspace=0,top=1,bottom=0,left=0,right=1)



plt.show()



