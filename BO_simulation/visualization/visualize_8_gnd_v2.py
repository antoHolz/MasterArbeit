import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def vis(argv):
    R=.3
    NUM_WP=48*6
    NUM_R=48*6
    NUM_CSTEPS=NUM_R+NUM_WP
    

    csv_name=argv[1]#'pid_BO_R_500_alpha_UCB_c1_0.8_c2_4_start_PID_0.01_rounds_11-11.28.2022_21.29'
    data_x=pd.read_csv('../files/csvs/'+csv_name+'/x0.csv')
    data_y=pd.read_csv('../files/csvs/'+csv_name+'/y0.csv')

    chges=np.array((20,40, int((data_x.shape[0]+NUM_R)/NUM_CSTEPS)))
    heights=np.array((0.05, 0.1,0.05))
    start=0

    #### BO #########################################
    # XY-trajectory
    for g in chges:
        # ground truth  
        k=np.arange(0,NUM_WP,1)
        x_true=R * np.cos((k / NUM_WP) * (2 * np.pi) + np.pi / 2)
        y_true=R * np.sin(2*((k / NUM_WP) * (2 * np.pi) + np.pi / 2))/2
        plt.plot(x_true,y_true,label='true_value',linestyle='dashed')

        for i in range(g-start):
            if(i==(g-start-1)):
                plt.plot(data_x.iloc[:,1][NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i)+NUM_WP], 
                        data_y.iloc[:,1][NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i)+NUM_WP], 
                        label=('round'+str(i) + '(best)'), color='purple', linewidth=2)
            else:
                plt.plot(data_x.iloc[:,1][NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i)+NUM_WP], 
                         data_y.iloc[:,1][NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i)+NUM_WP], 
                        label=('round'+str(i)), 
                        color=([1-i/float(g-start),i/(float(g-start)),0,0.8-i/(float(g-start)*3)]))
        start=g
    
        #plt.legend()
        plt.savefig('../files/csvs/'+csv_name+'/trajectories_until'+str(g)+'.png')
        plt.show()

    # Z-error
    data_z=pd.read_csv('../files/csvs/'+csv_name+'/z0.csv')
    true_z=heights[0]*np.ones(data_z.iloc[:,1].shape)
    for j in range(chges.size-1):
        true_z[chges[j]*NUM_CSTEPS:chges[j+1]*NUM_CSTEPS]=heights[j+1]*np.ones(true_z[chges[j]*NUM_CSTEPS:chges[j+1]*NUM_CSTEPS].shape)
    error=np.abs(true_z-data_z.iloc[:,1])
    plt.plot(data_z.iloc[:,0], error)
    plt.savefig('../files/csvs/'+csv_name+'/z_trajectory_error.png')
    plt.show()

    #### average cost per parameter combination ######
    data_performance=pd.read_csv('../files/csvs/'+csv_name+'/candidate_performance.csv').dropna(axis=0)
    (-data_performance['cost']).plot()
    plt.savefig('../files/csvs/'+csv_name+'/cost.png')
    plt.show()

    print(data_performance)

if __name__ == "__main__":
    vis(sys.argv)

