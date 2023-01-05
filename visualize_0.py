import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def vis(argv):
    ####  ground truth  ##############################
    R=.3
    NUM_WP=48*6
    i=np.arange(0,NUM_WP,1)
    x_true=R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)
    z_true=R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)+3*R 
    plt.plot(x_true,z_true,label='true_value',linestyle='dashed')

    ####  superimpose PID without BO  ################

    # csv_PID_name='save-flight-pid-11.25.2022_11.11.49'
    # data_x_PID=pd.read_csv('csvs/'+csv_PID_name+'/x0.csv')
    # data_y_PID=pd.read_csv('csvs/'+csv_PID_name+'/y0.csv')

    # for i in range(int(data_x_PID.shape[0]/NUM_WP+0.5)):
    #     plt.plot(data_x_PID.iloc[:,1][NUM_WP*i:NUM_WP*(1+i)], data_y_PID.iloc[:,1][NUM_WP*i:NUM_WP*(1+i)], label=('PID_round'+str(i)))

    #### BO #########################################
    csv_name=argv[1]#'pid_BO_R_500_alpha_UCB_c1_0.8_c2_4_start_PID_0.01_rounds_11-11.28.2022_21.29'
    data_x=pd.read_csv('../files/csvs/'+csv_name+'/x0.csv')
    data_z=pd.read_csv('../files/csvs/'+csv_name+'/z0.csv')

    for i in range(int(data_x.shape[0]/NUM_WP+0.5)):
        if(i==int(data_x.shape[0]/NUM_WP-0.5)):
            plt.plot(data_x.iloc[:,1][NUM_WP*i:NUM_WP*(1+i)], data_z.iloc[:,1][NUM_WP*i:NUM_WP*(1+i)], label=('round'+str(i) + '(best)'), color='purple', linewidth=2)
        else:
            plt.plot(data_x.iloc[:,1][NUM_WP*i:NUM_WP*(1+i)], data_z.iloc[:,1][NUM_WP*i:NUM_WP*(1+i)], label=('round'+str(i)), color=([1-i/float(data_x.shape[0]/NUM_WP+0.5),i/(float(data_x.shape[0]/NUM_WP+0.5)),0,0.8-i/(float(data_x.shape[0]/NUM_WP+0.5)*3)]))
     
            
    #### average cost per parameter combination ######
    data_performance=pd.read_csv('../files/csvs/'+csv_name+'/candidate_performance.csv')
    #mean_performance=data_performance.mean()
    #print(mean_performance)
    for col in data_performance.columns:
        print(col, data_performance[col][0], data_performance[col][1],data_performance[col][2] )

    #plt.legend()
    plt.savefig('../files/csvs/'+csv_name+'/trajectories.png')
    plt.show()

    iterations=np.arange(0,data_performance.columns.size-1,1)
    performance=[-float(i) for i in data_performance.iloc[1,1:]]
    plt.plot(iterations, performance)
    plt.savefig('../files/csvs/'+csv_name+'/cost.png')
    plt.show()

    # iterations=np.arange(0,data_performance.columns.size-2,1)
    # performance=[-float(i) for i in data_performance.iloc[2,2:]]
    # plt.plot(iterations, performance)
    # plt.show()


if __name__ == "__main__":
    vis(sys.argv)

