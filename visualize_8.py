import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def vis(argv):
    NUM_WP=48*6
    NUM_R=48*2
    NUM_CSTEPS=NUM_R+NUM_WP
    ####  superimpose PID without BO  ################

    # csv_PID_name='save-flight-pid-11.25.2022_11.11.49'
    # data_x_PID=pd.read_csv('csvs/'+csv_PID_name+'/x0.csv')
    # data_y_PID=pd.read_csv('csvs/'+csv_PID_name+'/y0.csv')

    # for i in range(int(data_x_PID.shape[0]/NUM_WP+0.5)):
    #     plt.plot(data_x_PID.iloc[:,1][NUM_WP*i:NUM_WP*(1+i)], data_y_PID.iloc[:,1][NUM_WP*i:NUM_WP*(1+i)], label=('PID_round'+str(i)))

    #### BO #########################################
    
    csv_name=argv[1]#'pid_BO_R_500_alpha_UCB_c1_0.8_c2_4_start_PID_0.01_rounds_11-11.28.2022_21.29'
    data_x=pd.read_csv('../files/csvs/'+csv_name+'/x0.csv')
    data_y=pd.read_csv('../files/csvs/'+csv_name+'/y0.csv')

    for i in range(int((data_x.shape[0]+NUM_R)/NUM_CSTEPS)):
        if(i==(int((data_x.shape[0]+NUM_R)/NUM_CSTEPS)-1)):
            plt.plot(data_x.iloc[:,1][NUM_CSTEPS*i:NUM_CSTEPS*i+NUM_WP], 
                    data_y.iloc[:,1][NUM_CSTEPS*i:NUM_CSTEPS*i+NUM_WP], 
                    label=('round'+str(i) + '(best)'), color='purple', linewidth=2)
        else:
            plt.plot(data_x.iloc[:,1][NUM_CSTEPS*i:NUM_CSTEPS*i+NUM_WP], 
                    data_y.iloc[:,1][NUM_CSTEPS*i:NUM_CSTEPS*i+NUM_WP], 
                    label=('round'+str(i)), 
                    color=([1-i/float(data_x.shape[0]/NUM_WP+0.5),i/(float(data_x.shape[0]/NUM_WP+0.5)),0,0.8-i/(float(data_x.shape[0]/NUM_WP+0.5)*3)]))
     
    ####  ground truth  ##############################
    R=.3
    i=np.arange(0,NUM_WP,1)
    x_true=R * np.cos((i / NUM_WP) * (2 * np.pi) + np.pi / 2)
    y_true=R * np.sin(2*((i / NUM_WP) * (2 * np.pi) + np.pi / 2))/2
    plt.plot(x_true,y_true,label='true_value',linestyle='dashed')

    #plt.legend()
    plt.savefig('../files/csvs/'+csv_name+'/trajectories.png')
    plt.show()

    #### average cost per parameter combination ######
    data_performance=pd.read_csv('../files/csvs/'+csv_name+'/candidate_performance.csv')

    iterations=np.arange(0,data_performance.columns.size-1,1)
    performance=[-float(i) for i in data_performance.iloc[1,1:]]
    plt.plot(iterations, performance)
    plt.savefig('../files/csvs/'+csv_name+'/cost.png')
    plt.show()

    for col in data_performance.columns:
        print(col, data_performance[col][0], data_performance[col][1],data_performance[col][2] )


    with open('../files/csvs/'+csv_name+'/performance.txt', 'w') as f:
        for col in data_performance.columns:
            f.write('\n'+str(col)+')'+ ' theta: '+str(data_performance[col][0])
            +'  normalized cost: '+str(data_performance[col][1])
            +'  absolute cost: '+str(data_performance[col][2]) )




if __name__ == "__main__":
    vis(sys.argv)

