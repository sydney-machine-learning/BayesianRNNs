from os import path
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import copy
import argparse
import time
import datetime
import multiprocessing
import gc
import matplotlib as mpl
import pandas as pd
from model import Model
from parallelTempering import ParallelTempering
import shutil 

import os 


mpl.use('agg')
weightdecay = 0.01
#Initialise and parse inputs
parser=argparse.ArgumentParser(description='PTBayeslands modelling')
parser.add_argument('-s','--samples', help='Number of samples', default=10000, dest="samples",type=int)
parser.add_argument('-r','--replicas', help='Number of chains/replicas, best to have one per availble core/cpu', default=8,dest="num_chains",type=int)
parser.add_argument('-t','--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ', default=2,dest="mt_val",type=int)
parser.add_argument('-swap','--swap', help='Swap Ratio', dest="swap_ratio",default=0.001,type=float)
parser.add_argument('-b','--burn', help='How many samples to discard before determing posteriors', dest="burn_in",default = 0.5,type=float)
parser.add_argument('-pt','--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",default=0.5,type=float)
parser.add_argument('-step','--step', help='Step size for proposals (0.02, 0.05, 0.1 etc)', dest="step_size",default=0.025,type=float)
parser.add_argument('-lr','--learn', help='learn rate for langevin gradient', dest="learn_rate",default=0.01,type=float)
parser.add_argument('-m','--model', help='1 to select RNN, 2 to select LSTM', dest = "net", default = 2, type= int)
parser.add_argument('-o','--optim', help='1 to select SGD, 2 to select Adam', dest = 'optimizer', default = 1, type = int)
args = parser.parse_args()


def histogram_trace(pos_points, fname): # this is function to plot (not part of class)

    size = 15

    plt.tick_params(labelsize=size)
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.grid(alpha=0.75)
    
    plt.hist(pos_points,  bins = 20, color='#0504aa', alpha=0.7)
    plt.title("Posterior distribution ", fontsize = size)
    plt.xlabel(' Parameter value  ', fontsize = size)
    plt.ylabel(' Frequency ', fontsize = size)
    plt.tight_layout()
    plt.savefig(fname + '_posterior.png')
    plt.clf()


    plt.tick_params(labelsize=size)
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.grid(alpha=0.75)
    plt.plot(pos_points)

    plt.title("Parameter trace plot", fontsize = size)
    plt.xlabel(' Number of Samples  ', fontsize = size)
    plt.ylabel(' Parameter value ', fontsize = size)
    plt.tight_layout()
    plt.savefig(fname  + '_trace.png')
    plt.clf()

def plot_figure(lista, title,path):

    list_points =  lista
    size = 20
    fname = path+'/trace_plots'
    width = 9

    font = 12
    fig = plt.figure(figsize=(10,12))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.set_title(' Posterior distribution', fontsize=  font+2)#, y=1.02)

    ax1 = fig.add_subplot(211)

    n, rainbins, patches = ax1.hist(list_points,  bins = 20,  alpha=0.5, facecolor='sandybrown', density=False)


    color = ['blue','red', 'pink', 'green', 'purple', 'cyan', 'orange','olive', 'brown', 'black']

    ax1.grid(True)
    ax1.set_ylabel('Frequency',size= font+1)
    ax1.set_xlabel('Parameter values', size= font+1)

    ax2 = fig.add_subplot(212)

    list_points = np.asarray(np.split(list_points,  args.num_chains ))

    ax2.set_facecolor('#f2f2f3')
    ax2.plot( list_points.T , label=None)
    ax2.set_title(r'Trace plot',size= font+2)
    ax2.set_xlabel('Samples',size= font+1)
    ax2.set_ylabel('Parameter values', size= font+1)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    if not os.path.exists(fname):
        os.makedirs(fname)

    plt.savefig(fname + '/' + title  + '_pos_.png', bbox_inches='tight', dpi=300, transparent=False)
    plt.close()

def main():
    n_steps_in, n_steps_out = 5,10
    net = 'RNN' if args.net == 1 else 'LSTM' 
    optimizer = 'SGD' if args.optimizer == 1 else 'Adam'

    print(f"Network is {net} and optimizer is {optimizer}")
    
    
    for j in [1, 2, 3, 5, 6] :
        print(j, ' is problem','\n\n\n') 
        problem=j
        folder = ".."
        if problem ==1:
            TrainData = pd.read_csv("timeseries_data/Lazer/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("timeseries_data/Lazer/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Lazer"
        if problem ==2:
            TrainData = pd.read_csv("timeseries_data/Sunspot/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("timeseries_data/Sunspot/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Sunspot"
        if problem ==3:
            TrainData = pd.read_csv("timeseries_data/Mackey/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("timeseries_data/Mackey/test1.csv",index_col = 0)
            TestData = TestData.values
            name="Mackey"
        if problem ==4:
            TrainData = pd.read_csv("timeseries_data/Lorenz/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("timeseries_data/Lorenz/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Lorenz"
        if problem ==5:
            TrainData = pd.read_csv("timeseries_data/Rossler/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("timeseries_data/Rossler/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Rossler"
        if problem ==6:
            TrainData = pd.read_csv("timeseries_data/Henon/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("timeseries_data/Henon/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Henon"
        if problem ==7:
            TrainData = pd.read_csv("timeseries_data/ACFinance/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("timeseries_data/ACFinance/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "ACFinance"

        train_x = np.array(TrainData[:,0:n_steps_in])
        train_y = np.array(TrainData[:,n_steps_in : n_steps_in+n_steps_out ])
        test_x = np.array(TestData[:,0:n_steps_in])
        test_y = np.array(TestData[:,n_steps_in : n_steps_in+n_steps_out])

        train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],1)
        train_y = train_y.reshape(train_y.shape[0],train_y.shape[1],1)
        test_x = test_x.reshape(test_x.shape[0],test_x.shape[1],1)
        test_y = test_y.reshape(test_y.shape[0],test_y.shape[1],1)
        print("shapes of train x and y",train_x.shape,train_y.shape)
        # shapes of train x and y (585, 5) (585, 10)

        ###############################
        #THESE ARE THE HYPERPARAMETERS#
        ###############################


        
        Hidden = 10 #originally it was 5; but in the paper to compare it is 10




        topology = [n_steps_in, Hidden,n_steps_out]
        NumSample = args.samples
        netw = topology
        maxtemp = args.mt_val 
        swap_ratio = args.swap_ratio
        num_chains =  args.num_chains


        swap_interval = 5
        
        #int(swap_ratio * NumSample/num_chains)   #how ofen you swap neighbours. note if swap is more than Num_samples, its off
        burn_in = args.burn_in
        learn_rate = args.learn_rate  # in case langevin gradients are used. Can select other values, we found small value is ok.
        langevn = ""
        #if j%2 == 1:
        use_langevin_gradients = True  # False leaves it as Random-walk proposals. Note that Langevin gradients will take a bit more time computationally
        #langevn = "T"
        #else:
        #   use_langevin_gradients = False
        #   langevn = "F"
        #   pass # we dont want to execute this.
        print(f'langevin is {use_langevin_gradients}')
        problemfolder = os.getcwd()+'/Result-all_'+net+f'_{optimizer}/'  #'/home/rohit/Desktop/PT/Res_LG-Lprob/'  # change this to your directory for results output - produces large datasets
        problemfolder_db = 'Result-summ_'+net+ f'_{optimizer}/'  # save main results
        filename = ""
        run_nb = 0
        while os.path.exists( problemfolder+name+langevn+'_%s' % (run_nb)):
            run_nb += 1
        if not os.path.exists( problemfolder+name+langevn+'_%s' % (run_nb)):
            os.makedirs(  problemfolder+name+langevn+'_%s' % (run_nb))
            pathx = (problemfolder+ name+langevn+'_%s' % (run_nb))
        filename = ""
        run_nb = 0
        while os.path.exists( problemfolder_db+name+langevn+'_%s' % (run_nb)):
            run_nb += 1
        if not os.path.exists( problemfolder_db+name+langevn+'_%s' % (run_nb)):
            os.makedirs(  problemfolder_db+name+langevn+'_%s' % (run_nb))
            path_db = (problemfolder_db+ name+langevn+'_%s' % (run_nb))
        # resultingfile = open( path+'/master_result_file.txt','a+')
        # resultingfile_db = open( path_db+'/master_result_file.txt','a+')
        timer = time.time()

        
        langevin_prob = 0.5


        pt = ParallelTempering( use_langevin_gradients,  learn_rate,  train_x,train_y,test_x,test_y, topology, num_chains, maxtemp, NumSample, 
                                swap_interval, langevin_prob, pathx,rnn_net = net, optimizer= optimizer)
        directories = [  pathx+'/predictions/', pathx+'/posterior', pathx+'/results', #path+'/surrogate', 
                            # path+'/surrogate/learnsurrogate_data', 
                            pathx+'/posterior/pos_w',  pathx+'/posterior/pos_likelihood',
                            pathx+'/posterior/accept_list'  ] #path+'/posterior/surg_likelihood',
        for d in directories:
            pt.make_directory((filename)+ d)
        pt.initialize_chains(burn_in)
        # pos_w, fx_train, fx_test,  rmse_train, rmse_test, acc_train, acc_test, likelihood_rep, swap_perc, accept_vec, accept = pt.run_chains()
        pos_w, fx_train, fx_test, rmse_train, rmse_test, indiv_rmse_train, indiv_rmse_test, likelihood_rep, swap_perc, accept_vec, accept = pt.run_chains()
        
        # print(indiv_rmse_test.shape)
        # print(indiv_rmse_train.shape)
    
        timer2 = time.time()
        list_end = accept_vec.shape[1]
        # o = [pos_w, fx_train, fx_test,  rmse_train, rmse_test, acc_train, acc_test,   likelihood_rep , swap_perc,    accept_vec, accept]
        # print("shapes")
        # for i,var in enumerate(['pos_w', 'fx_train', 'fx_test',  'rmse_train', 'rmse_test', 'acc_train', 'acc_test',   'likelihood_rep' , 'swap_perc',    'accept_vec', 'accept']):
        #     print(f'{var}: {o[i].shape}')
        # ''' to plot ax plots '''
        # break

        print(pos_w.shape, ' is shape of pos w \nInitiating Plotting Sequence')
        plot_fname = pathx
        # pt.make_directory(plot_fname + '/pos_plots')
        for s in range(20): # change this if you want to see all pos plots
            plot_figure(pos_w[s,:], 'pos_distri_'+str(s) , pathx)
        print(' images placed in folder with name: ',pathx)

        if not os.path.exists(pathx+ '/trace_plots_better'):
            os.makedirs(pathx+ '/trace_plots_better')
        #for i in range(pos_w.shape[0]):
        for i in range(20):
            histogram_trace(pos_w[i,:], pathx+ '/trace_plots_better/'+ str(i))

        #Plot to compare the rmse accross all the chains


        print(' begin visulisation of results')


        font = 12
        fig = plt.figure(figsize=(10,12))
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        # ax.set_title('RMSE Trace Plots', fontsize=  font+2)#, y=1.02)

        ax1 = fig.add_subplot(211)

        list_points = np.squeeze(np.asarray(np.split(rmse_train,args.num_chains))).T
        ax1.plot(list_points)
        ax1.legend([f'Chain {i}' for i in range(args.num_chains)])
        #this rmse train is something which is like {rmse trains of chain1, rmse trains of chain2 , ... }

    
        color = ['blue','red', 'pink', 'green', 'purple', 'cyan', 'orange','olive', 'brown', 'black']

        ax1.grid(True)
        ax1.set_ylabel('RMSE',size= font+1)
        ax1.set_xlabel('Samples', size= font+1)
        ax1.set_title(r'RMSE Train trace plot across chains', size = font +2)
        ax2 = fig.add_subplot(212)

        list_points = np.squeeze(np.asarray(np.split(rmse_test,args.num_chains))).T

        ax2.set_facecolor('#f2f2f3')
        ax2.plot( list_points , label=None)
        ax2.grid(True)
        ax2.set_title(r'RMSE Test trace plot across chains',size= font+2)
        ax2.set_xlabel('Samples',size= font+1)
        ax2.set_ylabel('RMSE', size= font+1)
        ax2.legend([f'Chain {i}' for i in range(args.num_chains)])

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        
        plt.savefig(pathx + '/rmse_comparison.png', bbox_inches='tight', dpi=300, transparent=False)
        plt.close()
        accept_ratio = accept_vec[:,  list_end-1:list_end]/list_end
        accept_per = np.mean(accept_ratio) * 100
        print(accept_per, ' accept_percentage')
        # timer2 = time.time()
        timetotal = (timer2 - timer) /60
        print ((timetotal), 'min taken for training')
        #PLOTS
        '''acc_tr = np.mean(acc_train [:])
        acctr_std = np.std(acc_train[:])
        acctr_max = np.amax(acc_train[:])
        acc_tes = np.mean(acc_test[:])
        acctest_std = np.std(acc_test[:])
        acctes_max = np.amax(acc_test[:])'''
        step_wise_rmse_train_mean = np.mean(indiv_rmse_train, axis= 0, keepdims= True)
        # print(step_wise_rmse_train_mean.shape)
        step_wise_rmse_train_std = np.std(indiv_rmse_train, axis = 0, keepdims= True)
        step_wise_rmse_train_best = np.amin(indiv_rmse_train, axis = 0, keepdims= True)
        rmse_tr = np.mean(rmse_train[:])   
        rmsetr_std = np.std(rmse_train[:])
        rmsetr_max = np.amin(rmse_train[:])
        step_wise_rmse_test_mean = np.mean(indiv_rmse_test, axis = 0, keepdims= True)
        step_wise_rmse_test_std = np.std(indiv_rmse_test, axis = 0, keepdims= True)
        step_wise_rmse_test_best = np.amin(indiv_rmse_test, axis = 0, keepdims= True)
        rmse_tes = np.mean(rmse_test[:])
        rmsetest_std = np.std(rmse_test[:])
        rmsetes_max = np.amin(rmse_test[:])
        outres = open(pathx+'/results/result.txt', "a+")
        outres_step = open(pathx + '/results/result_stepwise.txt', 'a+')
        # outres_db = open(path_db+'/result.txt', "a+")
        # outres_step_db = open(path + '/result_stepwise.txt', 'a+')
        resultingfile = open(problemfolder+'/master_result_file.txt','a+')
        resultingfile_step = open(problemfolder + '/master_step_result_file.txt', 'a+')
        #in this file i will store the table after confirming my doubt from ayush
        resultingfile_db = open( problemfolder_db+'/master_result_file.txt','a+')
        xv = name+langevn+'_'+ str(run_nb)
        step_wise_results = np.concatenate(
                                            (
                                                step_wise_rmse_train_mean,
                                                step_wise_rmse_train_std,
                                                step_wise_rmse_train_best,
                                                step_wise_rmse_test_mean,
                                                step_wise_rmse_test_std,
                                                step_wise_rmse_test_best
                                            ),
                                            axis= 0
                                        )
        # print(step_wise_results.shape)
        columns = [f'Step {i}' for i in range(1,11)]
        indices = ['train_mean', 'train_std','train_best', 'test_mean', 'test_std', 'test_best']
        df = pd.DataFrame(step_wise_results, index = indices, columns= columns)
        df['overall'] = [rmse_tr, rmsetr_std, rmsetr_max, rmse_tes, rmsetest_std, rmsetes_max]
        columns = ['overall'] + columns
        df = df.reindex(columns= columns)
        df = df.transpose()

        df.to_csv(path_or_buf= pathx + '/results/result_stepwise.csv', header = f'{net} {optimizer}')

        np.savetxt(outres_step, step_wise_results, fmt = '%1.4f', newline= ' \n')
        # np.savetxt(outres_step_db, step_wise_results, fmt= '%1.4f', newline = ' \n')
        allres =  np.asarray([ problem, NumSample, maxtemp, swap_interval, langevin_prob, learn_rate,  
                                rmse_tr, rmsetr_std, rmsetr_max, rmse_tes, rmsetest_std, rmsetes_max, 
                                swap_perc, accept_per, timetotal])
        # np.savetxt(outres_db,  allres   , fmt='%1.4f', newline=' '  )
        np.savetxt(resultingfile_db,   allres   , fmt='%1.4f',  newline=' ' )
        np.savetxt(resultingfile_db, [xv]   ,  fmt="%s", newline=' \n' )
        np.savetxt(outres,  allres   , fmt='%1.4f', newline=' '  )
        np.savetxt(resultingfile,   allres   , fmt='%1.4f',  newline=' ' )
        np.savetxt(resultingfile, [xv]   ,  fmt="%s", newline=' \n' )
        x = np.linspace(0, rmse_train.shape[0] , num=rmse_train.shape[0])
        # outres.close()
        # outres_db.close()
        # resultingfile.close()
        # resultingfile_db.close()


        #Creating Csv files as in dataframes



        '''plt.plot(x, rmse_train, '.',   label='Test')
        plt.plot(x, rmse_test,  '.', label='Train')
        plt.legend(loc='upper right')
        plt.xlabel('Samples', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.savefig(path+'/rmse_samples.png')
        plt.clf()	'''
        plt.plot(  rmse_train, '.',  label='Test')
        plt.plot(  rmse_test,  '.',  label='Train')
        plt.legend(loc='upper right')
        plt.xlabel('Samples', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.savefig(path_db+'/rmse_samples.pdf')
        plt.clf()
        plt.plot( rmse_train, '.',   label='Test')
        plt.plot( rmse_test, '.',   label='Train')
        plt.legend(loc='upper right')
        plt.xlabel('Samples', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.savefig(pathx+'/rmse_samples.pdf')
        plt.clf()
        likelihood = likelihood_rep[:,0] # just plot proposed likelihood
        likelihood = np.asarray(np.split(likelihood, num_chains))
    # Plots
        plt.plot(likelihood.T)
        plt.xlabel('Samples', fontsize=12)
        plt.ylabel(' Log-Likelihood', fontsize=12)
        plt.savefig(pathx+'/likelihood.png')
        plt.clf()
        plt.plot(likelihood.T)
        plt.xlabel('Samples', fontsize=12)
        plt.ylabel(' Log-Likelihood', fontsize=12)
        plt.savefig(path_db+'/likelihood.png')
        plt.clf()
        plt.plot(accept_vec.T )
        plt.xlabel('Samples', fontsize=12)
        plt.ylabel(' Number accepted proposals', fontsize=12)
        plt.savefig(pathx+'/accept.png')
        plt.clf()
        plt.plot(accept_vec.T )
        plt.xlabel('Samples', fontsize=12)
        plt.ylabel(' Number accepted proposals', fontsize=12)
        plt.savefig(path_db+'/accept.png')
        plt.clf()
        #mpl_fig = plt.figure()
        #ax = mpl_fig.add_subplot(111)
        # ax.boxplot(pos_w)
        # ax.set_xlabel('[W1] [B1] [W2] [B2]')
        # ax.set_ylabel('Posterior')
        # plt.legend(loc='upper right')
        # plt.title("Boxplot of Posterior W (weights and biases)")
        # plt.savefig(path+'/w_pos.png')
        # plt.savefig(path+'/w_pos.svg', format='svg', dpi=600)
        # plt.clf()
        #dir()

        #plotting step_wise_rmse_values

        print(' done with visulisation of results')




        gc.collect()
        outres.close()
        outres_step.close()
        # outres_step_db.close()
        resultingfile.close()
        resultingfile_db.close()
        # outres_db.close()


        dir_name = pathx  + '/posterior'
        print(dir_name)
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        
        dir_name = pathx  + '/predictions'
        print(dir_name)
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)

        #fname_remove = path_db  +'/pos_param.txt'

        #if os.path.exists(fname_remove):  # comment if you wish to keep pos file
        #    os.remove(fname_remove)

    

if __name__ == "__main__": main()

