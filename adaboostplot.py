import matplotlib.pyplot as plt

graph_titles = ["SG:62; $N_s$=12","SG:139; $N_s$=5","SG:189; $N_s$=9","SG:221; $N_s$=2","SG:221; $N_s$=5","SG:225; $N_s$=2"]

file_names = ["grp_62-site_12","grp_139-site_5","grp_189-site_9","grp_221-site_2","grp_221-site_5","grp_225-site_2"]

fig_maxv,ax_maxv = plt.subplots(2,3, figsize=(6.5,3.5), sharex=True,sharey=True)
dim_x = [i for i in range(1,21)]
fig_maxv.subplots_adjust(hspace=0.1,wspace=0.1)
ax_maxv[0,0].axvline(x = 6, color = 'grey', linestyle = '--')
ax_maxv[1,0].axvline(x = 4, color = 'grey', linestyle = '--')
ax_maxv[0,1].axvline(x = 4, color = 'grey', linestyle = '--')
ax_maxv[1,1].axvline(x = 6, color = 'grey', linestyle = '--')
ax_maxv[0,2].axvline(x = 3, color = 'grey', linestyle = '--')
ax_maxv[1,2].axvline(x = 3, color = 'grey', linestyle = '--')


for i in range(0,6):
    row=i%2
    col=i//2

    with open(file_names[i]+"/adaboost/adaboost_2500_2.dat") as f:
        data = [line.strip().split() for line in f]
        raw_maxv = float(data[1][2])
        maxv = [float(num) for num in data[2]]
    
    ax_maxv[row,col].axhline(y = raw_maxv, color = 'red', linestyle = '--')        
    ax_maxv[row,col].plot(dim_x,maxv, marker = '.',color="c")
    ax_maxv[row,col].set_title(graph_titles[i],x=0.5,y=0.01,fontsize=8)
    


ax_maxv[0,0].set_xticks([1,5,10,15,20],[1,5,10,15,20])    
ax_maxv[0,0].tick_params(axis='both', which='major', labelsize=8)
ax_maxv[1,0].tick_params(axis='both', which='major', labelsize=8)
ax_maxv[1,1].tick_params(axis='both', which='major', labelsize=8)
ax_maxv[1,2].tick_params(axis='both', which='major', labelsize=8)
fig_maxv.tight_layout()
fig_maxv.savefig("adaboost_max_score.pdf", format='pdf')

with open("grp_221-site_2/adaboost/adaboost_2500_2.dat") as f:
    data = [line.strip().split() for line in f]
    raw_time1 = float(data[4][2])
    time1 = [float(num) for num in data[5]]
with open("grp_221-site_5/adaboost/adaboost_2500_2.dat") as f:
    data = [line.strip().split() for line in f]
    raw_time2 = float(data[4][2])
    time2 = [float(num) for num in data[5]]
    

fig_time = plt.figure(figsize=[4,3.5])
plt.plot(dim_x,time1, marker = 'o', color="blue", label=graph_titles[3])
plt.plot(dim_x,time2, marker = 's', color="orange", label=graph_titles[4])
plt.axhline(y = raw_time1, linestyle = '--', color="c", label=graph_titles[3]+" raw data")
plt.axhline(y = raw_time2, linestyle = ':', color="red", label=graph_titles[4]+" raw data")
plt.xticks([1,5,10,15,20],[1,5,10,15,20],fontsize=8)
plt.tick_params(axis='both', which='major', labelsize=8)
plt.xlabel("No. of dimensions",fontsize=8)
plt.ylabel("Time (s)",fontsize=8)
plt.legend(fontsize=6, loc="lower right")
plt.tight_layout()
fig_time.savefig("adaboost_time.pdf", format='pdf')
#plt.show()
    
    