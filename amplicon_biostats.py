#the purpose of this code is to make graphics from sorted/analyzed data, run statistics, and to create rel depth OTU profiles by depth per O2 zone

#import required pyhon libraries
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.markers as mmarkers
import re
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import scipy as sp
from statsmodels.sandbox.stats.multicomp import multipletests
from matplotlib import cm
import seaborn as sns
import skbio.diversity
from skbio.diversity import * #pw_distances was renamed to beta_diversity and was moved to skbio.diversity
from skbio.stats.distance import mantel
import sklearn.metrics
from skbio.stats.ordination import *
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from skbio.stats.distance import anosim
from skbio.stats.composition import ancom

#my functions

def replFIwRI(s,mapfl):

    #print("s is",s)
    depth=mapfl["Depth"][mapfl.index==s].tolist()[0] #find corresponding depth using sample ID in mapping file

    #string test required to keep discrete size-fractions
    strtest=('GL3-2-7' in s) or ((s[s.rfind('C')-1]).isnumeric())

    if strtest:
        depth=str(depth)+"m P.A."
    else:
        depth=str(depth)+"m F.L."

    #print("depth is",depth)

    if "AC-GL4" in s:
        firstPart="Jul. "

    if "AC-GL3" in s:
        firstPart="Oct. "

    fullName=firstPart+depth
    #print(fullName,"\n")

    return fullName

def matrixcleanandplot(mat,contmapfl,retmat=False,watercol=True):

    #matrix clean-up for plot- replace field IDs with real IDs, sort
    plotmat=mat.copy()
    plotmat.index.rename(plotmat.index.name.replace("#",""), inplace=True)

    if watercol==True:

        trStr=""
        
        plotmat.rename(index=lambda x:replFIwRI(x,contmapfl),inplace=True)
        plotmat.rename(columns=lambda x:replFIwRI(x,contmapfl),inplace=True)
        #rename 5.0m in index and columns for sorting
        plotmat.rename(columns={"Oct. 5.0m F.L.": "Oct. 05.0m F.L.","Oct. 5.0m P.A.": "Oct. 05.0m P.A."},index={"Oct. 5.0m F.L.": "Oct. 05.0m F.L.","Oct. 5.0m P.A.": "Oct. 05.0m P.A."},inplace=True)

    else:
        trStr="trap"
        
    #sort, alphabetic, first indices, then columns
    plotmat.sort_index(ascending=True, inplace=True)
    plotmat=plotmat.reindex(sorted(plotmat.columns), axis=1)
    sns.heatmap(plotmat, annot=False,xticklabels=True,yticklabels=True,cmap="gnuplot2")
    plt.tight_layout()
    
    plt.savefig(root+"/pythoncode/"+trStr+"statsout/"+(mat.name)+".pdf")
    plt.close()
    
    if retmat:
        return plotmat
    
def matrixPCoA(mat,contmapfl,ret=False,tr=False):

    if tr==False:
        trStr=""
    else:
        trStr="trap"

    dm = skbio.DistanceMatrix(mat,mat.index) #format data, sampleIDs
    
    PCoAres = pcoa(dm)
    PCoA_prps=PCoAres.proportion_explained
    PCoA_prps.index.name="PC"
    
    PCoA_eigs=PCoAres.eigvals
    PCoA_eigs.index.name="PC"
    
    PCoA_mat=PCoAres.samples
    PCoA_mat.index.name="Sample ID"
    
    if tr==False:
    
        pltylabs=PCoA_mat.rename(index=lambda x:replFIwRI(x,contmapfl),inplace=False).index
    
    else:
    
        pltylabs=PCoA_mat.index
    
    PCoA_prps.to_csv(root+"/pythoncode/"+trStr+"statsout/"+mat.name+"_propsexplained.csv",header=["Proportion explained"],index=True)
    PCoA_prps.to_csv(root+"/pythoncode/"+trStr+"statsout/"+mat.name+"_eigenvalues.csv",header=["Eigenvalue"],index=True)
    PCoA_mat.to_csv(root+"/pythoncode/"+trStr+"statsout/"+mat.name+"_PCoAmatrix.csv",index=True)
    
    sns.heatmap(PCoA_mat,yticklabels=pltylabs,xticklabels=PCoAres.samples.columns)
    plt.tight_layout()
    plt.savefig(root+"/pythoncode/"+trStr+"statsout/"+(mat.name)+"PCoA.pdf")
    plt.close()
    
    if ret:
        return PCoAres

def permanovadm(mat,catmapfl):

    #Run PERMANOVA using categorical metadata and distance matrix

    varNameList=[]
    sampleSizeList=[]
    noGroupsList=[]
    pValList=[]


    for col in catmapfl.columns:
        if col != "DescriptionCat":
            #print(col)
            tempcol=pd.DataFrame(catmapfl[col].copy())
        
            #if there is NaN in the categorical map file column, the NaN rows need to be masked
            if catmapfl[col].isna().any():
                #print("file contains NaN")
                nanMask=catmapfl[col].isna()
                evalVals=tempcol[~nanMask]
            else:
                evalVals=tempcol
       
            #print("evalVals",evalVals,"\n")
    
            #due to the NaN masking, need to filter the distance matrix to match
            tempmat = mat[evalVals.index].loc[evalVals.index]
            dm = skbio.DistanceMatrix(tempmat,tempmat.index) #format data,sampleIDs
            resObj=(skbio.stats.distance.permanova(dm,evalVals,col))
            varNameList.append(col.replace("Cat",""))
            sampleSizeList.append(resObj.loc["sample size"])
            noGroupsList.append(resObj.loc["number of groups"])
            pValList.append(resObj.loc["p-value"])

    permanovaFr=pd.DataFrame({"variable":varNameList,"sample size":sampleSizeList,"number of groups":noGroupsList,"p-value":pValList})
    permanovaFr.set_index("variable",inplace=True)
    permanovaFr.sort_values("p-value", axis=0, ascending=True, inplace=True, kind="quicksort")

    permanovaFr.to_csv(root+"/pythoncode/statsout/"+(mat.name)+"permanovares.csv",index=True)


def creatbindict(col,bindf):

    metadataCol=col.replace("Cat","")
    metadataBinList=eval(bindf.reindex([metadataCol]).bin_separators[0])
    #print("bin separator values",metadataBinList)
    
    binListNumEval=[i for i in metadataBinList if len(str(abs(i)).replace(".",""))>5]
    
    if binListNumEval:
        metadataBinList=["{:.1e}".format(i) for i in metadataBinList]
 
    binTupleList=[]
    binTupleList.append(("0.0",'< '+str(metadataBinList[0])))

    metadataRange=np.arange(len(metadataBinList)+1)
    #print(metadataRange)
    
    for i in range(1,len(metadataRange)-1):
        binTupleList.append((str(float(i)),">= "+str(metadataBinList[i-1])+' '+" < "+str(metadataBinList[i])))

    binTupleList.append((str(float(metadataRange[-1])),">= "+str(metadataBinList[-1])))
    
    binDict=dict(binTupleList)
    #print(binDict)
    
    return binDict
    
def boxwhiskplot(grlist,nmlist,col):

    fig=plt.figure()
    plt.boxplot(grlist,labels=nmlist)
    titleName=col.replace("Cat","")
    titleName=titleName.rstrip()
    plt.title(titleName+"\n")
    plt.ylabel("pielou evenness\n")
    
    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()
    plt.savefig(root+"/pythoncode/statsout/pielou_e_"+titleName.replace("/","")+".pdf")
    plt.close()


def pc_plot(pdat, colorvar, matnm, mdbins, mdlc,pexp,pcomps = [1,2,3], name = "",trStr="",fileOut=False):
    print("colorvar",colorvar)
    
    plt.rcParams["font.family"] = "Serif" #universally changes the font without doing usetex=True, which breaks for some text formatting.
    
    c1 = "PC" + str(pcomps[0])
    c2 = "PC" + str(pcomps[1])
    
    if(len(pcomps) == 3):

        c3 = "PC" + str(pcomps[2])

    markerlist = []
    markerLegH=[]
    
    for item in pdat.index: #pdat is PCoA matrix, and pd.index=sample IDs. This iterates through the PCoA and makes a marker list corresponding to sample ID conditions
        print("item",item)
        if "GL4" in item:
            markerlist.append("s")
            markerLegH.append("late Jul.")
        elif "GL1" in item:
            markerlist.append("v")
            markerLegH.append("early Jul.")
        elif "GL3" in item:
            markerlist.append("o")
            markerLegH.append("early Oct.")
        else:
            markerlist.append("^")
            markerLegH.append("mid Aug.")
            
    cmap=sns.diverging_palette(10,240,s=60, as_cmap=True) #300,145, s=60

    fig = plt.figure()
    
    if(len(pcomps) == 3):
        ax1 = fig.add_subplot(111, projection="3d")
    else:
        ax1 = fig.add_subplot(111)
        
    cats = np.unique(colorvar) #unique cateogrical metadata as a list (colorvar is a series, taken as a single column of the metadata table)
    print("cats",cats)
    print("colorvar",colorvar)
    print("markerlist",markerlist)

    
    for i,m in enumerate(markerlist): #enumerate the marker list (which corresponds to the sample IDs in PCoA)
        print("i,m",i,m)
        print("colorvar.iloc[i] == cats",colorvar.iloc[i][0] == cats)
        print("np.where(colorvar.iloc[i] == cats)[0][0]",np.where(colorvar.iloc[i][0] == cats)[0][0])
        
        col = (np.where(colorvar.iloc[i][0] == cats)[0][0])/(len(cats) - 1)#goes through one marker-sample at a time, returns boolean of whether or not equals the category, then generates color value to feed into cmap
        #print("col",col)
        
        if(len(pcomps) == 3):
            m=ax1.scatter(pdat.iloc[i][c1], pdat.iloc[i][c2], pdat.iloc[i][c3], marker=m, s=60,c = [cmap(col)],edgecolor="black")
        else:
            m=ax1.scatter(pdat.iloc[i][c1], pdat.iloc[i][c2], marker=m, s=60,c = [cmap(col)],edgecolor="black")
    if(name == ""):
        name = colorvar.name.replace("Cat"," Cat.")
    
    if trStr:
        legend1h,legend1l=extraLeg(["v","s","^","o"],["early Jul.","late Jul.","mid Aug.","early Oct."])
    else:
        legend1h,legend1l=extraLeg(["s","o"],["Jul.","Oct."])
    
    legend1=fig.legend(handles=legend1h,labels=legend1l,bbox_to_anchor=[0.05,0.09], loc = 'center left',fontsize=9)
    ax1.add_artist(legend1)
    ##
    colorvar.name=colorvar.columns[0]
    if colorvar.name.replace("Cat","") in mdbins.index: #if there are numerical bin dividing values
        l2d=bin_Range_Labs(colorvar.name,mdbins,mdlc) #generate range text labels
        l2l=list(l2d.values())
        
        patches = [mpatches.Patch(color=cmap(i/(len(cats) - 1)), label=l2l[i]) for i in range(len(l2l))]
    else:
        if colorvar.name.replace("Cat","") == "SizeFrac":
            patches = [mpatches.Patch(color=cmap(i/(len(cats) - 1)), label=str(cats[i]+" $\mu$m")) for i in range(len(cats))]
        else:
            patches = [mpatches.Patch(color=cmap(i/(len(cats) - 1)), label=str(cats[i])) for i in range(len(cats))]
            
    legend2=fig.legend(handles=patches, ncol=2,bbox_to_anchor=[.55,.09],loc = 'center', fontsize=9) #bbox_to_anchor=[0.35,0.35]

    ax1.set_xlabel(c1+" ("+str(np.round(pexp.loc[c1]*100,1))+"%)")
    ax1.set_ylabel(c2+" ("+str(np.round(pexp.loc[c2]*100,1))+"%)")
    
    fig.subplots_adjust(bottom = 0.27)
    if(len(pcomps) == 3):
        ax1.set_zlabel(c3+" ("+str(np.round(pexp.loc[c3]*100,1))+"%)")


    if fileOut==True:
        if(len(pcomps) == 3):
            plt.savefig(root+"/pythoncode/"+trStr+"statsout/"+(matnm)+str(colorvar.name)+"3d.pdf", bbox_inches='tight')
        else:
            plt.savefig(root+"/pythoncode/"+trStr+"statsout/"+(matnm)+str(colorvar.name).replace(".","")+str(pcomps[0])+"_"+str(pcomps[1])+".pdf", bbox_inches='tight')
        plt.close()


def extraLeg(mlist,llist):
    fig=plt.figure()
    ax1 = fig.add_subplot(111)

    for count,m in enumerate(mlist):
        ax1.scatter([0,0],[0,0],color="none",marker=m,edgecolor="black",s=100,label=llist[count])
    
    handles, labels = ax1.get_legend_handles_labels()
    
    #print(handles,labels)
    plt.close()
    
    return handles,labels
    


    
def dfplline(PAfr,FLfr,cmf,lb,xlab,xll="",xul="",autob=True,retlims=False,leg=True):
    
    PAfrplt=PAfr.copy()
    FLfrplt=FLfr.copy()
    
    #look up depths by Sample ID
    PAfrplt["Depth"]=cmf["Depth"].loc[PAfrplt.index]
    FLfrplt["Depth"]=cmf["Depth"].loc[FLfrplt.index]
    
    PAfrplt.set_index("Depth",inplace=True)
    PAfrplt.sort_index(inplace=True)
    
    FLfrplt.set_index("Depth",inplace=True)
    FLfrplt.sort_index(inplace=True)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig1 = plt.figure(figsize=(5,7))
    ax1 = fig1.add_subplot(111)

    ax1.plot(PAfrplt.values,PAfrplt.index,color="tab:blue",marker="^",ms=8,mec="black",mew=1.5,label="$>$2.7 $\mu$m")
    ax1.plot(FLfrplt.values,FLfrplt.index,color="tab:orange",marker="o",ms=8,mec="black",mew=1.5,label="0.2 - 2.7 $\mu$m")


    ##Add two horizontal dotted lines for indicating the redoxcline
    if autob:
        ax1.plot((0, np.max([np.max(PAfrplt),np.max(FLfrplt)])*1.2), (15, 15), 'black', linestyle='dashed')
        ax1.plot((0, np.max([np.max(PAfrplt),np.max(FLfrplt)])*1.2), (lb, lb), 'black', linestyle='dashed')
    
    else:
        ax1.plot((0, xul*1.2), (15, 15), 'black', linestyle='dashed')
        ax1.plot((0, xul*1.2), (lb, lb), 'black', linestyle='dashed')

    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")

    ax1.tick_params(axis="x", which="both", labeltop =True,labelbottom = False)
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    plt.ylabel("Depth(m)",fontsize=16)

    ax1.set_xlabel(xlab, fontsize=16)
    ax1.xaxis.set_label_coords(0.45, 1.12)
    
    if autob:
        ax1.set_xlim(np.min([np.min(PAfrplt),np.min(FLfrplt)])-np.max([np.min(PAfrplt),np.min(FLfrplt)])*.13,np.max([np.max(PAfrplt),np.max(FLfrplt)])*1.01)
    else:
        ax1.set_xlim(xll,xul)
    
    if leg==True:
        plt.legend(fontsize=14,frameon=False)
        
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(root+"/pythoncode/statsout/"+(PAfr.name.replace("PA",""))+".pdf")
    plt.close()
    
    if retlims:
        return np.min([np.min(PAfrplt),np.min(FLfrplt)])-np.max([np.min(PAfrplt),np.min(FLfrplt)])*.13,np.max([np.max(PAfrplt),np.max(FLfrplt)])*1.01
        



def col_nan_eval(col,df):
    #print(col)
    tempcol=pd.DataFrame(df[col].copy())

    if df[col].isna().any():
        #print("file contains NaN")
        nanMask=df[col].isna()
        evalVals=tempcol[~nanMask]

    else:
        evalVals=tempcol
    
    return evalVals

def bin_Range_Labs(catCol,binseps,mdlc):

    c=catCol.replace("Cat","")
   # print(c) #match to bincategory format
    
    
    if c in binseps.index: #if there are bins (failsafe)
        #print(binseps.loc[c])
        bseps=binseps.loc[c]["bin_separators"]#pull bin separators(which is a string)
        bseps=bseps.replace("[","")
        bseps=bseps.replace("]","")

        bsepsL=bseps.split(",") #turn into list of strings
        #print(bsepsL)
    
        if c in mdlc.index:
            cRep=mdlc.loc[c]["Label"]
            unit=mdlc.loc[c]["Unit"]
        else:
            cRep=c
            unit = ''
        
        newLabs=[]

        if len(bsepsL)>1: #more than one value
            for i in range(len(bsepsL)):
                if i==0:
                    n= cRep+ "$<$"+bsepsL[i]+" " + unit
                elif i != len(bsepsL):
                    n=bsepsL[i-1]+" " + unit+" $\leq$" + cRep + "$<$ "+bsepsL[i]+" " + unit
                newLabs.append((i,n))

            newLabs.append((len(bsepsL)+1,cRep+"$\geq$"+bsepsL[-1]+" " + unit)) #for last bin, have to tack on after iterating
                
        else:
            newLabs.append((0,cRep+" <"+bsepsL[0]+" " + unit))
            newLabs.append((1,cRep+" >="+bsepsL[0]+" " + unit))
                
        #print("newlabs: ",newLabs,"\n")
        
        newLabsDict=dict(newLabs) #make key-value pairs to look up text label by categorical numeric value

    return newLabsDict

def groupby_bar_plot(df, groupby1, groupby2, data_cat,clrList,legCol,data_cat_err="",yaxlab="",xaxlab="",pltxlim=[],trStr=""):
   
    outname=df.name
    df = df.sort_values(groupby2, ascending=True)
    df = df.sort_values(groupby1, ascending=False)

    #print(df)
    
    df_gb = df.groupby([groupby1,groupby2], sort=False)
    df_gbd = df_gb[data_cat].aggregate(np.sum).unstack()
    
    if data_cat_err:
        df_gbe = df_gb[data_cat_err].aggregate(np.sum).unstack()

        
    fig,ax = plt.subplots(figsize=(5,7))
    
    if data_cat_err:
        if pltxlim:
            df_gbd.plot(kind='barh', ax = ax, xerr = df_gbe,edgecolor="black",capsize=5,color=clrList,xlim=pltxlim)
        else:
            df_gbd.plot(kind='barh', ax = ax, xerr = df_gbe,edgecolor="black",capsize=5,color=clrList)
    else:
        if pltxlim:
            df_gbd.plot(kind='barh', ax = ax, edgecolor="black",color=clrList,xlim=pltxlim)
        else:
            df_gbd.plot(kind='barh', ax = ax, edgecolor="black",color=clrList)
    
    plt.gca().invert_yaxis()
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_coords(0.45, 1.12)
    
    ax.set_xlabel(xaxlab, fontsize=14)
    ax.set_ylabel(yaxlab, fontsize=14)
    
    ax.tick_params(axis='x', which='both', labeltop =True,labelbottom =False)
    ax.xaxis.set_tick_params(labelsize=14)
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    plt.legend(labels=df[legCol])
    plt.savefig(root+"/pythoncode/"+trStr+"statsout/"+(outname)+"bar.pdf",bbox_inches="tight")
    plt.close()

def skewstats(df,mf,tf,sortcat=None, sortfiltvalueL=[],p_thresh=False):
    #transpose- sample=index, otus=cols
    filtered_df=df.T
    
    if sortcat !=None:
        #sort using metadata cat, retain only samples according to filter value(s)
        #reset index col so can use in lambda statement
        filtered_df.reset_index(inplace=True)
        #add sort data category as a column
        filtered_df[sortcat]=filtered_df["index"].apply(lambda x: mf.loc[x][sortcat])
        #put back index column
        filtered_df.set_index("index",inplace=True)
        #use .isin to find entries containing values in list
        filtered_df=filtered_df[filtered_df[sortcat].isin(sortfiltvalueL)]
        #print(filtered_df)
        
        #remove sorting col., re-transpose so that otu=index and sample=col
        filtered_df.drop([sortcat],axis=1,inplace=True)
        
    filtered_df=filtered_df.T
        
    #print(filtered_df)
    
    #pseudocount to avoid zeroes in calculation
    filtered_df=filtered_df.applymap(lambda x:x+1)
    
    if sortcat==None:
        filtered_df.name=df.name
    else:
        filtered_df.name=df.name+"_"+sortcat+"_filt"
    
    #print(filtered_df.name)
    
    #skewedness test
    inds=[] #taxon
    ps=[]   #skewedness test p-value
    ss=[]   #skewedness test statistic
    sv=[]   #skewedness value
    
    for ind in filtered_df.index:
        t=filtered_df.loc[ind].values #test data (counts of a particular ASV in entries)
    
        for n in t: #append the values to the same array to get values up to 8 without changing anything numerically (package requirement)
            #print(n)
            t=np.append(t,n)
            #print(t)
        
        #get statistic and p value
        s,p=sp.stats.skewtest(t)
        sk=sp.stats.skew(t)
        
        #print(s,p,sk)
        if p_thresh:
            #print("pthresh")
            if p<p_thresh:
                #print("< pthresh")
                inds.append(ind)
                ps.append(p)
                ss.append(s)
                sv.append(sk)
                
        else:
            inds.append(ind)
            ps.append(p)
            ss.append(s)
            sv.append(sk)


    skewtestFrame=pd.DataFrame({"#OTU ID":inds,"test statistic":ss,"p-value":ps,"skewedness":sv})
    skewtestFrame.set_index("#OTU ID",inplace=True)
    
    skewtestFrame=skewtestFrame.join(tf)
    skewtestFrame.drop(["Confidence"],axis=1,inplace=True)
    
    skewtestFrame.name=filtered_df.name+"_skewtest"
    skewtestFrame.to_csv(root+"/pythoncode/trapstatsout/"+skewtestFrame.name+".csv",index=True)

#required user input
#file import
root="/Users/ashley/Documents/Research/Gordon'sLab/FGL/amplicon"

taxdict={0:'Domain',1:'Phylum',2:'Class',3:'Order',4:'Family',5:'Genus',6:'Species'}

def watercol_runner():

    #Matrix import
    BCmat=pd.read_csv(root+"/Qiime2_analyzedFGLdatamodified_wreruns/diversity2/braydistance-matrix.tsv",delimiter="\t",index_col=0)
    BCmat.drop(["AC-GL3-0-2-14","AC-GL3-2-7-14"],axis=1,inplace=True)
    BCmat.drop(["AC-GL3-0-2-14","AC-GL3-2-7-14"],axis=0,inplace=True)
    BCmat.name="BCmat"
    BCmat.index.name="#Sample ID"
    
    #print("Deprecation test\n",BCmat.reindex(["AC-GL4-59"],columns=["AC-GL4-59"]),"\n")

    Jaccardmat=pd.read_csv(root+"/Qiime2_analyzedFGLdatamodified_wreruns/diversity2/jaccarddist-matrix.tsv",delimiter="\t",index_col=0)
    Jaccardmat.drop(["AC-GL3-0-2-14","AC-GL3-2-7-14"],axis=1,inplace=True)
    Jaccardmat.drop(["AC-GL3-0-2-14","AC-GL3-2-7-14"],axis=0,inplace=True)
    Jaccardmat.name="Jaccardmat"
    Jaccardmat.index.name="#Sample ID"

    wunimat=pd.read_csv(root+"/Qiime2_analyzedFGLdatamodified_wreruns/diversity2/weightunifr-matrix.tsv",delimiter="\t",index_col=0)
    wunimat.drop(["AC-GL3-0-2-14","AC-GL3-2-7-14"],axis=1,inplace=True)
    wunimat.drop(["AC-GL3-0-2-14","AC-GL3-2-7-14"],axis=0,inplace=True)
    wunimat.name="wunimat"
    wunimat.index.name="#Sample ID"

    #alpha diversity data
    pielouEvenness=pd.read_csv(root+"/Qiime2_analyzedFGLdatamodified_wreruns/diversity2/evenessalpha.tsv",delimiter="\t",index_col=0)
    pielouEvenness.drop(["AC-GL3-0-2-14","AC-GL3-2-7-14"],axis=0,inplace=True)
    pielouEvenness.name="pielou_e"
    pielouEvenness.index.name="#Sample ID"

    ObsOTUs=pd.read_csv(root+"/Qiime2_analyzedFGLdatamodified_wreruns/diversity2/observedOTUs.tsv",delimiter="\t",index_col=0)
    ObsOTUs.drop(["AC-GL3-0-2-14","AC-GL3-2-7-14"],axis=0,inplace=True)
    ObsOTUs.name="obsOTUs"
    ObsOTUs.index.name="#Sample ID"

    ShanDiv=pd.read_csv(root+"/Qiime2_analyzedFGLdatamodified_wreruns/diversity2/shannonvector.tsv",delimiter="\t",index_col=0)
    ShanDiv.drop(["AC-GL3-0-2-14","AC-GL3-2-7-14"],axis=0,inplace=True)
    ShanDiv.name="shanDiv"
    ShanDiv.index.name="#Sample ID"

    #metadata
    catmapfile=pd.read_csv(root+"/pythoncode/newmetadatafiles/cleanedcatmapfl_nq.csv",index_col="#Sample ID",na_values=["NAN"])
    catmapfile.drop(["DescriptionCat"],axis=1,inplace=True)

    contmapfile=pd.read_csv(root+"/pythoncode/newmetadatafiles/cleanedcontmapfl_nq.csv",index_col="#Sample ID",na_values=["NAN"])
    contmapfile.drop(["Description"],axis=1,inplace=True)

    metadatabin=pd.read_csv(root+"/pythoncode/nonqiimeinput/metadatabinning.txt", sep="\t",index_col="category")
    mdlabelconvs=pd.read_csv(root+"/pythoncode/nonqiimeinput/metadatalabels.csv",index_col="Category")

    #clean up distance matrices, create/save heatmaps
    plotlabBCmat=matrixcleanandplot(BCmat,contmapfile,retmat=True)
    plotlabJaccardmat=matrixcleanandplot(Jaccardmat,contmapfile,retmat=True)
    plotlabwunimat=matrixcleanandplot(wunimat,contmapfile,retmat=True)

    #ANOSIM - just use var for bulk, use group by to do analyses on each month separately
    anosimVarList=["SizeFracCat","ORP_mVCat","MonthCat"]

    #for name, group in catmapfile.groupby("MonthCat"): #separate by month first
        #print(name)
    for var in anosimVarList: #var will be group used in anosim

        BCdm = skbio.DistanceMatrix(BCmat,BCmat.index) #format data, sampleIDs
        wunidm = skbio.DistanceMatrix(wunimat,wunimat.index) #format data, sampleIDs
        
        BCaResults=anosim(BCdm, catmapfile, var, permutations=999) #format matrix,metadata,grouping
        wuniaResults=anosim(wunidm, catmapfile, var, permutations=999) #format matrix,metadata,grouping

        BCaResults.to_csv(root+"/pythoncode/statsout/BCanosim"+var+".csv",index=True)
        wuniaResults.to_csv(root+"/pythoncode/statsout/wunianosim"+var+".csv",index=True) 

    #Run PCoAs on the matrices
    #can determine if jaccard and bray-curtis matrices are significantly correlated with one another using Mantel correlation
    #mantel defaults are (x, y, method='pearson', permutations=999, alternative='two-sided', strict=True, lookup=None)
    r, p_value, n = mantel(BCmat, Jaccardmat)
    r2, p_value2, n2 = mantel(BCmat, wunimat)

    manteldf=pd.DataFrame({"r":[r],"p_value":[p_value],"n":[n]})
    manteldf2=pd.DataFrame({"r":[r2],"p_value":[p_value2],"n":[n2]})

    manteldf.to_csv(root+"/pythoncode/statsout/mantelBCJacc.csv",index=False)
    manteldf2.to_csv(root+"/pythoncode/statsout/mantelBCwuni.csv",index=False)

    #perform PCoA on distance matrices, show the proportions explained by each PC
    #samples is the position of the samples in the ordination space, row-indexed by the sample id.
    bc_PCoA=matrixPCoA(BCmat,contmapfile,ret=True)
    j_PCoA=matrixPCoA(Jaccardmat,contmapfile,ret=True)
    wuni_PCoA=matrixPCoA(wunimat,contmapfile,ret=True)

    diffDF=bc_PCoA.samples - j_PCoA.samples
    diffDF2=bc_PCoA.samples - wuni_PCoA.samples

    pltylabs=diffDF.rename(index=lambda x:replFIwRI(x,contmapfile),inplace=False).index
    sns.heatmap(bc_PCoA.samples - j_PCoA.samples,yticklabels=pltylabs,xticklabels=bc_PCoA.samples.columns)
    plt.tight_layout()
    plt.savefig(root+"/pythoncode/statsout/Jaccard_BC_PCoA_diff.pdf")
    plt.close()

    pltylabs=diffDF2.rename(index=lambda x:replFIwRI(x,contmapfile),inplace=False).index
    sns.heatmap(bc_PCoA.samples - wuni_PCoA.samples,yticklabels=pltylabs,xticklabels=bc_PCoA.samples.columns)
    plt.tight_layout()
    plt.savefig(root+"/pythoncode/statsout/wuni_BC_PCoA_diff.pdf")
    plt.close()

    #PCoA results plots, loop through all metadata to examine

    for col in catmapfile.columns:
        #print("categorical map file column ",col,"\n")
        evalVals=col_nan_eval(col,catmapfile)
        #print("evalVals\n",evalVals.index,"\n")
    
    #   pc_3dplot(bc_PCoA.samples.loc[evalVals.index],catmapfile[col].loc[evalVals.index],fileOut=False)
    #    pc_3dplot(j_PCoA.samples.loc[evalVals.index],catmapfile[col].loc[evalVals.index],fileOut=False)

        compsList=[[1,2],[1,3],[2,3]]
        for l in compsList:
            #print("PCoA samples evalVals\n",bc_PCoA.samples.reindex(evalVals.index),"\n")
            #print("PCoA evalVals at category",catmapfile.reindex(evalVals.index,columns=[col]),"\n")
            
#            return (bc_PCoA, catmapfile, BCmat, metadatabin,mdlabelconvs,bc_PCoA, evalVals)
            pc_plot(bc_PCoA.samples.reindex(evalVals.index),catmapfile.reindex(evalVals.index,columns=[col]),BCmat.name,metadatabin,mdlabelconvs,bc_PCoA.proportion_explained,pcomps = l,name="Categ.",fileOut=True)
            pc_plot(wuni_PCoA.samples.reindex(evalVals.index),catmapfile.reindex(evalVals.index,columns=[col]),wunimat.name,metadatabin,mdlabelconvs,wuni_PCoA.proportion_explained,pcomps = l,name="Categ.",fileOut=True)
    #Run PERMANOVA using categorical metadata and distance matrices
    permanovadm(BCmat,catmapfile)
    permanovadm(Jaccardmat,catmapfile)
    permanovadm(wunimat,catmapfile)

    #Cruise and size fraction pielou specific DFs for alpha div indices

    #pielou eveness
    GL4FLpl=pielouEvenness[~pielouEvenness.index.str.endswith("C") & pielouEvenness.index.str.contains("GL4")]
    GL4FLpl.name="GL4FLpielou"

    GL4PApl=pielouEvenness[pielouEvenness.index.str.endswith("C") & pielouEvenness.index.str.contains("GL4")]
    GL4PApl.name="GL4PApielou"

    GL3FLpl=pielouEvenness[pielouEvenness.index.str.contains("GL3-0-2")]
    GL3FLpl.name="GL3FLpielou"

    GL3PApl=pielouEvenness[pielouEvenness.index.str.contains("GL3-2-7")]
    GL3PApl.name="GL3PApielou"

    #observed OTUs
    GL4FLobsOTUs=ObsOTUs[~ObsOTUs.index.str.endswith("C") & ObsOTUs.index.str.contains("GL4")]
    GL4FLobsOTUs.name="GL4FLObsOTUs"

    GL4PAobsOTUs=ObsOTUs[ObsOTUs.index.str.endswith("C") & ObsOTUs.index.str.contains("GL4")]
    GL4PAobsOTUs.name="GL4PAObsOTUs"

    GL3FLobsOTUs=ObsOTUs[ObsOTUs.index.str.contains("GL3-0-2")]
    GL3FLobsOTUs.name="GL3FLObsOTUs"

    GL3PAobsOTUs=ObsOTUs[ObsOTUs.index.str.contains("GL3-2-7")]
    GL3PAobsOTUs.name="GL3PAObsOTUs"

    #Shannon Diversity

    GL4FLshanDiv=ShanDiv[~ShanDiv.index.str.endswith("C") & ShanDiv.index.str.contains("GL4")]
    GL4FLshanDiv.name="GL4FLshanDiv"

    GL4PAshanDiv=ShanDiv[ShanDiv.index.str.endswith("C") & ShanDiv.index.str.contains("GL4")]
    GL4PAshanDiv.name="GL4PAshanDiv"

    GL3FLshanDiv=ShanDiv[ShanDiv.index.str.contains("GL3-0-2")]
    GL3FLshanDiv.name="GL3FLshanDiv"

    GL3PAshanDiv=ShanDiv[ShanDiv.index.str.contains("GL3-2-7")]
    GL3PAshanDiv.name="GL3PAshanDiv"

    #pielou evenness profiles
    ll,ul=dfplline(GL3PApl,GL3FLpl,contmapfile,21.0,"Pielou Evenness",retlims=True,leg=False)
    dfplline(GL4PApl,GL4FLpl,contmapfile,20.0,"Pielou Evenness",ll,ul,False,leg=False)

    #observed OTU profiles
    ll,ul=dfplline(GL3PAobsOTUs,GL3FLobsOTUs,contmapfile,21.0,"Observed OTUs",retlims=True)
    dfplline(GL4PAobsOTUs,GL4FLobsOTUs,contmapfile,20.0,"Observed OTUs",ll,ul,False)

    #Shannon diversity profiles
    ll,ul=dfplline(GL3PAshanDiv,GL3FLshanDiv,contmapfile,21.0,"Shannon Diversity",retlims=True)
    dfplline(GL4PAshanDiv,GL4FLshanDiv,contmapfile,20.0,"Shannon Diversity",ll,ul,False,leg=False)

    plt.rc('text', usetex=False)

    ##Kruskall-Wallace tests with alpha diversity table eveness/pielou_e vector
    filenames = []
    filtered_group_comparisons = []
    k_w_all_out=[]

    for col in catmapfile.columns:

        #print(col)
        evalVals=col_nan_eval(col,catmapfile)

        
        #create data range strings for metadata columns using the metadatabinfile. Must cast the categorical values as
        #floats because otherwise cannot deal with cols w NaNs
        if ("SizeFrac" not in col) & ("Month" not in col) & ("oxcat" not in col) & ("redoxclinepos" not in col) & ("year" not in col):
            bindict=creatbindict(col,metadatabin)
    

        #join alpha diversity to NaN-cleaned metadata column by index
        pielouMetadataColPair=pd.concat([evalVals,pielouEvenness],axis=1,join="inner")
    
        names = [] # to append each of the metadata column's unique categorical values and the corresponding number of pielou_e values as n value
        groups = [] #to append the corresponding pielou values

        #use groupby to find pielou_e values by metadata category
        for name, group in pielouMetadataColPair.groupby(col):
            #print("name: ",name,"group: ",group) #name is unique value in metadata column, group is their pieloue values. When dict or series is passed into the 'by' position argument, the Series or dict VALUES will be used to determine the groups

            groups.append(list(group[pielouEvenness.name]))#from that metadatagroup, append the pielou_e values
                    
            if ("SizeFrac" not in col) & ("Month" not in col) & ("oxcat" not in col) & ("redoxclinepos" not in col) & ("year" not in col):
                names.append("%s (n=%d)" % (bindict[str(float(name))], len(group)))
            else:
                names.append("%s (n=%d)" % (name, len(group)))
            
        boxwhiskplot(groups,names,col)
    
        filenames.append(col+"pielou_e")
        #print(names)

        # perform Kruskal-Wallis across all groups, append results for each metdatadata variable to make into dataframe at the end
        kw_H_all, kw_p_all = sp.stats.mstats.kruskalwallis(*groups)
        #print("kwall:",kw_H_all, kw_p_all,"\n")
    
        #initialize an emptry string with a space to join the names list into a string for output
        nameStinitalize=", "
    
        #reformat categorical metadata column name for output
        tableCol=col.replace("Cat","")
        tableCol=tableCol.rstrip()
    
        #append output entry
        k_w_all_out.append((tableCol,nameStinitalize.join(names),kw_H_all,kw_p_all))

        # perform pairwise Kruskal-Wallis across all pairs of groups and correct for multiple comparisons

        kw_H_pairwise = []
    
        for i in range(len(names)): #for each unique metadata value in a metadata column
            for j in range(i): #for a range of the same length
                try:
                    H, p = sp.stats.mstats.kruskalwallis(groups[i], #one of the group's pielou values
                                                        groups[j]) #against all groups' pielou values
                    kw_H_pairwise.append([names[j], names[i], H, p])
                except ValueError:
                    #print("oops")
                    filtered_group_comparisons.append(
                        ["%s:%s" % (column, names[i]),
                        "%s:%s" % (column, names[j])])
                     
        kw_H_pairwise = pd.DataFrame(
            kw_H_pairwise, columns=["Group 1", "Group 2", "H", "p-value"])
        kw_H_pairwise.set_index(["Group 1", "Group 2"], inplace=True)
        kw_H_pairwise["q-value"] = multipletests(
        kw_H_pairwise["p-value"], method="fdr_bh")[1] #corrects for multiple comparison false discovery rate, Bonferroni correction not appropriate bc too many combinations
        kw_H_pairwise.sort_index(inplace=True)
        #print("adjusted pairwise:",kw_H_pairwise)

        pairwise_fn = "kruskal-wallis-pairwise-"+tableCol.replace("/","")+".csv"
        #print(pairwise_fn,"\n")

        kw_H_pairwise.to_csv(root+"/pythoncode/statsout/"+pairwise_fn,index=True)
    
    k_w_all_out_dataframe=pd.DataFrame(k_w_all_out,columns=["metadata variable","groups","H-value","P-value"])
    k_w_all_out_dataframe.sort_values("P-value", axis=0, ascending=True, inplace=True, kind="quicksort")

    k_w_all_out_dataframe.to_csv(root+"/pythoncode/statsout/kruskal-wallis-all.csv",index=False)



def trap_runner():
    
    dropCols=["tr-blk","GEN-DONOR","MOCK-EVEN","MOCK-STAG"]
    monthNumericDict={"early July":7.1,"late July":7.2,"early October":10.0,"mid August":8.0}
    
    #metadata
    catmapfile=pd.read_csv(root+"/pythoncode/newmetadatafilestraps/cleanedcatmapfl.csv",index_col="#Sample ID",na_values=["NAN"])
    contmapfile=pd.read_csv(root+"/pythoncode/newmetadatafilestraps/cleanedcontmapfl.csv",index_col="#Sample ID",na_values=["NAN"])
    metadatabin=pd.read_csv(root+"/Qiime2_analyzed_FGL_traps/trapmetadatabinning.txt", sep="\t",index_col="category")
    mdlabelconvs=pd.read_csv(root+"/pythoncode/nonqiimeinput/metadatalabels.csv",index_col="Category")

    #taxonomy file
    taxfile=pd.read_csv(root+"/Qiime2_analyzed_FGL_traps/taxa/taxonomy.tsv",delimiter="\t",index_col="Feature ID")

    #Matrix import
    BCmat=pd.read_csv(root+"/Qiime2_analyzed_FGL_traps/diversity2/braydistance-matrix.tsv",delimiter="\t",index_col="#Sample ID")
    BCmat.drop(dropCols,axis=1,inplace=True)
    BCmat.drop(dropCols,axis=0,inplace=True)
    BCmat.name="BCmat_trap"

    wunimat=pd.read_csv(root+"/Qiime2_analyzed_FGL_traps/diversity2/weightunifr-matrix.tsv",delimiter="\t",index_col="#Sample ID")
    wunimat.drop(dropCols,axis=1,inplace=True)
    wunimat.drop(dropCols,axis=0,inplace=True)
    wunimat.name="wunimat_trap"
    
    ShanDiv=pd.read_csv(root+"/Qiime2_analyzed_FGL_traps/diversity2/shannonvector.tsv",delimiter="\t",index_col=[0])
    ShanDiv.drop(dropCols,axis=0,inplace=True)
    ShanDiv.name="shanDiv_trap"
    ShanDiv.index.name="#Sample ID"
    ShanDiv.reset_index(inplace=True)
    
    ShanDiv["month"]=ShanDiv["#Sample ID"].apply(lambda x: catmapfile.loc[x]["MonthCat"])
    ShanDiv["monthnumeric"]=ShanDiv["month"].apply(lambda x: monthNumericDict[x])
    ShanDiv["trapPos"]=ShanDiv["#Sample ID"].apply(lambda x: catmapfile.loc[x]["posCat"])
 
    pielouEvenness=pd.read_csv(root+"/Qiime2_analyzed_FGL_traps/diversity2/evenessalpha.tsv",delimiter="\t",index_col=[0])
    pielouEvenness.drop(dropCols,axis=0,inplace=True)
    pielouEvenness.name="Pielou_trap"
    pielouEvenness.index.name="#Sample ID"
    pielouEvenness.reset_index(inplace=True)
    
    pielouEvenness["month"]=pielouEvenness["#Sample ID"].apply(lambda x: catmapfile.loc[x]["MonthCat"])
    pielouEvenness["monthnumeric"]=pielouEvenness["month"].apply(lambda x: monthNumericDict[x])
    pielouEvenness["trapPos"]=pielouEvenness["#Sample ID"].apply(lambda x: catmapfile.loc[x]["posCat"])
    
    ASVfilteredtable=pd.read_csv(root+"/Qiime2_analyzed_FGL_traps/dada2_output/exported_table_filt/table_filt.tsv",delimiter="\t",index_col="#OTU ID",skiprows=[0])
    ASVfilteredtable.drop(dropCols,axis=1,inplace=True)
    ASVfilteredtable.name="ASVfilteredtabletraps"

    #alpha diversity bar plots
    groupby_bar_plot(ShanDiv, "trapPos", "monthnumeric", "shannon",["white","lightgray","dimgray","black"],"month",xaxlab=r"\textit{H}",pltxlim=[0,10],trStr="trap")
    groupby_bar_plot(pielouEvenness, "trapPos", "monthnumeric", "pielou_e",["white","lightgray","dimgray","black"],"month",xaxlab="Pielou Evenness",pltxlim=[0,1.2],trStr="trap")
    #skewedness test
    skewstats(ASVfilteredtable,catmapfile,taxfile,sortcat="posCat", sortfiltvalueL=["shallow"],p_thresh=0.06)

    #clean up distance matrices, create/save heatmaps
    plotlabBCmat=matrixcleanandplot(BCmat,contmapfile,retmat=True,watercol=False)
    plotlabwunimat=matrixcleanandplot(wunimat,contmapfile,retmat=True,watercol=False)

    #perform PCoA on distance matrices, show the proportions explained by each PC
    #samples is the position of the samples in the ordination space, row-indexed by the sample id.
    bc_PCoA=matrixPCoA(BCmat,contmapfile,ret=True,tr=True)
    wuni_PCoA=matrixPCoA(wunimat,contmapfile,ret=True,tr=True)
    
    #ANOSIM, trap position - just for bulk, use group by to do analyses separately if want to

    dm = skbio.DistanceMatrix(wunimat,wunimat.index) #format data, sampleIDs
    aBulkResults=anosim(dm, catmapfile, "posCat", permutations=999) #format matrix,metadata,grouping
    aBulkResults.to_csv(root+"/pythoncode/trapstatsout/anosimTrapPos.csv",index=True)

    for col in catmapfile.columns:
        evalVals=col_nan_eval(col,catmapfile)
        #print(evalVals.index)
        #pc_plot(bc_PCoA.samples.loc[evalVals.index],catmapfile[col].loc[evalVals.index],BCmat.name,metadatabin,mdlabelconvs,bc_PCoA.proportion_explained,pcomps = [1,2,3],name="Categ.",fileOut=False)
        #pc_plot(wuni_PCoA.samples.loc[evalVals.index],catmapfile[col].loc[evalVals.index],wunimat.name,metadatabin,mdlabelconvs,wuni_PCoA.proportion_explained,pcomps = [1,2,3],name="Categ.",fileOut=False)
        compsList=[[1,2],[1,3],[2,3]]
        for l in compsList:
            pc_plot(wuni_PCoA.samples.reindex(evalVals.index),catmapfile.reindex(evalVals.index,columns=[col]),wunimat.name,metadatabin,mdlabelconvs,wuni_PCoA.proportion_explained,pcomps = l,name="Categ.",trStr="trap",fileOut=True)
