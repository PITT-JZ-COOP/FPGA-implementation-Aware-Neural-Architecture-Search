 # -*- coding: utf-8 -*
'''
Author: Weiwen Jiang, Xinyi Zhang
'''

import math
import networkx as nx
import sys
import random
import os
import sys
import math
from math import *
import csv
import time
import copy 

class DESIGN_PARA:
    def __init__(self,IMG_SIZE, BITWIDTH,IMG_CHANNEL,DSP_RESOURCE):
        self.IMG_SIZE = IMG_SIZE
        self.BITWIDTH = BITWIDTH
        self.IMG_CHANNEL = IMG_CHANNEL
        self.DSP_RESOURCE = DSP_RESOURCE
        self.DSP_ALLOCATION =[]
# layer_performance(Tm,Tn,Tr,Tc,M,N,R,C,K,BITWIDTH)

    def dsp_allo(self,predic_actions):
        total_mac=0
        for i in range(1,len(predic_actions),2):
            if i==1:
                total_mac=total_mac + self.IMG_SIZE*self.IMG_SIZE*predic_actions[i]*predic_actions[i-1]*predic_actions[i-1]*self.IMG_CHANNEL
            else:
                total_mac=total_mac + self.IMG_SIZE*self.IMG_SIZE*predic_actions[i]*predic_actions[i-1]*predic_actions[i-1]*predic_actions[i-2]
        for i in range(1,len(predic_actions),2):
            if i==1:
                dsp_allocated=self.DSP_RESOURCE*(self.IMG_SIZE*self.IMG_SIZE*predic_actions[i]*predic_actions[i-1]*predic_actions[i-1]*self.IMG_CHANNEL)/total_mac
                if math.floor(dsp_allocated)<1:
                    self.DSP_ALLOCATION.append(1)
                else:
                    self.DSP_ALLOCATION.append(math.floor(dsp_allocated))
            else:
                dsp_allocated=self.DSP_RESOURCE*(self.IMG_SIZE*self.IMG_SIZE*predic_actions[i]*predic_actions[i-1]*predic_actions[i-1]*predic_actions[i-2])/total_mac
                if math.floor(dsp_allocated)<1:
                    self.DSP_ALLOCATION.append(1)
                else:
                    self.DSP_ALLOCATION.append(math.floor(dsp_allocated))
        print("DDDDDDDDDDDDDDDDDDebug DSP_ALLO",self.DSP_ALLOCATION)
        return self.DSP_ALLOCATION


    def get_design(self,predic_actions): ## here, layers denote a convolution operation
        layers_size=[]
        layer_size=[0]*5
        # layer_size=[M,N,R,C,K]
        # print(predic_actions)
        for i in range(0,len(predic_actions)-1,2):
            # print(i)
            if i == 0:  
                layer_size[4]=predic_actions[i]
                layer_size[0]=predic_actions[i+1]
                layer_size[1]=self.IMG_CHANNEL
                layer_size[2]=self.IMG_SIZE # RC= 1 + (N-K+Padding)/stride
                layer_size[3]=self.IMG_SIZE # RC= 1 + (N-K+Padding)/stride
            else:
                layer_size[4]=predic_actions[i]
                layer_size[0]=predic_actions[i+1]
                layer_size[1]=predic_actions[i-1]
                layer_size[2]=self.IMG_SIZE # RC= 1 + (N-K+Padding)/stride
                layer_size[3]=self.IMG_SIZE # RC= 1 + (N-K+Padding)/stride
            # print(layer_size)
            layers_size.append(copy.deepcopy(layer_size))
            # layers_size.append(layer_size)
        return layers_size

    def get_design_para(self,predic_actions):
        layers_size=self.get_design(predic_actions)
        self.DSP_ALLOCATION=self.dsp_allo(predic_actions)

        layers_para=[]
        for i in range(len(layers_size)):
            dsp_bound=self.DSP_ALLOCATION[i]
            
            if self.BITWIDTH==32:
                dsp_bound=math.floor(dsp_bound/5)
            if self.BITWIDTH==16:
                dsp_bound=dsp_bound
            obj_layer=layers_size[i]
            [M,N,R,C,K]=layers_size[i]

            MAX=sys.maxsize
            layer_para=[0]*9
            for Tm in range(1,M+1):
                for Tn in range(1,N+1):
                    for Tr in range(20,21):
                        for Tc in range(20, 21):
                            if Tm*Tn>dsp_bound:
                                continue                           
                            propose_layer=self.layer_performance(Tm,Tn,Tr,Tc,M,N,R,C,K)
                            # print(propose_layer)
                            if propose_layer[1] < MAX:
                                MAX = propose_layer[1]
                                layer_para[0]=M
                                layer_para[1]=N
                                layer_para[2]=R
                                layer_para[3]=C
                                layer_para[4]=Tm
                                layer_para[5]=Tn
                                layer_para[6]=Tr
                                layer_para[7]=Tc
                                layer_para[8]=K
                                # print(layer_para)
            layers_para.append(copy.deepcopy(layer_para))
        # print(layers_para,"end geting design")
        return layers_para

    def layer_performance(self,Tm,Tn,Tr,Tc,M,N,R,C,K):
        
    # Operation numbers
        OP = R*C*M*K*K*N*2
    # AXI in_stream to IP, in_stream is 32 bit (bitwidth) by default.
        W_p=2
        I_p=2
        O_p=2
        A_p=2
        

        tW_mem=Tm*Tn*K*K/W_p
        tI_mem=Tn*Tr*Tc/I_p
        tComp=K*K*Tr*Tc
        tO_mem=Tm*Tr*Tc/O_p   

        EachLat = []
        # single FPGA double buffer 1/thourouput and latency 
        Lat1 = max(tI_mem,tW_mem,tComp)
        Lat2 = max(ceil(N/Tn)*Lat1,tO_mem)
        Lat = ceil(R/Tr) * ceil(C/Tc) * ceil(M/Tm) * Lat2 #+ (tO_mem + Lat1)
        TH = float(OP)/(Lat)*10**8
        EachLat = [tI_mem,tW_mem,tComp,tO_mem/ceil(N/Tn)]                      

        bI = 2 * Tn * ceil(Tr*Tc*self.BITWIDTH/18000)
        bO = 2 * Tm * ceil(Tr*Tc*self.BITWIDTH/18000)
        if self.BITWIDTH ==32:
            bW = 2* Tm * Tn * ceil(K*K*self.BITWIDTH/18000) #float
        else:
            bW = Tm * Tn * ceil(4*K*K*self.BITWIDTH/18000)    #fix point             
        BRAM = bI+bO+bW
        BRAM_R=BRAM/1824

        if(self.BITWIDTH == 16):
            DSP = Tm*Tn                                        
        elif(self.BITWIDTH==32):
            DSP = 5*Tm*Tn                                        
        else:
            DSP = 0
       
        return TH,Lat
    '''
    Cn denotes the convolution to laye n, layer 1 is the starter(input img). Thus, para N, Tn in C0 is not used.  
    '''


    def get_conv_names(self,predic_actions):
        layersname=["c1"] # c1 is regarding as the operation to input img, doing nothing
        for i in range(0,int(len(predic_actions)/2)):
            temp_name= "c" + str(i+2)
            layersname.append(temp_name)
        return layersname
    # print(get_conv_names(predic_actions))

    def get_layers(self,predic_actions):
        layers_para_list=self.get_design_para(predic_actions)
        layersname=self.get_conv_names(predic_actions)
        layers=[]
        # print(layersname)
        for i in range(len(layersname)):
            layer_temp=[1]*10
            if i==0:
                layer_temp[0]=layersname[i]
                layer_temp[1]=self.IMG_CHANNEL
                layer_temp[2]=self.IMG_CHANNEL
                layer_temp[3]=self.IMG_SIZE
                layer_temp[4]=self.IMG_SIZE
                layer_temp[5]=self.IMG_CHANNEL
                layer_temp[6]=self.IMG_CHANNEL
                layer_temp[7]=10
                layer_temp[8]=10


            else:
                layer_temp[0]=layersname[i]
                layer_temp=layer_temp[0:1]
                layer_temp.extend(layers_para_list[i-1])
            layers.append(copy.deepcopy(layer_temp))
        return layers
