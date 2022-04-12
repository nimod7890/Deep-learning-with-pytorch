import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data

import cv2
import numpy as np
import os.path as osp
from itertools import product as product
from math import sqrt as sqrt

import xml.etree.ElementTree as ET

from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

from utils.match import match


class makeDatapathList():
    def __init__(self, rootpath):
        self.rootpath=rootpath
        self.img_path_template=osp.join(rootpath,'JPEGImage','%s.jpg')
        self.anno_path_template=osp.join(rootpath,'Annotations','%s.xml')
    
    def make_list(self,phase):
        id_names=osp.join(self.rootpath+f"ImageSets/Main/{phase}.txt")
        img_list=[]
        anno_list=[]
        for line in open(id_names):
            file_id=line.strip()
            img_path=self.img_path_template%file_id
            anno_path=self.anno_path_template%file_id
            img_list.append(img_path)
            anno_list.append(anno_path)
        return [img_list,anno_list]

    def __call__(self,phase):
        return self.make_list(phase)





class anno_xml2list(object):
    def __init__(self, classes):
        self.classes=classes
    
    def __call__(self, xml_path,width,height):
        
        ret=[]
        xml=ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):

            #감지 어려운 경우 annotation에서 제외
            if int(obj.find('difficult').text)==1:
                continue
            
            bound_box=[]
            name=obj.find('name').text.lower().strip()
            info=obj.find('bndbox')
            points=['xmin','ymin','xmax','ymax']

            for point in points:

                #(1,1)이 원점이므로 (0,0)으로 변경
                cur_pixel=int(info.find(point).text)-1
                cur_pixel/=width if point=='xmin' or point=='xmax' else height

                bound_box.append(cur_pixel)
            
            label_idx=self.classes.index(name)
            bound_box.append(label_idx)
            ret.append(bound_box)
            
        return np.array(ret)




class dataTransform():
    def __init__(self, input_size,color_mean):
        self.data_transform={
            'train':Compose([
                ConvertFromInts(),#int2float32
                ToAbsoluteCoords(),#annotation data 규격화
                PhotometricDistort(), #random color distortion
                Expand(color_mean), #확대
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),#annotation data 0-1 사이 값으로 규격화
                Resize(input_size),
                SubtractMeans(color_mean) #RGB 평균값 빼기
            ]),
            'val':Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }
    def __call__(self,img,phase,boxes,labels):
        return self.data_transform[phase](img,boxes,labels)





class VOCDataset(data.Dataset):
    def __init__(self,img_list,anno_list,phase,transform,transform_anno):
        self.img_list=img_list
        self.anno_list=anno_list
        self.phase=phase
        self.transform=transform
        self.transform_anno=transform_anno

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
        im,gt,h,w=self.pull_item(idx)
        return im,gt
    
    def pull_item(self,idx):

        image_file_path=self.img_list[idx]
        img=cv2.imread(image_file_path)
        height,width,_channels=img.shape

        anno_file_path=self.anno_list[idx]
        anno_list=self.transform_anno(anno_file_path,width,height)

        img,boxes,labels=self.transform(img,self.phase,anno_list[:,:4],anno_list[:,4])
        img=torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

        gt=np.hstack((boxes,np.expand_dims(labels,axis=1)))
        
        return img,gt,height,width






def od_collate_fn(batch):
    imgs=[]
    annos=[]
    for sample in batch:
        imgs.append(sample[0])
        annos.append(torch.FloatTensor(sample[1]))
    imgs=torch.stack(imgs,dim=0)
    return imgs,annos







def VGGModule():
    layers=[]
    #faster than list()
    in_channels=3
    cfg=[64,64,'M',128,128,'M',256,256,256,'MC',512,512,512,'M',512,512,512]
    for v in cfg:
        if v=='M':
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
            # faster than '+='
        elif v=='MC':
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True))
            #ceil mode: float 올림
            #floor mode(default): float 내림
        else:
            layers+=[
                nn.Conv2d(in_channels,v,kernel_size=3,padding=1),
                nn.ReLU(inplace=True)
            ]
            #faster than .extend()
            in_channels=v
    
    layers+=[
        nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
        nn.Conv2d(512,1024,kernel_size=3,padding=6,dilation=6),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024,1024,kernel_size=1),
        nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)








def ExtrasModule():
    in_channels=1024 #output of vgg module
    cfg=[256,512,128,256,128,256,128,256]
    layers=[
        nn.Conv2d(in_channels,cfg[0],kernel_size=(1)),
        nn.Conv2d(cfg[0],cfg[1],kernel_size=(3),stride=2,padding=1),
        nn.Conv2d(cfg[1],cfg[2],kernel_size=(1)),
        nn.Conv2d(cfg[2],cfg[3],kernel_size=(3),stride=2,padding=1),
        nn.Conv2d(cfg[3],cfg[4],kernel_size=(1)),
        nn.Conv2d(cfg[4],cfg[5],kernel_size=(3)),
        nn.Conv2d(cfg[5],cfg[6],kernel_size=(1)),
        nn.Conv2d(cfg[6],cfg[7],kernel_size=(3)),
    ]
    #activation function(ReLU)은 foward propagation 부분에서
    return nn.ModuleList(layers)







def LocConfModule(num_classes=21,bbox_aspect_num=[4,6,6,6,4,4]):

    loc_layers,conf_layers=[],[]
    cfg=[512,1024,512,256,256,256]
    
    for idx,v in enumerate(cfg):
        loc_layers.append(nn.Conv2d(v,bbox_aspect_num[idx]*4,kernel_size=3,padding=1))
        conf_layers.append(nn.Conv2d(v,bbox_aspect_num[idx]*num_classes,kernel_size=3,padding=1))

    return nn.ModuleList(loc_layers),nn.ModuleList(conf_layers)



class L2Norm(nn.Module):
    
    def __init__(self,input_channels=512,scale=20):
        super(L2Norm,self).__init__()
        self.weight=nn.Parameter(torch.Tensor(input_channels))
        self.scale=scale
        self.reset_parameters()
        self.eps=1e-10

    def reset_parameters(self):
        init.constant_(self.weight,self.scale)

    def forward(self,x):
        norm=x.pow(2).sum(dim=1,keepdim=True).sqrt()+self.eps
        x=torch.div(x,norm)
    
        weights=self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out=weights*x
        return out







class DBox(object):
    def __init__(self,cfg):
        super(DBox,self).__init__()
        self.image_size=cfg['input_size']
        self.feature_maps=cfg['feature_maps']
        self.num_priors=len(cfg['feature_maps'])
        self.steps=cfg['steps'] #size of defaultbox pixel
        self.min_sizes=cfg['min_sizes']
        self.max_sizes=cfg['max_sizes']
        self.aspect_ratios=cfg['aspect_ratios'] #정사각형 dbox의 화면비
    
    def make_dbox_list(self):
        mean=[]
        for k,f in enumerate(self.feature_maps):
            for i,j in product(range(f),repeat=2):
                f_k=self.image_size/self.steps[k] #size of source map
                cx,cy=(j+0.5)/f_k,(i+0.5)/f_k #중심 좌표 (0-1 값으로 정규화되어있음)
                '''
                default box 네 종류
                - 6종류일 경우 3:1, 1:3 비율 추가
                '''
                #1:1 small
                s_k=self.min_sizes[k]/self.image_size 
                mean+=[cx,cy,s_k,s_k]
                #1:1 big
                s_k_big=sqrt(s_k*(self.max_sizes[k]/self.image_size))
                mean+=[cx,cy,s_k_big,s_k_big]
                #1:2,2:1
                for ar in self.aspect_ratios[k]:
                    sq=sqrt(ar)
                    mean+=[cx,cy,s_k*sq,s_k/sq]
                    mean+=[cx,cy,s_k/sq,s_k*sq]
        output=torch.Tensor(mean).view(-1,4) #dbox를 tensor 형태로 변환
        output.clamp_(max=1,min=0) #크기를 최소 0, 최대 1로 바꿔 dbox가 화면 밖으로 나가지 않도록 함
        return output







def decode(loc,dbox_list):
    boxes=torch.cat((
        dbox_list[:,:2]+loc[:,:2]*0.1*dbox_list[:,2:],
        dbox_list[:,2:]*torch.exp(loc[:,2:]*0.2)),
        dim=1
    )
    boxes[:,:2]-=boxes[:,2:]/2 #(cx, cy)->(xmin,ymin)
    boxes[:,2:]+=boxes[:,:2] #(w,h)->(xmax,ymax)
    
    return boxes








def nm_suppression(boxes,scores,overlap=0.45,top_k=200):
    count=0 #최종 bounding box 개수
    keep=scores.new(scores.size(0)).zero_().long() #최종 bounding box index
    
    #bounding box 넓이  
    x1,y1,x2,y2=[boxes[:,i] for i in range(4)] 
    area=torch.mul(x2-x1,y2-y1)

    _x1,_y1,_x2,_y2,_w,_h=[boxes.new()]*6
    
    #신뢰도 상위 top_k개의 bounding box idx 
    v,idx=scores.sort()
    idx=idx[-top_k:]
    
    while idx.numel()>0:
        max_idx=idx[-1]

        keep[count]=max_idx
        count+=1
        if idx.size(0)==1:
            break
        idx=idx[:-1] 

        for v,_v in [[x1,_x1],[y1,_y1],[x2,_x2],[y2,_y2]]:
            # 0-idx 사이 bounding box 정보를 _v에 저장
            torch.index_select(v,0,idx,out=_v) 
            #xmin,ymin,xmax, ymax 범위 제한
            if _v is _x1 or _v is _y1:
                _v=torch.clamp(_v,min=v[max_idx]) 
            else: 
                _v=torch.clamp(_v,max=v[max_idx]) 

        for v,two,one in [[_w,_x2,_x1],[_h,_y2,_y1]]:
            v.resize_as_(two)
            v=two-one 
            v=torch.clamp(v,min=0.0) #ReLU 같은 역할: 음수 다 0
        inter=_w*_h #넓이

        rem_areas=torch.index_select(area,0,idx) #bouding box 원래 면적
        union=(rem_areas-inter)+area[max_idx] #area, clamped area (OR)
        IoU=inter/union #측정한 구역 전체에서 bounding box 겹치는 비율
        #IoU<overlap 인 idx만 남김
        idx=idx[IoU.le(overlap)] #le: less than(==equal to)
        
    return keep,count








class Detect(Function):
    def __init__(self,conf_thresh=0.01,top_k=200,nms_thresh=0.45):
        self.softmax=nn.Softmax(dim=-1)
        self.conf_thresh=conf_thresh #신뢰도 기준
        self.top_k=top_k 
        self.nms_thresh=nms_thresh #IoU 기준
     
    def forward(self,loc_data,conf_data,dbox_list,conf_thresh=0.01,top_k=200,nms_thresh=0.45):
        self.softmax=nn.Softmax(dim=-1)
        self.conf_thresh=conf_thresh #신뢰도 기준
        self.top_k=top_k 
        self.nms_thresh=nms_thresh #IoU 기준
        
        num_batch=loc_data.size(0)
        # num_dbox=loc_data.size(1)
        num_classes=conf_data.size(2)

        conf_data=self.softmax(conf_data) #normalization

        output=torch.zeros(num_batch,num_classes,self.top_k,5) #init output init
        conf_preds=conf_data.transpose(2,1) #transpose: 2개의 차원 순서 맞교환 [a,b,c]->[a,c,b] 
        for i in range(num_batch):
            decoded_boxes=decode(loc_data[i],dbox_list)
            conf_scores=conf_preds[i].clone() 
            for cl in range(1,num_classes):
                #신뢰도 기준 통과한 bounding box 인덱스 마스킹
                c_mask=conf_scores[cl].gt(self.conf_thresh) #gt: greater than(넘으면 1, 못넘으면 0)
                scores=conf_scores[cl][c_mask] # bounding box 수 (?)
                if scores.nelement()==0:  
                    continue
                l_mask=c_mask.unsqueeze(1).expand_as(decoded_boxes) #decoded box에 맞게 크기 조정
                boxes=decoded_boxes[l_mask].view(-1,4) #dimension (n) -> (n,4)
                idx,count=nm_suppression(boxes,scores,self.nms_thresh,self.top_k) #중복 bbox 삭제
                output[i,cl,:count]=torch.cat((scores[idx[:count]].unsqueeze(1),boxes[idx[:count]]),1)
        return output





class SSD(nn.Module):
    def __init__(self,phase,cfg):
        super(SSD,self).__init__()
        self.phase=phase
        self.num_classes=cfg["num_classes"]
        #SSD Networks
        self.vgg=VGGModule()
        self.extras=ExtrasModule()
        self.L2NOrm=L2Norm()
        self.loc,self.conf=LocConfModule(cfg["num_classes"],cfg["bbox_aspect_num"])
        #Default box
        dbox=DBox(cfg)
        self.dbox_list=dbox.make_dbox_list()
        if phase=='inference':
            self.detect=Detect()
    
    def forward(self,x):
        sources,loc,conf=[],[],[]
        
        for k in range(23):
            x=self.vgg[k](x)
        x=self.L2NOrm(x)
        sources.append(x)

        for k in range(23,len(self.vgg)):
            x=self.vgg[k](x)
        sources.append(x)

        for i,k in enumerate(self.extras):
            x=F.relu(k(x),inplace=True)
            if i%2==1:
                sources.append(x)
        
        for (x,l,c) in zip(sources,self.loc,self.conf):
            loc.append(l(x).permute(0,2,3,1).contiguous())
            conf.append(c(x).permute(0,2,3,1).contiguous())
            #permute: 두개 이상의 차원 순서 교체
            #contiguous: 메모리상에서 연속적으로 존재하도록 함 (view 사용위해서 필요)
        loc=torch.cat([o.view(o.size(0),-1)for o in loc],1)
        loc=loc.view(loc.size(0),-1,4)
        conf=torch.cat([o.view(o.size(0),-1)for o in conf],1)
        conf=conf.view(conf.size(0),-1,self.num_classes)

        output=(loc,conf,self.dbox_list)
        if self.phase=="inference":
            return self.detect.apply(output[0],output[1],output[2])
            #apply 안 쓰면 runtimeerror
        else: return output



class MultiBoxLoss(nn.Module):
    def __init__(self,jaccard_thresh=0.5, neg_pos=3,device='cpu'):
        super(MultiBoxLoss,self).__init__()
        self.jaccard_thresh=jaccard_thresh #for match function
        self.negpos_ratio=neg_pos #negative dbox:positive dbox
        self.device=device

    def forward(self,predictions,targets):
        loc_data,conf_data,dbox_list=predictions
        num_batch=loc_data.size(0)
        num_dbox=loc_data.size(1)
        num_classes=conf_data.size(2)

        '''dbox에 가장 가까운 정답 bbox의 정보 저장''' 

        #init
        conf_t_label=torch.LongTensor(num_batch,num_dbox).to(self.device) #label
        loc_t=torch.Tensor(num_batch,num_dbox,4).to(self.device) #location 

        for idx in range(num_batch):
            # 현재 minibatch의 정답 annotation 
            bbox=targets[idx][:,:-1].to(self.device) #bbox 
            bbox_labels=targets[idx][:,-1].to(self.device) #label: [object_1, label_1, o_2,l_2,...]

            # init
            dbox=dbox_list.to(self.device)

            # get label, location info using match function
            # if jaccard overlap< 0.5, set label index 0(background class)
            
            variance=[0.1,0.2] #DBox에서 BBox로의 보정 과정에서 필요한 계수
            match(self.jaccard_thresh,bbox,dbox,variance,bbox_labels,loc_t,conf_t_label,idx)
        
        pos_mask=conf_t_label>0 #object 감지된 bbox 추출 (positive dbox가 계산된 bbox)

        '''location loss: Smooth L1 Loss'''
        pos_idx=pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data) #pos_mask를 loc_data 크기로 변형
        loc_p=loc_data[pos_idx].view(-1,4) #dbox location info 
        loc_t=loc_t[pos_idx].view(-1,4) #bbox location info
        
        loss_l=F.smooth_l1_loss(loc_p,loc_t,reduction='sum')

        '''class loss: cross entropy'''
        # 1. CLASS PREDICTION LOSS
        batch_conf=conf_data.view(-1,num_classes)
        loss_c=F.cross_entropy(batch_conf,conf_t_label.view(-1),reduction='none') # 차원 보존을 위해 더하지 않고 reduction='none'


        # 2. HARD NEGATIVE MINING

        # (1) if object detect(label > 1), loss will be 0
        num_pos=pos_mask.long().sum(1,keepdim=True) #object class의 예측 수 for each minibatch
        loss_c=loss_c.view(num_batch,-1)
        loss_c[pos_mask]=0 


        # (2) background dbox 개수 조정
        num_neg=torch.clamp(num_pos*self.negpos_ratio,max=num_dbox) # object class(num_pos): background class = self.negpos_ratio:1


        # (3) loss 낮은 dbox 제외

        # loss 크기 큰 순서대로 정렬한 idx 기억
            # loss_idx[idx_rank[0]]=원래 loss_c의 0번째 요소 
        _,loss_idx=loss_c.sort(1,descending=True) #loss 큰 순으로 정렬
            # loss_idx: 내림차순으로 정렬된 loss의 원래 idx 값
        _,idx_rank=loss_idx.sort(1) #rank 지정
            # idx_rank: 다시 원 상태로 되돌린 후, loss_idx의 순서 기억 -> 내림차순 시 새로운 idx
        
        #num_neg보다 loss 큰(idx_rank 값 낮은) dbox 추출
        neg_mask=idx_rank<(num_neg).expand_as(idx_rank) 


        # (4) hard negative mining
        
        pos_idx_mask=pos_mask.unsqueeze(2).expand_as(conf_data) #positive dbox conf
        neg_idx_mask=neg_mask.unsqueeze(2).expand_as(conf_data) #negative dbox conf

        # pos나 neg 중 mask 가 1인 index를 추출 
        conf_hnm=conf_data[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1,num_classes) 
        conf_t_label_hnm=conf_t_label[(pos_mask+neg_mask).gt(0)] 
        
        
        # (5) loss 
        loss_c=F.cross_entropy(conf_hnm,conf_t_label_hnm,reduction='sum')

        N=num_pos.sum() #object detect된 bbox 개수

        loss_l/=N
        loss_c/=N

        return loss_l,loss_c