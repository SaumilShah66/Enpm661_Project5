#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import time
import argparse

class FinalMap():
    def __init__(self, height, width, clr, fact=1):
        """Initializes final map
        height:     row dimension [pixels]
        width:      column dimension [pixels]
        c:        clearance from map border"""
        height = height + 1
        width = width + 1
        clr = clr*fact
        self.c = clr
        self.f = fact
        self.grid= np.ones([height,width,3], dtype ='uint8')*255
        if(clr != -1):
            self.grid[0:(clr+1),:,0] = 0
            self.grid[height-(clr+1):height,:,0] = 0
            self.grid[:,0:(clr+1),0] = 0
            self.grid[:, width-(clr+1):width,0] = 0

    # Obstacle in top right
    def circ(self, radius, h, w):
        """Customizable circle obstacle
            radius:     radius dimention [pixels]
            h:          circle center location in map's coordinate system
            w:          circle center location in map's coordinate system"""
        h = h*self.f
        w = w*self.f
        radius = radius*self.f
        finalRad = radius + self.c
        if(h-finalRad<0):
            ha=0;
        else:
            ha= h - finalRad

        if(h+finalRad >= self.grid.shape[0]):
            hb= self.grid.shape[0]
        else:
            hb= h + finalRad

        if(w-finalRad<0):
            wa=0;
        else:
            wa= w - finalRad

        if(w+finalRad >= self.grid.shape[1]):
            wb= self.grid.shape[1]
        else:
            wb= w + finalRad

        for h_ in range(ha, hb):
            for w_ in range(wa, wb):
                eqn= (h_-h)**2 + (w_-w)**2
                if(eqn<=(finalRad**2)):
                    self.grid[h_,w_,0] = 0
        return

    # Obstacle in bottom left
    def sqr(self, side, h, w):
        """Fixed square shape for obstacle"""
        side = side*self.f
        h = h*self.f
        w = w*self.f
        hlen = side/2
        h1, w1 = h - hlen, w - hlen # top left
        h2, w2 = h - hlen, w + hlen # top right
        h3, w3 = h + hlen, w + hlen # bottom right
        h4, w4 = h + hlen, w - hlen # bottom left

        l1y = h1 - self.c
        l2x = w2 + self.c
        l3y = h3 + self.c
        l4x = w4 - self.c

        for x in range(self.grid.shape[1]):
            for y in range (self.grid.shape[0]):
                if(y>=l1y and y<=l3y and x<=l2x and x>=l4x):
                    self.grid[y,x, 0]=0
        # Highlighting the rectangle vertices
        self.grid[int(h1),int(w1),0:2]= 0
        self.grid[int(h2),int(w2),0:2]= 0
        self.grid[int(h3),int(w3),0:2]= 0
        self.grid[int(h4),int(w4),0:2]= 0
        return

# Class to have map representation as math equations and methods to check if a point
# is in free space or in the obstacle
class Obs():
    def __init__(self, height, width, clr, fact=1):
        """Initializes final map
        ht:     row dimension [pixels]
        wd:     column dimension [pixels]
        c:      clearance from map border"""
        self.ht = height + 1
        self.wd = width + 1
        clr = clr*fact
#         clr = int(math.ceil(clr))
        self.c = clr
        self.f = fact

    def bound(self, cord):
        # cord is (x,y)
        c1 = cord.copy()
        c1 = c1 - [self.c, self.c]
        c1= c1>=[0,0]
        rslt_bount_lw = np.bitwise_and(c1[:,0], c1[:,1]).reshape(-1,1)      
        c2 = cord.copy()
        c2 = c2 - [self.wd-self.c, self.ht-self.c]
        c2= c2<=[0,0]
        rslt_bount_up = np.bitwise_and(c2[:,0], c2[:,1]).reshape(-1,1)
        rslt = np.bitwise_and(rslt_bount_lw, rslt_bount_up).reshape(-1,1)
        return rslt

    def circ(self, radius, h, w, cord):
        cord = cord - [h,w]
        dis = np.linalg.norm(cord, axis=1) # eqn= np.sqrt((y-h)**2 + (x-w)**2)
        finalRad = radius + self.c
        rslt = (dis>finalRad).reshape(-1,1)
        return rslt
    
    def circ1(self, radius, h, w, cord):
        cord = cord - [w,h]
        dis = np.linalg.norm(cord, axis=1) # eqn= np.sqrt((y-h)**2 + (x-w)**2)
        finalRad = radius + self.c
        rslt = (dis>finalRad).reshape(-1,1)
        return rslt

    def sqr(self,side, h, w, x, y):
        side = side*self.f
        h = h*self.f
        w = w*self.f
        hlen = side/2
        h1, w1 = h - hlen, w - hlen # top left
        h2, w2 = h - hlen, w + hlen # top right
        h3, w3 = h + hlen, w + hlen # bottom right
        h4, w4 = h + hlen, w - hlen # bottom left

        l1y = h1 - self.c
        l2x = w2 + self.c
        l3y = h3 + self.c
        l4x = w4 - self.c

        if(y>=l1y and y<=l3y and x<=l2x and x>=l4x):
            return False
        return True
    
    def notObs(self, cord):
        cord = cord.reshape(-1,2) 
        rb = self.bound(cord)
        rc1 = self.circ(100,200,225,cord)
        rc2 = self.circ(50,400,100,cord)
        rc3 = self.circ(40,375,375,cord)
        rslt = np.bitwise_and(rb,rc1).reshape(-1,1)
        rslt = np.bitwise_and(rslt,rc2).reshape(-1,1)
        rslt = np.bitwise_and(rslt,rc3).reshape(-1,1)
        return rslt
    
    def notObs1(self, cord):
        cord = cord.reshape(-1,2) 
        rb = self.bound(cord)
        rc1 = self.circ1(100,200,225,cord)
        rc2 = self.circ1(50,400,100,cord)
        rc3 = self.circ1(40,375,375,cord)
        rslt = np.bitwise_and(rb,rc1).reshape(-1,1)
        rslt = np.bitwise_and(rslt,rc2).reshape(-1,1)
        rslt = np.bitwise_and(rslt,rc3).reshape(-1,1)
        return rslt

class AllNodes():
    # Initialize class object
    def __init__(self, height, width, depth):
        """Initializes to keep track of all nodes explored"""
        self.h_ = height+1
        self.w_ = width+1
        self.d_ = depth
        self.allStates=[]
        self.visited= np.zeros([self.h_, self.w_])
        self.ownIDarr= np.ones([self.h_, self.w_], dtype='int64')*(-1)
        self.pIDarr= np.ones([self.h_, self.w_], dtype='int64')*(-1)
        self.cost2come= np.ones([self.h_, self.w_], dtype=np.float64)*(float('inf'))
        self.actDone= np.ones([self.h_, self.w_], dtype='int64')*(-1)
        self.totCost= np.zeros([self.h_, self.w_], dtype='f')

    # Function to get update cost in cost2come array
    def updateCost2Come(self, cord, cost, pid, actId, currState, goalState, scale):
        if(self.cost2come[cord[0], cord[1]] > cost):
            self.cost2come[cord[0], cord[1]] = cost
            c2g = scale*heu(currState,goalState)
            self.pIDarr[cord[0], cord[1]] = pid
            self.actDone[cord[0], cord[1]] = actId
            self.totCost[cord[0], cord[1]] = cost + c2g
        return

    # Function to add new unique node in the Nodes data set
    def visit(self, node, state):
        ownId = int(len(self.allStates))
        self.ownIDarr[node[0], node[1]] = ownId
        self.visited[node[0], node[1]] = 2
        self.allStates.append(state)
        self.totCost[node[0], node[1]]= self.totCost[node[0], node[1]]*(-1)
        return ownId

    # Function to get own id
    def getOwnId(self,node):
        return self.ownIDarr[node[0], node[1]]

    # Function to get parent id
    def getParentId(self,node):
        return self.pIDarr[node[0], node[1]]

    # Function to get state of the node i.e. coordinate [h,w[]
    def getStates(self, idx):
        return self.allStates[idx]
    
    def removeLastState(self):
        del self.allStates[-1]
        return int(len(self.allStates))
    
    def minCostIdx(self, step, explore):
        try:
            totTemp = self.totCost*explore
            newMin= np.min(totTemp[totTemp>0])
            if(newMin == float('inf')):
                status= False
                new_parentState = -1
                p_state = -1
            else:
                status= True
                index= np.argwhere(totTemp==newMin)[0]
                idx = self.pIDarr[index[0],index[1]]
                p_state = self.getStates(idx)
                degree = self.actDone[index[0],index[1]]
                new_parentState = p_state+ action(step, degree+p_state[2])
                new_parentState[2] = new_parentState[2] - p_state[2]
            return status, newMin, new_parentState, p_state
        except:
            newMin = float ('inf')
            status= False
            new_parentState = -1
            p_state = -1
            return status, newMin, new_parentState, p_state

def thresh(states, tx=0.5, ty=0.5):
    states= states.reshape(-1,2)
    statesInt = np.array(states, dtype ='int32')
    diff = states - statesInt
    diff[:,0] =  np.where(diff[:,0]<ty, 0, 1)
    diff[:,1] =  np.where(diff[:,1]<tx, 0, 1)
    node = statesInt + diff
    return np.array(node, dtype='int32')

def action(step, degree):
    if(degree >=360):
        degree = degree-360
    t= math.radians(degree)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype= 'f')
    base_move = np.array([step,0], dtype= 'f')
    move_hw= (np.matmul(rot, base_move.T))
    return np.array([-move_hw[1], move_hw[0], degree], dtype='f')

def heu(current, goal): #defining heuristic function as euclidian distance between current node and goal
    h = math.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)
    return h

def threshold_state(state, tx=0.5, ty=0.5, tt=30):
    node= np.ones(3, dtype='f')*(-1)
    if(state[0] - int(state[0]) <ty):
        node[0] = int(state[0])
    else:
        node[0] = int(state[0]) + 1

    if(state[1] - int(state[1]) < tx):
        node[1] = int(state[1])
    else:
        node[1] = int(state[1]) + 1

    if(state[2] - (tt*(state[2]//tt)) >= tt/2):
        node[2] = (tt*(state[2]//tt)) + tt
    else:
        node[2] = (tt*(state[2]//tt))
    return np.array(node, dtype='int32')

def goalReached(curr, goal):
    rad = 5
    dis = np.sqrt((curr[0] - goal[0])**2 + (curr[1] - goal[1])**2)
    if (dis<= rad):
        # print('goal clearance radius ', rad)
        # print('actual accepted distance from goal', dis)
        return 1
    else:
        return 0

def Astar(initCord, clrn, exp=0, scale=2):
    actionSet= []
    fact = 1
    hc= 250
    wc = -250
    clrr = np.array(clrn, dtype='f')
    cart= np.array(initCord, dtype='f')
    print(clrr) 
    if(clrr[0] >= 0):
        clr = clrr[0]
    else:
        print('Clearnace cannot be negative.....\nTerminating')
        return actionSet, -1
    # clr = float(int(clr*100)/100.)
    rad = 17.7 # 35,4/2 cm
    tot= int(math.ceil((rad+clr))*fact)
    map1 = FinalMap(500, 500, tot)
    map1.circ(100, 200, 225)
    map1.circ(50, 400, 100)
    map1.circ(40, 375, 375)
    obsChk= Obs(500, 500, tot)
    cv2.imwrite('grid_init.jpg',map1.grid)
    map2= cv2.imread('grid_init.jpg')
    map2 = cv2.cvtColor(map2, cv2.COLOR_BGR2RGB)
    plt.grid()
    plt.ion()
    plt.imshow(map2)
    print('Generating voronoi diagram.....')
    density = 0.8 # in percentage of total pixels
    samples = np.random.randint(tot, 501-tot, size=(int(500*500*density), 2))
    vor = Voronoi(samples, incremental=True)
    print('Voronoi generated....')
    plt.show()
    init= np.array([hc-cart[1], cart[0]-wc, cart[2]], dtype='f')
    if(not obsChk.notObs(init[:-1])):
        print('Start position cannot be in the obstacle.....\nTerminating')
        return actionSet, -1
    checkInp= True
    while checkInp:
        print('Total clearance set at: ', tot)
        print('Enter the step "d" in integer (1<=d<total clearance (value as above)): ')
        step_d= int(input())
        if(step_d<1 or step_d>= tot):
            print('Wrong step size, try again.....')
        else:
            checkInp= False
	# checkInp= True
	# while checkInp:
	#     print('Enter the initial starting coordinates in y as (0,500) & x as (0,500) and angle; ')
	#     print('With origin at top left, Enter in the order of y, x, theta [separated by commas]: ')
	#     cart= np.array(input(), dtype='f')
	#     if(len(cart)!=3):
	#         print 'Wrong input....Try again \nNote: Only 3 numbers needed inside the map boundaries'
	#     else:
	#         init= np.array([cart[0], cart[1], cart[2]], dtype='f')
	#         if(not obsChk.notObs(init[:-1])):
	#             print 'Start position cannot be in the obstacle.....Try again'
	#         else:
	#             checkInp = False

	checkInp= True
	while checkInp:
	    print('Enter the goal coordinates with origin at the top left as y, x [separated by commas]: ')
	    fs= np.array(input(), dtype='f')
	    if(len(fs)!=2):
	        print 'Wrong input....Try again \nNote: Only 2 numbers needed inside the map boundaries'
	    else:
	        finalState= np.array([fs[0], fs[1],0], dtype='f')
	        if(not obsChk.notObs(finalState[:-1])):
	            print 'Goal position cannot be in the obstacle.....Try again'
	        else:
	            checkInp = False
	height = 500
	width = 500
	parentState = init
	parentNode = threshold_state(parentState)
	finalNode = threshold_state(finalState)
	samples2 = np.array([parentState[:-1], finalState[:-1], [finalState[0]-2, finalState[1]-2],
	                    [finalState[0]+2, finalState[1]-2], [finalState[0]-2, finalState[1]+2],
	                    [finalState[0]+2, finalState[1]+2]])
	Voronoi.add_points(vor,samples2)
	Voronoi.close(vor)
	A = vor.vertices.copy() #(x,y) with origin at bottom left
	A[:,1] = 500 - A[:,1]
	isOKvert = obsChk.notObs1(A)
	allOKvert = A*isOKvert
	allOKvert = allOKvert[np.unique(np.nonzero(allOKvert)[0])]
	allOKvertThres = thresh(allOKvert)
	graph = AllNodes(height,width,12)
	parentState = init
	parentNode = threshold_state(parentState)
	finalNode = threshold_state(finalState)
	parentCost= 0
	graph.updateCost2Come(parentNode, parentCost, 0,-1, parentState, finalState, scale)
	parent_ownId= graph.visit(parentNode, parentState)
	reached= goalReached(parentNode, finalNode)
	found = False
	if(reached):
	    found =True
	    print('Input position is within the goal region')

	for i in allOKvertThres:
	    graph.visited[i[1], i[0]] = 1

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	if exp:
		vw = cv2.VideoWriter('maskVideo.avi', fourcc, 10.0, (501, 501))
		vw2 = cv2.VideoWriter('visitVideo.avi', fourcc, 10.0, (501, 501))
		tempImg = np.zeros([501,501,3], dtype='uint8')
		tempVisit = np.zeros([501,501,3], dtype ='uint8')
		tempVisit[:,:,1] =graph.visited*120
		tempVisit[:,:,0] =graph.visited*120
	itr =0
	flag =0
	count = 0
	step_d =3
	Rmin = tot
	print('Processing...Please wait')
	start_time = time.time()
	while(found != True):
	    itr+=1
	    mask = FinalMap(500, 500, -1)
	    mask.circ(Rmin, parentNode[0], parentNode[1])
	    finalMask = ((mask.grid[:,:,0]/255)-1)/255 #dtype uint8 makes -1 value as 255 as [0,255] is only allowed
	    explore = ((graph.visited*finalMask)==1)*1
	    if exp:
		    tempImg[:,:,0] =explore*255
		    vw.write(tempImg)
	    for angle in range(0,360,30): # Iterating for all possible angles
	        chk =True
	        for sd in range(1,step_d+1):
	            step = action(sd, angle+parentState[2])
	            tempState = parentState + step
	            tempState[2] = tempState[2] - parentState[2]
	            if(not obsChk.notObs(tempState[:-1])):
	                chk = False
	                break;
	        if(chk):
	            actId = angle
	            tempNode = threshold_state(tempState)
	            if(explore[tempNode[0], tempNode[1]] == 1):
	                tempCost2Come = parentCost + step_d
	                graph.updateCost2Come(tempNode, tempCost2Come, parent_ownId, actId, tempState, finalState, scale)
	    status, minCost, new_parentState, org_parentState = graph.minCostIdx(step_d, explore)
	    if(status):
	        parentState = new_parentState
	        parentNode = threshold_state(parentState)
	        map1.grid[parentNode[0], parentNode[1],0]=255
	        map1.grid[parentNode[0], parentNode[1],1]=0
	        map1.grid[parentNode[0], parentNode[1],2]=0
	        parentCost = graph.cost2come[parentNode[0], parentNode[1]]
            parent_ownId = graph.visit(parentNode, parentState)
            # cv2.imwrite('gridExplored.jpg',map1.grid)
            if exp:
                tempVisit[:,:,2] =graph.visited*120
                vw2.write(tempVisit)
            reached= goalReached(parentState, finalState)
            if(reached):
                found =True
                print('Solved')
                break
        else:
            lenRemain = graph.removeLastState()
            if lenRemain:
                parentState = graph.getStates(-1)
                parentNode = threshold_state(parentState)
                parentCost = graph.cost2come[parentNode[0], parentNode[1]]
                parent_ownId = graph.getOwnId(parentNode)
            else:
                print('No solution exist, terminating....')
                count=1
                return actionSet, -1
    if exp:
    	vw.release()
    	vw2.release()
    cv2.imwrite('gridExplored.jpg',map1.grid)
    plt.imshow(map1.grid)
    plt.show()
    plt.pause(0.0001)
    print("Time explored = %2.3f seconds " % (time.time() - start_time))
    map2= cv2.imread('gridExplored.jpg')
    map3= cv2.imread('grid_init.jpg')

    if(not count):
	    reached_state = graph.getStates(int(len(graph.allStates))-1)
	    reached_node = threshold_state(reached_state)
	    ans = graph.getOwnId(reached_node)
	    print '\nYellow area shows all the obstacles and White area is the free space'
	    print 'Blue color show all the explored Nodes (area)'
	    print 'Red line shows optimal path (traced from start node to final node)'

    allNodes=[]
    nextState= graph.getStates(ans)
    nextNode = threshold_state(nextState)
    g_actId = graph.actDone[nextNode[0], nextNode[1]]
    allNodes.append(nextNode)
    actionSet.append(g_actId)
    while(ans!=0 and count==0):
        startState= nextState
        startNode = nextNode
        ans= graph.getParentId(startNode)
        nextState= graph.getStates(ans)
        nextNode = threshold_state(nextState)
        g_actId = graph.actDone[nextNode[0], nextNode[1]]
        allNodes.append(nextNode)
        actionSet.append(g_actId)
    idx = len(allNodes)-1
    vw1 = cv2.VideoWriter('Vid_backTrack_on_explored.avi', fourcc, 10.0, (501, 501))
    vw2 = cv2.VideoWriter('Vid_backTrack.avi', fourcc, 10.0, (501, 501))
    while idx >0:
        startNode = allNodes[idx]
        nextNode = allNodes[idx-1]
        idx -=1
        cv2.line(map2, (startNode[1],startNode[0]),(nextNode[1],nextNode[0]),(0,0,255),1)
        cv2.line(map3, (startNode[1],startNode[0]),(nextNode[1],nextNode[0]),(0,0,255),1)
        vw1.write(map2)
        vw2.write(map3)
    vw1.release()
    vw2.release()
    plt.imshow(map2)
    plt.show()
    plt.pause(0.0001)
    cv2.imwrite('back_tracking_explored.jpg',map2)
    cv2.imwrite('back_tracking.jpg',map3)
    if(count==0):
        actionSet.reverse()
    input('Path computed: ')
    plt.ioff()
    return actionSet, step_d

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--exp', default=0, type = int, help='Set to 1 save videos of exploration (default: 0)')
    Parser.add_argument('--scale', default=1.0, type = float, help='Weight for heuristic function (default: 1)')
    Parser.add_argument('--Start', default="[10, 10, 90]", help='Give inital point')
    Parser.add_argument('--End', default="[0, -3, 0]", help='Give final point')
    Parser.add_argument('--Clearance', default=0.1, help='Give robot clearance')
    Args = Parser.parse_args()

    initial = [float(i) for i in Args.Start[1:-1].split(',')]
    goal = [float(i) for i in Args.End[1:-1].split(',')]
    clearance = [float(Args.Clearance)]
    exp = Args.exp
    scale = Args.scale
    Astar(initial, clearance, exp, scale)

if __name__ == '__main__':
    main()
