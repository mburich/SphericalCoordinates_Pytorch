import torch as torch
import numpy as np

class SphericalCoordiInv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        r =x[:,0]
        cos_t = torch.cos(x[:,1:])
        sin_t = torch.sin(x[:,1:])
        sin_mul = 0*sin_t
        for i in range(sin_t.shape[1]):
            sin_mul[:,i] = torch.prod(sin_t[:,0:(i+1)],1)

        bb1 = r.repeat(x.shape[1]-1,1).t()*cos_t
        ff1 = torch.cat((bb1, r.reshape(-1, 1)), dim=1)

        bb2 = r.repeat(x.shape[1]-1,1).t()*sin_mul
        tt = x[:,0]*0.0+1.0#.to('cuda:0'),
        vv = torch.cat((tt.reshape(-1,1),sin_mul),dim=1)
        g = ff1*vv
        #print(x)
        #print(g)

        ctx.save_for_backward(x)
        return g

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.zeros(input.shape[1],input.shape[0],input.shape[1]).to('cuda')
#        grad_input = torch.zeros(input.shape[1],input.shape[0],input.shape[1])
        if grad_output.shape[1]==2:
            aa = (torch.cos(input[:,1])*grad_output[:,0])
            bb = (-input[:,0]*torch.sin(input[:,1])*grad_output[:,0])
            cc = (torch.sin(input[:,1])*grad_output[:,1])
            dd = (input[:,0]*torch.cos(input[:,1])*grad_output[:,1])
            grad_input[0, :, 0] = aa
            grad_input[0, :, 1] = bb
            grad_input[1, :, 0] = cc
            grad_input[1, :, 1] = dd

        elif grad_output.shape[1]==3:
            grad_input[0, :, 0] = (torch.cos(input[:,1])*grad_output[:,0])
            grad_input[0, :, 1] = -input[:,0]*(torch.sin(input[:,1])*grad_output[:,0])
            grad_input[0, :, 2] = 0.0*grad_input[0, :, 2]
            grad_input[1, :, 0] = torch.sin(input[:,1])*torch.cos(input[:,2])*grad_output[:,1]
            grad_input[1, :, 1] = input[:,0]*torch.cos(input[:,1])*torch.cos(input[:,2])*grad_output[:,1]
            grad_input[1, :, 2] = -input[:,0]*torch.sin(input[:,1])*torch.sin(input[:,2])*grad_output[:,1]
            grad_input[2, :, 0] = torch.sin(input[:,1])*torch.sin(input[:,2])*grad_output[:,2]
            grad_input[2, :, 1] = input[:,0]*torch.cos(input[:,1])*torch.sin(input[:,2])*grad_output[:,2]
            grad_input[2, :, 2] = input[:,0]*torch.sin(input[:,1])*torch.cos(input[:,2])*grad_output[:,2]
        #print(grad_input)
        return grad_input

class SphericalCoordiInvExpanded(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xt):
        l = int(xt.shape[1]/2)
        x = xt[:,0:l]
        r =x[:,0]
        cos_t = torch.cos(x[:,1:])
        sin_t = torch.sin(x[:,1:])
        sin_mul = 0*sin_t
        for i in range(sin_t.shape[1]):
            sin_mul[:,i] = torch.prod(sin_t[:,0:(i+1)],1)

        bb1 = r.repeat(x.shape[1]-1,1).t()*cos_t
        ff1 = torch.cat((bb1, r.reshape(-1, 1)), dim=1)

        bb2 = r.repeat(x.shape[1]-1,1).t()*sin_mul
        tt = x[:,0]*0.0+1.0#.to('cuda:0'),
        vv = torch.cat((tt.reshape(-1,1),sin_mul),dim=1)
        g = ff1*vv
        #print(x)
        #print(g)

        ctx.save_for_backward(xt)
        g = torch.cat((g,xt[:,l:]), dim=1)
        return g

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input = torch.zeros(input.shape[1],input.shape[0],input.shape[1]).to('cuda')
#        grad_input = torch.zeros(input.shape[1],input.shape[0],input.shape[1])
        if grad_output.shape[1]==4:
            #print(grad_output[:,2])
            #print(grad_input[2, : ,2].size())
            aa = (torch.cos(input[:,1])*grad_output[:,0])
            bb = (-input[:,0]*torch.sin(input[:,1])*grad_output[:,0])
            cc = (torch.sin(input[:,1])*grad_output[:,1])
            dd = (input[:,0]*torch.cos(input[:,1])*grad_output[:,1])
            grad_input[0, :, 0] = aa
            grad_input[0, :, 1] = bb
            grad_input[1, :, 0] = cc
            grad_input[1, :, 1] = dd
            grad_input[2, : ,2] = grad_output[:,2]
            grad_input[3, : ,3] = grad_output[:,3]


        elif grad_output.shape[1]==6:
            grad_input[0, :, 0] = (torch.cos(input[:,1])*grad_output[:,0])
            grad_input[0, :, 1] = -input[:,0]*(torch.sin(input[:,1])*grad_output[:,0])
            grad_input[0, :, 2] = 0.0*grad_input[0, :, 2]
            grad_input[1, :, 0] = torch.sin(input[:,1])*torch.cos(input[:,2])*grad_output[:,1]
            grad_input[1, :, 1] = input[:,0]*torch.cos(input[:,1])*torch.cos(input[:,2])*grad_output[:,1]
            grad_input[1, :, 2] = -input[:,0]*torch.sin(input[:,1])*torch.sin(input[:,2])*grad_output[:,1]
            grad_input[2, :, 0] = torch.sin(input[:,1])*torch.sin(input[:,2])*grad_output[:,2]
            grad_input[2, :, 1] = input[:,0]*torch.cos(input[:,1])*torch.sin(input[:,2])*grad_output[:,2]
            grad_input[2, :, 2] = input[:,0]*torch.sin(input[:,1])*torch.cos(input[:,2])*grad_output[:,2]
            grad_input[3, : ,3] = grad_output[:,3]
            grad_input[4, : ,4] = grad_output[:,4]
            grad_input[5, : ,5] = grad_output[:,5]

        #print(grad_input)
        return grad_input

class SphericalCoordi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rori = torch.sqrt(torch.sum(x**2,dim=1))

        r=0.0*rori.repeat(x.shape[1]-1, 1).t()
        for i in range(r.shape[1]):
            r[:,i] = torch.sqrt(torch.sum(x[:,i:]**2,1))
        phi = torch.acos(x[:,:-1]/r)
        phi_l = phi
        phi_l2 = 2*np.pi-phi_l
        phi_l[x[:, -1] < 0,-1] = phi_l2[x[:, -1] <0,-1]
        g = torch.cat((rori.reshape(-1, 1), phi_l), 1)
        #g = torch.cat((x,g),dim=1)
        g[g!=g] = 0.0
        return g


class SphLayer(torch.nn.Module):
    def __init__(self):
        super(SphLayer, self).__init__()
    #@staticmethod
    def forward(self, input):
        return SphericalCoordi.apply(input)


class SphLayerInv(torch.nn.Module):
    def __init__(self):
        super(SphLayerInv, self).__init__()
    def forward(self, input):
        return SphericalCoordiInv.apply(input)

class SphLayerInvExpanded(torch.nn.Module):
    def __init__(self):
        super(SphLayerInvExpanded, self).__init__()
    def forward(self, input):
        return SphericalCoordiInvExpanded.apply(input)


class SphericalCoordiExpanded(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rori = torch.sqrt(torch.sum(x**2,dim=1))

        r=0.0*rori.repeat(x.shape[1]-1, 1).t()
        for i in range(r.shape[1]):
            r[:,i] = torch.sqrt(torch.sum(x[:,i:]**2,1))
        phi = torch.acos(x[:,:-1]/r)
        phi_l = phi
        phi_l2 = 2*np.pi-phi_l
        phi_l[x[:, -1] < 0,-1] = phi_l2[x[:, -1] <0,-1]
        g = torch.cat((rori.reshape(-1, 1), phi_l), 1)
        g = torch.cat((g,x),dim=1)
        g[g!=g] = 0.0
        return g


class SphLayerExpanded(torch.nn.Module):
    def __init__(self):
        super(SphLayerExpanded, self).__init__()
    #@staticmethod
    def forward(self, input):
        return SphericalCoordiExpanded.apply(input)

