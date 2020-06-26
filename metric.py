import torch

def E1_loss(y_pred, y_true):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    
    _t, _p = y_true, y_pred
    
    return torch.mean(torch.mean((_t - _p) ** 2, axis = 1)) / 2e+04

def E2_loss(y_pred, y_true):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    _t, _p = y_true, y_pred
    
    return torch.mean(torch.mean((_t - _p) ** 2 / (_t + 1e-06), axis = 1))

def total_loss(y_pred, y_true):
    xy_t, xy_p = y_true[:,:2], y_pred[:,:2]
    mv_t, mv_p = y_true[:,2:], y_pred[:,2:]
    
    e1 = torch.mean(torch.mean((xy_t - xy_p) ** 2, axis = 1)) / 2e+04
    e2 = torch.mean(torch.mean((mv_t - mv_p) ** 2 / (mv_t + 1e-06), axis = 1))
    
    return e1 + e2