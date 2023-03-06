import torch
import torch.nn as nn
import torch.distributed as dist


class SemanticMetrics(nn.Module):
    def __init__(self, num_classes, distributed: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.distributed = distributed

        self.tp = torch.nn.Parameter(torch.zeros(num_classes, dtype=int), requires_grad=False)
        self.fp = torch.nn.Parameter(torch.zeros(num_classes, dtype=int), requires_grad=False)
        self.fn = torch.nn.Parameter(torch.zeros(num_classes, dtype=int), requires_grad=False)
        self.tn = torch.nn.Parameter(torch.zeros(num_classes, dtype=int), requires_grad=False)
    
    def reset(self):
        self.tp[:] = 0
        self.fp[:] = 0
        self.fn[:] = 0
        self.tn[:] = 0
        
    def update(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Update the intersections and unions tensors with the values calculated from the given prediction and label tensors.

        Parameters:
            prediction (torch.Tensor): Tensor with shape (batch_size, num_classes, height, width) and dtype int containing the class predictions.
            label (torch.Tensor): Tensor with shape (batch_size, height, width) and dtype int containing the class labels.

        Returns:
            None
        """
        num_classes = prediction.shape[1]
        
        # label [b, h, w] -> [b, num_classes, h, w]
        for i in range(num_classes):          
            label_mask = (label == i).int()
            pred_mask = prediction[:, i, :, :]
            tp = torch.bitwise_and(label_mask, pred_mask).sum()
            fp = torch.bitwise_and(label_mask == 0, pred_mask == 1).sum()
            fn = torch.bitwise_and(label_mask == 1, pred_mask == 0).sum()
            tn = torch.bitwise_and(label_mask == 0, pred_mask == 0).sum()

            self.tp[i] += tp
            self.fp[i] += fp
            self.fn[i] += fn
            self.tn[i] += tn
    
    def _sync(self):
        if self.distributed:
            tp_list = [torch.zeros_like(self.tp) for _ in range(dist.get_world_size())]
            fp_list = [torch.zeros_like(self.fp) for _ in range(dist.get_world_size())]
            fn_list = [torch.zeros_like(self.fn) for _ in range(dist.get_world_size())]
            tn_list = [torch.zeros_like(self.tn) for _ in range(dist.get_world_size())]

            dist.all_gather(tp_list, self.tp)
            dist.all_gather(fp_list, self.fp)
            dist.all_gather(fn_list, self.fn)
            dist.all_gather(tn_list, self.tn)

            tp = torch.sum(torch.stack(tp_list, dim=0), dim=0)
            fp = torch.sum(torch.stack(fp_list, dim=0), dim=0)
            fn = torch.sum(torch.stack(fn_list, dim=0), dim=0)
            tn = torch.sum(torch.stack(tn_list, dim=0), dim=0)
            return tp, fp, fn, tn
        
        return self.tp, self.fp, self.fn, self.tn
    
    def dice(self, average: bool = True) -> torch.Tensor:
        """
        Calculate the Jaccard score (intersection over union) for each class and the average Jaccard score.

        Parameters:
            average (bool): If True, returns the average Jaccard score, otherwise returns the Jaccard score for each class.

        Returns:
            torch.Tensor: A tensor containing either the average Jaccard score or the Jaccard score for each class.
        """ 
        tp, fp, fn, tn = self._sync()

        dice = 2 * tp / (2 * tp + fp + fn)
        if torch.isnan(dice).any():
            raise ValueError(f"Dice score contains NaN. Check your predictions!")
        if average:
            return torch.mean(dice)
        return dice

    def jaccard(self, average: bool = True):
        """
        Calculate the Jaccard score (intersection over union) for each class and the average Jaccard score.

        Parameters:
            average (bool): If True, returns the average Jaccard score, otherwise returns the Jaccard score for each class.

        Returns:
            torch.Tensor: A tensor containing either the average Jaccard score or the Jaccard score for each class.
        """ 
        tp, fp, fn, tn = self._sync()

        jaccard = tp / (tp + fp + fn)
        if torch.isnan(jaccard).any():
            raise ValueError(f"Intersections and/or unions is NaN! Check your predictions.")
        if average:
            return torch.mean(jaccard)
        return jaccard
        

    def accuracy(self, average: bool = True):
        """
        Calculate the accuracy for each class and the average accuracy.

        Parameters:
            average (bool): If True, returns the average accuracy, otherwise returns the accuracy for each class.

        Returns:
            torch.Tensor: A tensor containing either the average accuracy or the accuracy for each class.
        """ 
        tp, fp, fn, tn = self._sync()

        accuracy = (tp + tn) / (tp + fp + fn + tn)
        if torch.isnan(accuracy).any():
            raise ValueError(f"Intersections and/or unions is NaN! Check your predictions.")
        if average:
            return torch.mean(accuracy)
        return accuracy