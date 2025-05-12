# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.
    Paper: https://arxiv.org/abs/1708.02002
    Implementation based on various sources.
    Assumes inputs are logits (raw scores from the model).
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C, *) logits
        # targets: (N, C, *) ground truth labels (0 or 1)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # pt is the probability of the ground truth class for each example
        # For y=1, pt = sigmoid(inputs). For y=0, pt = 1 - sigmoid(inputs)
        # This can be written as exp(-BCE_loss)
        pt = torch.exp(-BCE_loss)

        # Calculate Focal Loss
        # For y=1, loss = -alpha * (1-pt)^gamma * log(pt)
        # For y=0, loss = -(1-alpha) * (1-pt)^gamma * log(pt)
        # This can be simplified if we use alpha_t
        # alpha_t is alpha for positive class, (1-alpha) for negative class
        alpha_t = torch.where(targets == 1, torch.tensor(self.alpha, device=inputs.device),
                              torch.tensor(1 - self.alpha, device=inputs.device))

        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:  # 'none'
            return F_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for image segmentation.
    Assumes inputs are probabilities (after sigmoid/softmax).
    """

    def __init__(self, smooth=1e-6, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction  # 'mean' or 'none'

    def forward(self, inputs_probs, targets):
        # inputs_probs: (N, C, *) probabilities (e.g., after sigmoid for binary)
        # targets: (N, C, *) ground truth labels (0 or 1)

        # Flatten inputs and targets to (N*C*...,)
        inputs_probs = inputs_probs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs_probs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (inputs_probs.sum() + targets.sum() + self.smooth)

        loss = 1 - dice_coeff

        # Note: Dice loss is usually per-image or per-batch.
        # If reduction is 'mean' and we have multiple channels/classes,
        # this simple implementation averages over all pixels.
        # For multi-class, often Dice is calculated per class and then averaged.
        # For binary, this is fine.

        return loss  # Already a scalar if sum is used. Reduction param is for consistency.


class CombinedLoss(nn.Module):
    """
    Combines Focal Loss (or BCEWithLogitsLoss) and Dice Loss.
    """

    def __init__(self,
                 loss1_type='FocalLoss',  # 'FocalLoss' or 'BCEWithLogitsLoss'
                 loss1_weight=0.5,
                 loss1_params=None,  # Dict of params for FocalLoss or BCEWithLogitsLoss
                 dice_weight=0.5,
                 dice_params=None,  # Dict of params for DiceLoss (e.g., smooth)
                 model_outputs_logits=True):  # Crucial: does the model output logits or probabilities?
        super(CombinedLoss, self).__init__()

        if loss1_params is None: loss1_params = {}
        if dice_params is None: dice_params = {}

        self.loss1_type = loss1_type
        self.loss1_weight = loss1_weight
        self.dice_weight = dice_weight
        self.model_outputs_logits = model_outputs_logits

        if self.loss1_type == 'FocalLoss':
            if not self.model_outputs_logits:
                print(
                    "Warning: FocalLoss in CombinedLoss expects logits, but model_outputs_logits is False. Ensure input to FocalLoss is logits.")
            self.loss1 = FocalLoss(**loss1_params)
        elif self.loss1_type == 'BCEWithLogitsLoss':
            if not self.model_outputs_logits:
                print("Warning: BCEWithLogitsLoss in CombinedLoss expects logits, but model_outputs_logits is False.")
            self.loss1 = nn.BCEWithLogitsLoss(**loss1_params)  # reduction handled by Focal if used
        elif self.loss1_type == 'BCELoss':  # If model outputs probabilities for loss1
            if self.model_outputs_logits:
                print("Warning: BCELoss in CombinedLoss expects probabilities, but model_outputs_logits is True.")
            self.loss1 = nn.BCELoss(**loss1_params)
        else:
            raise ValueError(f"Unsupported loss1_type: {self.loss1_type}")

        self.dice_loss = DiceLoss(**dice_params)
        print(
            f"CombinedLoss: Using {self.loss1_type} (weight {self.loss1_weight}) and DiceLoss (weight {self.dice_weight})")

    def forward(self, inputs, targets):
        # inputs: raw output from the model
        # targets: ground truth labels

        loss1_val = 0
        dice_input_probs = None

        if self.model_outputs_logits:
            # Loss1 (Focal or BCEWithLogits) takes logits
            loss1_val = self.loss1(inputs, targets)
            # Dice loss needs probabilities
            dice_input_probs = torch.sigmoid(inputs)
        else:  # Model outputs probabilities
            # Loss1 (BCELoss) takes probabilities
            loss1_val = self.loss1(inputs, targets)
            # Dice loss also needs probabilities
            dice_input_probs = inputs

        dice_loss_val = self.dice_loss(dice_input_probs, targets)

        total_loss = self.loss1_weight * loss1_val + self.dice_weight * dice_loss_val

        return total_loss


if __name__ == '__main__':
    # Example Usage and Testing
    print("--- Testing Loss Functions ---")
    N, C, L = 4, 1, 256  # Batch, Channels, Length

    # Test with logits
    print("\n--- Testing with Logits Input ---")
    dummy_logits = torch.randn(N, C, L, requires_grad=True)
    dummy_targets = torch.randint(0, 2, (N, C, L)).float()

    # Focal Loss
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    f_loss = focal_loss_fn(dummy_logits, dummy_targets)
    print(f"Focal Loss (logits): {f_loss.item()}")
    f_loss.backward(retain_graph=True)  # Retain graph if dummy_logits is used by other losses

    # Dice Loss (needs probs from logits)
    dice_loss_fn = DiceLoss(smooth=1e-5)
    dummy_probs_from_logits = torch.sigmoid(dummy_logits)
    d_loss_from_logits = dice_loss_fn(dummy_probs_from_logits, dummy_targets)
    print(f"Dice Loss (from logits via sigmoid): {d_loss_from_logits.item()}")
    # d_loss_from_logits.backward() # Can't backward twice on same leaf without retain_graph

    # Combined Loss (Focal + Dice, model outputs logits)
    combined_loss_fn_focal_dice = CombinedLoss(
        loss1_type='FocalLoss', loss1_weight=0.5, loss1_params={'alpha': 0.25, 'gamma': 2.0},
        dice_weight=0.5, dice_params={'smooth': 1e-5},
        model_outputs_logits=True
    )
    c_loss_fd = combined_loss_fn_focal_dice(dummy_logits.clone().detach().requires_grad_(True),
                                            dummy_targets)  # Use clone for separate backward
    print(f"Combined (Focal+Dice, logits): {c_loss_fd.item()}")
    c_loss_fd.backward()

    # Combined Loss (BCEWithLogits + Dice, model outputs logits)
    combined_loss_fn_bce_dice = CombinedLoss(
        loss1_type='BCEWithLogitsLoss', loss1_weight=0.7,
        dice_weight=0.3, dice_params={'smooth': 1e-5},
        model_outputs_logits=True
    )
    c_loss_bd = combined_loss_fn_bce_dice(dummy_logits.clone().detach().requires_grad_(True), dummy_targets)
    print(f"Combined (BCEWithLogits+Dice, logits): {c_loss_bd.item()}")
    c_loss_bd.backward()

    # Test with probabilities
    print("\n--- Testing with Probabilities Input ---")
    dummy_probs = torch.rand(N, C, L, requires_grad=True)  # Already probabilities

    # Dice Loss
    d_loss_probs = dice_loss_fn(dummy_probs, dummy_targets)
    print(f"Dice Loss (probs): {d_loss_probs.item()}")
    d_loss_probs.backward(retain_graph=True)

    # Combined Loss (BCELoss + Dice, model outputs probabilities)
    combined_loss_fn_bce_dice_probs = CombinedLoss(
        loss1_type='BCELoss', loss1_weight=0.6,
        dice_weight=0.4, dice_params={'smooth': 1e-5},
        model_outputs_logits=False  # Important!
    )
    c_loss_bd_p = combined_loss_fn_bce_dice_probs(dummy_probs.clone().detach().requires_grad_(True), dummy_targets)
    print(f"Combined (BCELoss+Dice, probs): {c_loss_bd_p.item()}")
    c_loss_bd_p.backward()

    print("Loss function tests complete.")


# In losses.py
class FocalBCELoss(nn.Module):
    def __init__(self,
                 focal_weight=0.5,
                 focal_params=None,  # alpha, gamma
                 bce_weight=0.5,
                 bce_loss_type='BCEWithLogitsLoss',  # 'BCEWithLogitsLoss' or 'BCELoss'
                 bce_params=None,
                 # This flag is crucial. It indicates if the 'inputs' to forward() are logits.
                 # FocalLoss (current version) always needs logits.
                 # BCEWithLogitsLoss needs logits. BCELoss needs probs.
                 inputs_are_logits=True):
        super(FocalBCELoss, self).__init__()

        if focal_params is None: focal_params = {}
        if bce_params is None: bce_params = {}  # e.g. reduction for BCE, pos_weight for BCEWithLogits

        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.bce_loss_type = bce_loss_type
        self.inputs_are_logits = inputs_are_logits  # This should match model's output nature

        if not self.inputs_are_logits and (self.focal_weight > 0 or self.bce_loss_type == 'BCEWithLogitsLoss'):
            # This is a problematic configuration.
            # Current FocalLoss expects logits. BCEWithLogitsLoss expects logits.
            # If inputs are not logits, these losses will not work as intended.
            raise ValueError(
                "FocalBCELoss configuration error: "
                "If inputs_are_logits is False (model outputs probabilities), "
                "then FocalLoss cannot be used (with current implementation) "
                "and bce_loss_type cannot be 'BCEWithLogitsLoss'."
            )

        if self.focal_weight > 0:
            self.focal_loss_fn = FocalLoss(**focal_params)
        else:
            self.focal_loss_fn = None

        if self.bce_weight > 0:
            if self.bce_loss_type == 'BCEWithLogitsLoss':
                self.bce_loss_component_fn = nn.BCEWithLogitsLoss(**bce_params)
            elif self.bce_loss_type == 'BCELoss':
                self.bce_loss_component_fn = nn.BCELoss(**bce_params)
            else:
                raise ValueError(f"Unsupported bce_loss_type: {self.bce_loss_type}")
        else:
            self.bce_loss_component_fn = None

        print(
            f"FocalBCELoss initialized: Focal_w={self.focal_weight}, BCE_w={self.bce_weight} ({self.bce_loss_type}). Inputs are logits: {self.inputs_are_logits}")

    def forward(self, inputs, targets):
        # inputs: raw output from the model
        # targets: ground truth labels

        total_loss = 0.0
        loss_focal_val = 0.0
        loss_bce_val = 0.0

        # --- Calculate Focal Loss ---
        if self.focal_loss_fn and self.focal_weight > 0:
            if not self.inputs_are_logits:
                # This state should have been caught by __init__, but defensive check.
                # This implies model outputs probabilities, but FocalLoss needs logits.
                # To make it work, one would need to apply inverse sigmoid to 'inputs', which is numerically unstable.
                # Or, FocalLoss needs to be adapted to take probabilities.
                # For now, this is an error condition.
                raise RuntimeError("FocalLoss requires logits, but inputs_are_logits is False.")
            loss_focal_val = self.focal_loss_fn(inputs, targets)
            total_loss += self.focal_weight * loss_focal_val

        # --- Calculate BCE Component Loss ---
        if self.bce_loss_component_fn and self.bce_weight > 0:
            if self.bce_loss_type == 'BCEWithLogitsLoss':
                if not self.inputs_are_logits:
                    raise RuntimeError("BCEWithLogitsLoss requires logits, but inputs_are_logits is False.")
                loss_bce_val = self.bce_loss_component_fn(inputs, targets)
            elif self.bce_loss_type == 'BCELoss':
                if self.inputs_are_logits:
                    # Convert logits to probabilities for BCELoss
                    probs = torch.sigmoid(inputs)
                    loss_bce_val = self.bce_loss_component_fn(probs, targets)
                else:
                    # Inputs are already probabilities
                    loss_bce_val = self.bce_loss_component_fn(inputs, targets)
            total_loss += self.bce_weight * loss_bce_val

        return total_loss