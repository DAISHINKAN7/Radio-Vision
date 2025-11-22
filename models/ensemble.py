"""
Ensemble Model - Combines predictions from multiple models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleClassifier(nn.Module):
    """
    Ensemble multiple models by averaging their logits
    Can use uniform weights or learned weights
    """
    def __init__(self, models, weights=None, learnable_weights=False):
        """
        Args:
            models: List of model instances
            weights: Optional list of weights for each model (default: uniform)
            learnable_weights: If True, learn optimal weights during training
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

        if weights is None:
            weights = [1.0 / self.num_models] * self.num_models

        if learnable_weights:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

        self.learnable_weights = learnable_weights

    def forward(self, *inputs):
        """
        Forward pass through all models and combine predictions

        If models expect different inputs (e.g., multimodal), pass them as separate args
        Otherwise, pass single input for all models
        """
        # Normalize weights to sum to 1
        if self.learnable_weights:
            normalized_weights = F.softmax(self.weights, dim=0)
        else:
            normalized_weights = self.weights / self.weights.sum()

        # Get predictions from all models
        predictions = []
        for i, model in enumerate(self.models):
            if len(inputs) > 1:
                # Multimodal models might need multiple inputs
                pred = model(*inputs)
            else:
                pred = model(inputs[0])
            predictions.append(pred)

        # Weighted average of logits
        ensemble_logits = sum(w * p for w, p in zip(normalized_weights, predictions))
        return ensemble_logits

    def get_weights(self):
        """Return current ensemble weights"""
        if self.learnable_weights:
            return F.softmax(self.weights, dim=0).detach().cpu().numpy()
        else:
            return (self.weights / self.weights.sum()).cpu().numpy()


class WeightedEnsemble(nn.Module):
    """
    Learnable weighted ensemble with small network to combine predictions
    """
    def __init__(self, models, num_classes):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

        # Small network to learn combination weights
        self.combiner = nn.Sequential(
            nn.Linear(num_classes * self.num_models, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, *inputs):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            if len(inputs) > 1:
                pred = model(*inputs)
            else:
                pred = model(inputs[0])
            predictions.append(pred)

        # Concatenate all predictions
        combined = torch.cat(predictions, dim=1)

        # Learn combination
        return self.combiner(combined)


class VotingEnsemble:
    """
    Simple voting ensemble (for inference only, not trainable)
    """
    def __init__(self, models, voting='soft'):
        """
        Args:
            models: List of trained models
            voting: 'soft' for probability averaging, 'hard' for majority vote
        """
        self.models = models
        self.voting = voting

    @torch.no_grad()
    def predict(self, *inputs):
        """Get ensemble prediction"""
        predictions = []
        for model in self.models:
            model.eval()
            if len(inputs) > 1:
                pred = model(*inputs)
            else:
                pred = model(inputs[0])

            if self.voting == 'soft':
                pred = F.softmax(pred, dim=1)
            else:
                pred = pred.argmax(dim=1, keepdim=True)

            predictions.append(pred)

        if self.voting == 'soft':
            # Average probabilities
            avg_pred = torch.stack(predictions).mean(dim=0)
            return avg_pred.argmax(dim=1)
        else:
            # Majority vote
            stacked = torch.cat(predictions, dim=1)
            votes, _ = torch.mode(stacked, dim=1)
            return votes

    @torch.no_grad()
    def predict_proba(self, *inputs):
        """Get probability predictions"""
        predictions = []
        for model in self.models:
            model.eval()
            if len(inputs) > 1:
                pred = model(*inputs)
            else:
                pred = model(inputs[0])
            predictions.append(F.softmax(pred, dim=1))

        return torch.stack(predictions).mean(dim=0)
