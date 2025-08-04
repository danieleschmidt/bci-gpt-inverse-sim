"""Token-level EEG decoding for fine-grained thought analysis."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

from ..core.models import BCIGPTModel


@dataclass
class TokenPrediction:
    """Single token prediction with metadata."""
    token_id: int
    token_text: str
    probability: float
    logit: float
    confidence: float
    eeg_features: Optional[np.ndarray] = None


@dataclass
class SequencePrediction:
    """Sequence of token predictions."""
    tokens: List[TokenPrediction]
    full_text: str
    sequence_probability: float
    average_confidence: float
    processing_time: float


class TokenDecoder:
    """Token-level decoder for EEG-to-token mapping."""
    
    def __init__(self, 
                 model: BCIGPTModel,
                 device: str = "cuda"):
        """Initialize token decoder.
        
        Args:
            model: Trained BCI-GPT model
            device: Device for inference
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Get tokenizer if available
        self.tokenizer = getattr(model, 'tokenizer', None)
        if not self.tokenizer:
            warnings.warn("No tokenizer available in model")
            
    def decode_single_token(self, 
                          eeg_data: torch.Tensor,
                          return_top_k: int = 10) -> List[TokenPrediction]:
        """Decode a single token from EEG data.
        
        Args:
            eeg_data: EEG data tensor (batch_size, channels, sequence_length)
            return_top_k: Number of top predictions to return
            
        Returns:
            List of top-k token predictions
        """
        with torch.no_grad():
            # Forward pass
            outputs = self.model(eeg_data)
            logits = outputs['logits']
            
            # Handle different logit shapes
            if logits.dim() == 3:
                # Sequence of logits, take the last one
                token_logits = logits[0, -1, :]  # (vocab_size,)
            else:
                token_logits = logits[0]  # (vocab_size,)
            
            # Get probabilities
            probs = F.softmax(token_logits, dim=0)
            
            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(probs, return_top_k)
            
            predictions = []
            for i in range(return_top_k):
                token_id = top_k_indices[i].item()
                prob = top_k_probs[i].item()
                logit = token_logits[token_id].item()
                
                # Decode token text
                if self.tokenizer:
                    try:
                        token_text = self.tokenizer.decode([token_id])
                    except:
                        token_text = f"<token_{token_id}>"
                else:
                    token_text = f"<token_{token_id}>"
                
                # Calculate confidence (entropy-based)
                confidence = self._calculate_token_confidence(probs, token_id)
                
                # Extract EEG features if available
                eeg_features = None
                if 'eeg_features' in outputs:
                    eeg_features = outputs['eeg_features'][0].cpu().numpy()
                
                predictions.append(TokenPrediction(
                    token_id=token_id,
                    token_text=token_text,
                    probability=prob,
                    logit=logit,
                    confidence=confidence,
                    eeg_features=eeg_features
                ))
                
        return predictions
    
    def decode_sequence(self,
                       eeg_data: torch.Tensor,
                       max_length: int = 20,
                       beam_size: int = 3,
                       temperature: float = 1.0) -> SequencePrediction:
        """Decode a sequence of tokens using beam search.
        
        Args:
            eeg_data: EEG data tensor (batch_size, channels, sequence_length)
            max_length: Maximum sequence length
            beam_size: Beam search width
            temperature: Sampling temperature
            
        Returns:
            Best sequence prediction
        """
        import time
        start_time = time.time()
        
        if not self.tokenizer:
            # Simple single token decoding if no tokenizer
            single_pred = self.decode_single_token(eeg_data, return_top_k=1)[0]
            return SequencePrediction(
                tokens=[single_pred],
                full_text=single_pred.token_text,
                sequence_probability=single_pred.probability,
                average_confidence=single_pred.confidence,
                processing_time=time.time() - start_time
            )
        
        # Initialize beam search
        beams = [BeamSearchState(
            token_ids=[self.tokenizer.bos_token_id or self.tokenizer.eos_token_id],
            log_prob=0.0,
            tokens=[]
        )]
        
        completed_sequences = []
        
        for step in range(max_length):
            new_beams = []
            
            for beam in beams:
                if len(beam.token_ids) > 1 and beam.token_ids[-1] == self.tokenizer.eos_token_id:
                    completed_sequences.append(beam)
                    continue
                
                # Prepare input for model
                input_ids = torch.tensor([beam.token_ids], device=self.device)
                
                with torch.no_grad():
                    # Forward pass with both EEG and text
                    outputs = self.model(eeg_data, input_ids)
                    logits = outputs['logits'][0, -1, :]  # Last token logits
                    
                    # Apply temperature
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=0)
                    
                    # Get top beam_size candidates
                    top_probs, top_indices = torch.topk(probs, beam_size)
                    
                    for i in range(beam_size):
                        token_id = top_indices[i].item()
                        prob = top_probs[i].item()
                        log_prob = beam.log_prob + torch.log(top_probs[i]).item()
                        
                        # Create token prediction
                        if self.tokenizer:
                            token_text = self.tokenizer.decode([token_id])
                        else:
                            token_text = f"<token_{token_id}>"
                            
                        confidence = self._calculate_token_confidence(probs, token_id)
                        
                        token_pred = TokenPrediction(
                            token_id=token_id,
                            token_text=token_text,
                            probability=prob,
                            logit=logits[token_id].item(),
                            confidence=confidence
                        )
                        
                        new_beam = BeamSearchState(
                            token_ids=beam.token_ids + [token_id],
                            log_prob=log_prob,
                            tokens=beam.tokens + [token_pred]
                        )
                        new_beams.append(new_beam)
            
            # Keep top beam_size beams
            beams = sorted(new_beams, key=lambda x: x.log_prob, reverse=True)[:beam_size]
            
            # Early stopping if all beams completed
            if not beams:
                break
        
        # Add remaining beams to completed sequences
        completed_sequences.extend(beams)
        
        if not completed_sequences:
            # Fallback to single token
            single_pred = self.decode_single_token(eeg_data, return_top_k=1)[0]
            return SequencePrediction(
                tokens=[single_pred],
                full_text=single_pred.token_text,
                sequence_probability=single_pred.probability,
                average_confidence=single_pred.confidence,
                processing_time=time.time() - start_time
            )
        
        # Select best sequence
        best_beam = max(completed_sequences, key=lambda x: x.log_prob / len(x.tokens))
        
        # Generate full text
        if self.tokenizer and best_beam.tokens:
            full_text = self.tokenizer.decode([t.token_id for t in best_beam.tokens])
        else:
            full_text = " ".join([t.token_text for t in best_beam.tokens])
        
        # Calculate metrics
        sequence_prob = np.exp(best_beam.log_prob)
        avg_confidence = np.mean([t.confidence for t in best_beam.tokens]) if best_beam.tokens else 0.0
        
        return SequencePrediction(
            tokens=best_beam.tokens,
            full_text=full_text,
            sequence_probability=sequence_prob,
            average_confidence=avg_confidence,
            processing_time=time.time() - start_time
        )
    
    def analyze_token_attention(self,
                               eeg_data: torch.Tensor,
                               token_ids: List[int]) -> Dict[str, np.ndarray]:
        """Analyze attention patterns between EEG and tokens.
        
        Args:
            eeg_data: EEG data tensor (batch_size, channels, sequence_length)
            token_ids: List of token IDs to analyze
            
        Returns:
            Dictionary containing attention analysis
        """
        if not token_ids:
            return {}
            
        input_ids = torch.tensor([token_ids], device=self.device)
        
        with torch.no_grad():
            outputs = self.model(eeg_data, input_ids)
            
            analysis = {}
            
            # EEG features attention
            if 'eeg_features' in outputs:
                eeg_features = outputs['eeg_features'][0].cpu().numpy()  # (seq_len, hidden_dim)
                analysis['eeg_attention'] = np.mean(np.abs(eeg_features), axis=1)  # Average across features
            
            # Token-level analysis
            if 'fused_features' in outputs:
                fused_features = outputs['fused_features'][0].cpu().numpy()  # (text_seq_len, hidden_dim)
                analysis['token_attention'] = np.mean(np.abs(fused_features), axis=1)
            
            # Logit analysis
            logits = outputs['logits'][0].cpu().numpy()  # (text_seq_len, vocab_size)
            analysis['token_confidences'] = np.max(F.softmax(torch.from_numpy(logits), dim=-1), dim=-1)[0].numpy()
            
        return analysis
    
    def get_token_embeddings(self,
                           eeg_data: torch.Tensor,
                           token_ids: List[int]) -> np.ndarray:
        """Extract token embeddings from EEG-conditioned model.
        
        Args:
            eeg_data: EEG data tensor (batch_size, channels, sequence_length)
            token_ids: List of token IDs
            
        Returns:
            Token embeddings array (n_tokens, embedding_dim)
        """
        if not token_ids:
            return np.array([])
            
        input_ids = torch.tensor([token_ids], device=self.device)
        
        with torch.no_grad():
            outputs = self.model(eeg_data, input_ids)
            
            if 'fused_features' in outputs:
                embeddings = outputs['fused_features'][0].cpu().numpy()  # (seq_len, hidden_dim)
            elif 'text_features' in outputs:
                embeddings = outputs['text_features'][0].cpu().numpy()
            else:
                # Fallback to EEG features
                eeg_features = outputs['eeg_features'][0].cpu().numpy()
                # Average pool to match token sequence length
                if len(token_ids) > 1:
                    embeddings = np.repeat(
                        np.mean(eeg_features, axis=0, keepdims=True), 
                        len(token_ids), 
                        axis=0
                    )
                else:
                    embeddings = np.mean(eeg_features, axis=0, keepdims=True)
        
        return embeddings
    
    def _calculate_token_confidence(self, 
                                   probs: torch.Tensor, 
                                   token_id: int) -> float:
        """Calculate confidence score for a token prediction.
        
        Args:
            probs: Probability distribution over vocabulary
            token_id: Predicted token ID
            
        Returns:
            Confidence score between 0 and 1
        """
        # Confidence based on probability and entropy
        token_prob = probs[token_id].item()
        
        # Calculate entropy (uncertainty)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy
        
        # Confidence combines high probability and low entropy
        confidence = token_prob * (1 - normalized_entropy)
        
        return confidence
    
    def compare_predictions(self,
                          pred1: SequencePrediction,
                          pred2: SequencePrediction) -> Dict[str, Any]:
        """Compare two sequence predictions.
        
        Args:
            pred1: First prediction
            pred2: Second prediction
            
        Returns:
            Comparison metrics
        """
        comparison = {
            'length_diff': len(pred1.tokens) - len(pred2.tokens),
            'prob_diff': pred1.sequence_probability - pred2.sequence_probability,
            'confidence_diff': pred1.average_confidence - pred2.average_confidence,
            'text_similarity': self._calculate_text_similarity(pred1.full_text, pred2.full_text),
            'token_overlap': self._calculate_token_overlap(pred1.tokens, pred2.tokens)
        }
        
        return comparison
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two strings."""
        # Simple character-based similarity
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
            
        # Jaccard similarity on character level
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_token_overlap(self, 
                               tokens1: List[TokenPrediction],
                               tokens2: List[TokenPrediction]) -> float:
        """Calculate token overlap between two predictions."""
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
            
        ids1 = set(t.token_id for t in tokens1)
        ids2 = set(t.token_id for t in tokens2)
        
        intersection = len(ids1.intersection(ids2))
        union = len(ids1.union(ids2))
        
        return intersection / union if union > 0 else 0.0


@dataclass
class BeamSearchState:
    """State for beam search decoding."""
    token_ids: List[int]
    log_prob: float
    tokens: List[TokenPrediction]