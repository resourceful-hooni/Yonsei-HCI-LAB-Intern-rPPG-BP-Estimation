"""
tcn_attention_model.py - TCN + Attention Architecture for BP Estimation
================================================================================
Author: Yonsei HCI LAB
Date: 2026-01-22
Version: 1.0.0

Architecture Design:
--------------------
Based on state-of-the-art research in physiological signal processing:
1. WaveNet-style TCN (Bai et al., 2018) - Dilated Causal Convolutions
2. Squeeze-and-Excitation (Hu et al., 2018) - Channel Attention
3. Multi-Head Self-Attention (Vaswani et al., 2017) - Global Context
4. Residual Connections (He et al., 2016) - Gradient Flow

Key Design Choices:
-------------------
- Dilated Causal Convolution: Captures long-range dependencies without increasing parameters
- Receptive Field: With dilation factors [1,2,4,8,16], RF = 2 * kernel_size * sum(dilations) = 2*3*31 = 186 samples per block
- 3 TCN blocks → Total RF = 558+ samples (4.5+ seconds at 125Hz)
- SE-Block: Channel-wise attention for feature selection
- Multi-Head Attention: Global temporal attention across entire sequence

Input: (batch_size, 875, 1) - 7 seconds @ 125Hz rPPG signal
Output: [SBP, DBP] - Blood pressure predictions (normalized)

References:
-----------
[1] Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", 2018
[2] Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
[3] Vaswani et al., "Attention Is All You Need", NeurIPS 2017
[4] Slapnicar et al., "Blood Pressure Estimation from PPG using Spectro-Temporal CNN", IEEE JBHI 2019
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization, 
    Activation, Add, Multiply, GlobalAveragePooling1D,
    LayerNormalization, Concatenate, Reshape, Permute
)
import numpy as np


# =============================================================================
# CUSTOM LAYERS
# =============================================================================

class CausalConv1D(layers.Layer):
    """
    Causal Convolution Layer
    Ensures no future information leakage - crucial for real-time inference
    
    Implements: y[t] = sum(w[k] * x[t-k]) for k=0 to kernel_size-1
    """
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(CausalConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = (kernel_size - 1) * dilation_rate
        
    def build(self, input_shape):
        self.conv = Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='causal',  # TensorFlow handles causal padding
            dilation_rate=self.dilation_rate,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-4)
        )
        super(CausalConv1D, self).build(input_shape)
        
    def call(self, inputs):
        return self.conv(inputs)
    
    def get_config(self):
        config = super(CausalConv1D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        })
        return config


class SqueezeExcitation(layers.Layer):
    """
    Squeeze-and-Excitation Block
    Channel-wise attention mechanism for feature recalibration
    
    Architecture:
    1. Global Average Pooling (Squeeze): (B, T, C) -> (B, C)
    2. FC -> ReLU -> FC -> Sigmoid (Excitation): (B, C) -> (B, C)
    3. Channel-wise multiplication: (B, T, C) * (B, 1, C) -> (B, T, C)
    
    Reduction ratio 16 is standard from SE-Net paper
    """
    def __init__(self, reduction_ratio=16, **kwargs):
        super(SqueezeExcitation, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(channels // self.reduction_ratio, 8)  # Min 8 channels
        
        self.global_pool = GlobalAveragePooling1D()
        self.fc1 = Dense(reduced_channels, activation='relu', 
                        kernel_initializer='he_normal')
        self.fc2 = Dense(channels, activation='sigmoid',
                        kernel_initializer='he_normal')
        self.reshape = Reshape((1, channels))
        
        super(SqueezeExcitation, self).build(input_shape)
        
    def call(self, inputs):
        # Squeeze: Global pooling
        x = self.global_pool(inputs)  # (B, C)
        
        # Excitation: Learn channel importance
        x = self.fc1(x)  # (B, C/r)
        x = self.fc2(x)  # (B, C)
        x = self.reshape(x)  # (B, 1, C)
        
        # Scale: Channel-wise multiplication
        return Multiply()([inputs, x])
    
    def get_config(self):
        config = super(SqueezeExcitation, self).get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config


class TCNResidualBlock(layers.Layer):
    """
    Temporal Convolutional Network Residual Block
    
    Architecture:
    Input ─┬─> CausalConv -> BN -> ReLU -> Dropout ─┐
           │                                        │
           │   CausalConv -> BN -> ReLU -> Dropout ─┤
           │                                        │
           │   SE-Block (optional) ─────────────────┤
           │                                        │
           └─> 1x1 Conv (if needed) ───────────────Add -> ReLU -> Output
    
    Dilation pattern: [1, 2, 4, 8, 16] for exponential receptive field growth
    """
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.1, 
                 use_se=True, **kwargs):
        super(TCNResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.use_se = use_se
        
    def build(self, input_shape):
        # First causal conv layer
        self.conv1 = CausalConv1D(self.filters, self.kernel_size, self.dilation_rate)
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(self.dropout_rate)
        
        # Second causal conv layer
        self.conv2 = CausalConv1D(self.filters, self.kernel_size, self.dilation_rate)
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(self.dropout_rate)
        
        # SE Block
        if self.use_se:
            self.se = SqueezeExcitation(reduction_ratio=16)
        
        # Residual connection (1x1 conv if channel mismatch)
        self.match_channels = input_shape[-1] != self.filters
        if self.match_channels:
            self.residual_conv = Conv1D(self.filters, 1, padding='same',
                                        kernel_initializer='he_normal')
            
        super(TCNResidualBlock, self).build(input_shape)
        
    def call(self, inputs, training=None):
        # First conv block
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = Activation('relu')(x)
        x = self.dropout1(x, training=training)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = Activation('relu')(x)
        x = self.dropout2(x, training=training)
        
        # SE attention
        if self.use_se:
            x = self.se(x)
        
        # Residual connection
        if self.match_channels:
            residual = self.residual_conv(inputs)
        else:
            residual = inputs
            
        x = Add()([x, residual])
        x = Activation('relu')(x)
        
        return x
    
    def get_config(self):
        config = super(TCNResidualBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate,
            'use_se': self.use_se
        })
        return config


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention Layer
    Captures global temporal dependencies across the entire sequence
    
    Architecture:
    Q, K, V = Linear projections of input
    Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
    Output = Concat(head_1, ..., head_h) @ W_o
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads
        
    def build(self, input_shape):
        self.wq = Dense(self.d_model, kernel_initializer='glorot_uniform')
        self.wk = Dense(self.d_model, kernel_initializer='glorot_uniform')
        self.wv = Dense(self.d_model, kernel_initializer='glorot_uniform')
        self.dense = Dense(self.d_model, kernel_initializer='glorot_uniform')
        self.dropout = Dropout(self.dropout_rate)
        
        super(MultiHeadSelfAttention, self).build(input_shape)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, heads, seq, depth)
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # Linear projections
        q = self.wq(inputs)  # (B, T, d_model)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        # Split into multiple heads
        q = self.split_heads(q, batch_size)  # (B, h, T, d_k)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attention_scores = tf.matmul(q, k, transpose_b=True) / scale  # (B, h, T, T)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention to values
        attention_output = tf.matmul(attention_weights, v)  # (B, h, T, d_k)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.dense(concat_attention)
        
        return output
    
    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config


class PositionalEncoding(layers.Layer):
    """
    Sinusoidal Positional Encoding
    Adds position information to the input embeddings
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, max_len=1000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        
    def build(self, input_shape):
        d_model = input_shape[-1]
        
        # Create positional encoding matrix
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((self.max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        
        super(PositionalEncoding, self).build(input_shape)
        
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({'max_len': self.max_len})
        return config


class AttentionBlock(layers.Layer):
    """
    Complete Attention Block with Layer Normalization and Feed-Forward Network
    
    Architecture:
    Input -> LayerNorm -> MultiHeadAttention -> Dropout -> Add -> 
          -> LayerNorm -> FFN -> Dropout -> Add -> Output
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.mha = MultiHeadSelfAttention(self.d_model, self.num_heads, self.dropout_rate)
        self.ffn1 = Dense(self.dff, activation='relu', kernel_initializer='he_normal')
        self.ffn2 = Dense(self.d_model, kernel_initializer='he_normal')
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)
        
        super(AttentionBlock, self).build(input_shape)
        
    def call(self, inputs, training=None):
        # Multi-head attention with residual
        attn_output = self.mha(self.layernorm1(inputs), training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = Add()([inputs, attn_output])
        
        # Feed-forward with residual
        ffn_output = self.ffn1(self.layernorm2(out1))
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = Add()([out1, ffn_output])
        
        return out2
    
    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config


# =============================================================================
# MAIN MODEL ARCHITECTURE
# =============================================================================

def create_tcn_attention_model(
    input_shape=(875, 1),
    # TCN Parameters
    tcn_filters=[64, 128, 128],       # Filters per TCN stack
    tcn_kernel_size=3,                 # Kernel size for dilated convolutions
    tcn_dilations=[1, 2, 4, 8, 16],   # Dilation factors per stack
    tcn_dropout=0.1,                   # Dropout in TCN blocks
    use_se=True,                       # Use SE-attention in TCN
    # Attention Parameters
    attention_heads=4,                 # Number of attention heads
    attention_dim=128,                 # Attention model dimension
    attention_dff=256,                 # Feed-forward dimension
    attention_dropout=0.1,             # Attention dropout
    num_attention_layers=2,            # Number of attention blocks
    # Dense Parameters
    dense_units=[128, 64],             # Dense layer units
    dense_dropout=0.2,                 # Dense layer dropout
    # Output
    output_activation=None,            # Activation for output (None for regression)
    name='TCN_Attention_BP'
):
    """
    Create TCN + Attention Model for Blood Pressure Estimation
    
    Architecture Overview:
    ----------------------
    Input (875, 1)
        │
        ▼
    ┌─────────────────────────────────────────────┐
    │  TCN ENCODER                                │
    │  ├─ TCN Stack 1 (64 filters, dilations 1-16)│
    │  ├─ TCN Stack 2 (128 filters, dilations 1-16)│
    │  └─ TCN Stack 3 (128 filters, dilations 1-16)│
    │  Each stack: 5 residual blocks + SE attention│
    └─────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────┐
    │  GLOBAL ATTENTION                           │
    │  ├─ Dense projection to attention_dim       │
    │  ├─ Positional Encoding                     │
    │  ├─ Multi-Head Self-Attention × 2           │
    │  └─ Global Average Pooling                  │
    └─────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────┐
    │  REGRESSION HEAD                            │
    │  ├─ Dense(128) → ReLU → Dropout             │
    │  ├─ Dense(64) → ReLU → Dropout              │
    │  └─ Output: [SBP, DBP]                      │
    └─────────────────────────────────────────────┘
    
    Parameters:
    -----------
    input_shape : tuple
        Input shape (sequence_length, channels)
    tcn_filters : list
        Number of filters for each TCN stack
    tcn_kernel_size : int
        Kernel size for TCN convolutions
    tcn_dilations : list
        Dilation factors within each TCN stack
    use_se : bool
        Whether to use Squeeze-Excitation in TCN blocks
    attention_heads : int
        Number of attention heads
    attention_dim : int
        Dimension of attention layers
    num_attention_layers : int
        Number of stacked attention blocks
    dense_units : list
        Units for dense layers before output
        
    Returns:
    --------
    model : tf.keras.Model
        Compiled TCN-Attention model
    """
    
    # Input layer
    inputs = Input(shape=input_shape, name='rppg_input')
    x = inputs
    
    # =========================================================================
    # TCN ENCODER
    # =========================================================================
    # Each TCN stack processes the signal at different scales
    # Dilations grow exponentially: 1, 2, 4, 8, 16
    # Total receptive field per stack = kernel_size * sum(dilations) * 2
    # = 3 * (1+2+4+8+16) * 2 = 186 samples ≈ 1.5 seconds
    # 3 stacks = ~4.5+ seconds coverage
    
    for stack_idx, filters in enumerate(tcn_filters):
        for dilation in tcn_dilations:
            x = TCNResidualBlock(
                filters=filters,
                kernel_size=tcn_kernel_size,
                dilation_rate=dilation,
                dropout_rate=tcn_dropout,
                use_se=use_se,
                name=f'tcn_stack{stack_idx+1}_d{dilation}'
            )(x)
    
    # =========================================================================
    # ATTENTION ENCODER
    # =========================================================================
    # Project to attention dimension
    x = Dense(attention_dim, activation='relu', name='attention_projection')(x)
    
    # Add positional encoding
    x = PositionalEncoding(max_len=input_shape[0], name='positional_encoding')(x)
    x = Dropout(attention_dropout, name='pos_dropout')(x)
    
    # Stack attention blocks
    for i in range(num_attention_layers):
        x = AttentionBlock(
            d_model=attention_dim,
            num_heads=attention_heads,
            dff=attention_dff,
            dropout_rate=attention_dropout,
            name=f'attention_block_{i+1}'
        )(x)
    
    # Global pooling
    x = GlobalAveragePooling1D(name='global_pool')(x)
    
    # =========================================================================
    # REGRESSION HEAD
    # =========================================================================
    for i, units in enumerate(dense_units):
        x = Dense(units, activation='relu', name=f'dense_{i+1}',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(1e-4))(x)
        x = Dropout(dense_dropout, name=f'dense_dropout_{i+1}')(x)
    
    # Output layers - separate heads for SBP and DBP
    sbp_output = Dense(1, activation=output_activation, name='sbp_output',
                      kernel_initializer='glorot_normal')(x)
    dbp_output = Dense(1, activation=output_activation, name='dbp_output',
                      kernel_initializer='glorot_normal')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=[sbp_output, dbp_output], name=name)
    
    return model


def compile_tcn_attention_model(model, learning_rate=0.001, loss_weights=None):
    """
    Compile TCN-Attention model with optimized settings
    
    Loss Function: Huber Loss (robust to outliers)
    Optimizer: AdamW (Adam with decoupled weight decay)
    
    Parameters:
    -----------
    model : tf.keras.Model
        Model to compile
    learning_rate : float
        Initial learning rate
    loss_weights : dict
        Optional weights for SBP/DBP losses
    """
    if loss_weights is None:
        loss_weights = {'sbp_output': 1.0, 'dbp_output': 1.0}
    
    # Use Huber loss for robustness to outliers
    # delta=1.0 means MSE for errors < 1, MAE for errors > 1
    huber_loss = tf.keras.losses.Huber(delta=1.0)
    
    # AdamW optimizer - better generalization than vanilla Adam
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss={
            'sbp_output': huber_loss,
            'dbp_output': huber_loss
        },
        loss_weights=loss_weights,
        metrics={
            'sbp_output': ['mae', 'mse'],
            'dbp_output': ['mae', 'mse']
        }
    )
    
    return model


def get_model_summary(model):
    """
    Get detailed model summary including receptive field calculation
    """
    print("\n" + "="*80)
    print("TCN + Attention Model Architecture Summary")
    print("="*80)
    
    # Model summary
    model.summary()
    
    # Calculate receptive field
    kernel_size = 3
    dilations = [1, 2, 4, 8, 16]
    num_stacks = 3
    
    rf_per_stack = kernel_size * sum(dilations) * 2
    total_rf = rf_per_stack * num_stacks
    
    print("\n" + "-"*80)
    print("Receptive Field Analysis:")
    print("-"*80)
    print(f"Kernel Size: {kernel_size}")
    print(f"Dilations per stack: {dilations}")
    print(f"Number of stacks: {num_stacks}")
    print(f"RF per stack: {rf_per_stack} samples")
    print(f"Approx. total RF: {total_rf}+ samples")
    print(f"At 125Hz: {total_rf/125:.2f}+ seconds coverage")
    print("-"*80)
    
    # Parameter count
    trainable = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    
    print(f"\nParameter Count:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Non-trainable: {non_trainable:,}")
    print(f"  Total: {trainable + non_trainable:,}")
    print("="*80 + "\n")


# =============================================================================
# CUSTOM OBJECTS FOR MODEL LOADING
# =============================================================================

def get_custom_objects():
    """Return custom objects dict for model loading"""
    return {
        'CausalConv1D': CausalConv1D,
        'SqueezeExcitation': SqueezeExcitation,
        'TCNResidualBlock': TCNResidualBlock,
        'MultiHeadSelfAttention': MultiHeadSelfAttention,
        'PositionalEncoding': PositionalEncoding,
        'AttentionBlock': AttentionBlock
    }


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_tcn_attention_model():
    """Test model creation and forward pass"""
    print("\n" + "="*80)
    print("Testing TCN + Attention Model")
    print("="*80)
    
    # Create model
    model = create_tcn_attention_model(
        input_shape=(875, 1),
        tcn_filters=[64, 128, 128],
        tcn_kernel_size=3,
        tcn_dilations=[1, 2, 4, 8, 16],
        attention_heads=4,
        attention_dim=128,
        num_attention_layers=2
    )
    
    # Compile
    model = compile_tcn_attention_model(model, learning_rate=0.001)
    
    # Summary
    get_model_summary(model)
    
    # Test forward pass
    print("Testing forward pass...")
    test_input = np.random.randn(4, 875, 1).astype(np.float32)
    outputs = model.predict(test_input, verbose=0)
    
    print(f"Input shape: {test_input.shape}")
    print(f"SBP output shape: {outputs[0].shape}")
    print(f"DBP output shape: {outputs[1].shape}")
    print(f"SBP values (sample): {outputs[0].flatten()}")
    print(f"DBP values (sample): {outputs[1].flatten()}")
    
    print("\n[OK] Model test passed!")
    print("="*80 + "\n")
    
    return model


if __name__ == "__main__":
    test_tcn_attention_model()
