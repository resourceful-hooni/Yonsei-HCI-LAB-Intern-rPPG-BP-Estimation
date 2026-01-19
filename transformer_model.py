"""
transformer_model.py - Transformer Architecture for BP Estimation
Phase 4: Transformer-based rPPG to BP model
"""

import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import layers, Model
import numpy as np


def positional_encoding(length, depth):
    """Generate positional encoding for Transformer"""
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )
    
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)


class MultiHeadAttention(layers.Layer):
    """Multi-Head Attention layer"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )
        
        output = self.dense(concat_attention)
        return output, attention_weights


def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate attention scores"""
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """Feed Forward Network"""
    return ks.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])


class EncoderLayer(layers.Layer):
    """Transformer Encoder Layer"""
    
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, x, training=False):
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TransformerEncoder(layers.Layer):
    """Transformer Encoder"""
    
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = layers.Dropout(rate)
    
    def call(self, x, training=False):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)
        
        return x


def create_transformer_model(input_shape=(875, 1), d_model=128, num_heads=4, 
                             num_layers=3, dff=256, dropout_rate=0.1):
    """
    Create Transformer model for BP estimation
    
    Args:
        input_shape: Input shape (seq_length, features)
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dff: Feed-forward dimension
        dropout_rate: Dropout rate
    
    Returns:
        model: Keras model
    """
    print(f"\n[*] Creating Transformer model...")
    
    inputs = layers.Input(shape=input_shape, name='input_signal')
    
    # Embedding layer
    x = layers.Dense(d_model, name='embedding')(inputs)
    
    # Add positional encoding
    seq_len = input_shape[0]
    pos_enc = positional_encoding(seq_len, d_model)
    x = x + pos_enc[:, :seq_len, :]
    x = layers.Dropout(dropout_rate)(x)
    
    # Transformer encoder
    encoder = TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        rate=dropout_rate
    )
    x = encoder(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layers (BP only for now)
    sbp_output = layers.Dense(1, name='sbp_output')(x)
    dbp_output = layers.Dense(1, name='dbp_output')(x)
    
    model = Model(inputs=inputs, outputs=[sbp_output, dbp_output])
    
    print(f"   [OK] Model created")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shapes: {[o.shape for o in model.outputs]}")
    print(f"   Total params: {model.count_params():,}")
    
    return model


def compile_transformer_model(model, learning_rate=0.001):
    """Compile Transformer model"""
    print(f"\n[*] Compiling model (learning rate: {learning_rate})")
    
    optimizer = ks.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae'],
        loss_weights=[1.0, 1.0]
    )
    
    print(f"   [OK] Compilation complete")
    
    return model


def test_transformer_model():
    """Test transformer model with dummy data"""
    print("\n" + "="*70)
    print("TRANSFORMER MODEL TEST")
    print("="*70)
    
    model = create_transformer_model(
        input_shape=(875, 1),
        d_model=128,
        num_heads=4,
        num_layers=3,
        dff=256
    )
    
    model = compile_transformer_model(model)
    
    print(f"\n[OK] Forward pass test with dummy data")
    
    dummy_input = np.random.randn(2, 875, 1).astype(np.float32)
    print(f"   Input shape: {dummy_input.shape}")
    
    predictions = model.predict(dummy_input, verbose=0)
    print(f"   SBP predictions shape: {predictions[0].shape}")
    print(f"   DBP predictions shape: {predictions[1].shape}")
    print(f"   SBP sample: {predictions[0][0]}")
    print(f"   DBP sample: {predictions[1][0]}")
    
    print("\n" + "="*70)
    print("OK Test passed successfully!")
    print("="*70)


if __name__ == '__main__':
    test_transformer_model()
