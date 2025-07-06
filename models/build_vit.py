import tensorflow as tf
from tensorflow.keras import layers

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size=16, embed_dim=768, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='VALID'
        )
        self.flatten = layers.Reshape((-1, embed_dim))  # (batch, num_patches, embed_dim)

    def call(self, images):
        x = self.projection(images)      # shape: (B, H/patch, W/patch, embed_dim)
        x = self.flatten(x)              # shape: (B, num_patches, embed_dim)
        return x
class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.cls_token = self.add_weight(name="cls_token", shape=(1, 1, embed_dim), initializer="zeros", trainable=True)
        self.pos_embedding = self.add_weight(name="pos_embedding", shape=(1, num_patches + 1, embed_dim), initializer="random_normal", trainable=True)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, x.shape[-1]])
        x = tf.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embedding
        return x


class MLP(layers.Layer):
    def __init__(self, mlp_dim, embed_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = layers.Dense(mlp_dim, activation='gelu')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.fc2 = layers.Dense(embed_dim)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(mlp_dim, embed_dim, dropout_rate)

    def call(self, x):
        attn_output = self.attn(self.norm1(x), self.norm1(x))
        x = x + self.dropout1(attn_output)
        x = x + self.mlp(self.norm2(x))
        return x

def build_vit_base(image_size=224, patch_size=16, num_layers=12,
                   hidden_dim=768, mlp_dim=3072, num_heads=12,
                   num_classes=1000, dropout_rate=0.1):
    
    num_patches = (image_size // patch_size) ** 2
    inputs = layers.Input(shape=(image_size, image_size, 3))

    # Patch + Position embedding
    x = PatchEmbedding(patch_size=patch_size, embed_dim=hidden_dim)(inputs)
    x = AddPositionEmbedding(num_patches=num_patches, embed_dim=hidden_dim)(x)

    # Transformer Encoder Blocks
    for _ in range(num_layers):
        x = TransformerEncoder(embed_dim=hidden_dim,
                               num_heads=num_heads,
                               mlp_dim=mlp_dim,
                               dropout_rate=dropout_rate)(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x[:, 0]  # Extract class token
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ViT-Base")

if __name__ == "__main__":
    model = build_vit_base()
    model.summary()
