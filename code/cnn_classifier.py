import keras
from keras import layers, models, regularizers
from diffusion_tools import PositionEncoder

def conv_block(x, filters, kernel_size, activation, padding, reg, batch_norm, spatial_dropout=None):
    x = layers.Conv2D(filters, kernel_size, padding=padding, activation=activation, kernel_regularizer=reg)(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    if spatial_dropout:
        x = layers.SpatialDropout2D(spatial_dropout)(x)
    return x


def create_diffusion_network(image_size,
                             conv_layers,
                             dense_layers,
                             p_dropout=None,
                             p_spatial_dropout=None,
                             lambda_l2=None,
                             lrate=0.001,
                             loss='mse',
                             metrics=None,
                             padding='same',
                             conv_activation='relu',
                             dense_activation='relu',
                             nsteps=50,
                             time_embedding_dim=30):
    """
    U-Net based network for diffusion training as described in HW7
    Uses PositionEncoder from diffusion_tools.py
    """

    # Inputs
    label_input = layers.Input(shape=(image_size[0], image_size[1], 7), name='label_input')
    image_input = layers.Input(shape=(image_size[0], image_size[1], 3), name='image_input')
    time_input = layers.Input(shape=(1,), dtype='int32', name='time_input')

    # Prepare time encoding
    time_embed = PositionEncoder(max_steps=nsteps, max_dims=time_embedding_dim)(time_input)
    time_embed = keras.ops.expand_dims(time_embed, axis=1)  # [batch, 1, embed]
    time_embed = keras.ops.expand_dims(time_embed, axis=1)  # [batch, 1, 1, embed]
    time_embed = keras.ops.tile(time_embed, [1, image_size[0], image_size[1], 1])  # [batch, H, W, embed]

    # Concatenate inputs
    x = layers.Concatenate(axis=-1)([image_input, label_input, time_embed])

    reg = regularizers.l2(lambda_l2) if lambda_l2 else None
    skip_connections = []

    # Encoder
    for layer in conv_layers:
        x = conv_block(x, filters=layer['filters'], kernel_size=layer['kernel_size'],
                       activation=conv_activation, padding=padding, reg=reg,
                       batch_norm=layer.get('batch_normalization', False), spatial_dropout=p_spatial_dropout)
        skip_connections.append(x)
        if layer.get('pool_size'):
            x = layers.AveragePooling2D(pool_size=layer['pool_size'])(x)

    # Bottleneck
    for layer in dense_layers:
        x = conv_block(x, filters=layer['units'], kernel_size=(3, 3),
                       activation=dense_activation, padding=padding, reg=reg,
                       batch_norm=layer.get('batch_normalization', False), spatial_dropout=p_spatial_dropout)
    if p_dropout:
        x = layers.Dropout(p_dropout)(x)

    # Decoder
    for i, layer in reversed(list(enumerate(conv_layers))):
        if layer.get('pool_size'):
            x = layers.UpSampling2D(size=layer['pool_size'])(x)
        if i < len(skip_connections):
            skip = skip_connections[i]
            target_shape = keras.ops.shape(x)[1:3]
            label_down = layers.Resizing(target_shape[0], target_shape[1], interpolation='nearest')(label_input)
            time_down = layers.Resizing(target_shape[0], target_shape[1], interpolation='nearest')(time_embed)
            x = layers.Concatenate()([x, skip, label_down, time_down])
        x = conv_block(x, filters=layer['filters'], kernel_size=layer['kernel_size'],
                       activation=conv_activation, padding=padding, reg=reg,
                       batch_norm=layer.get('batch_normalization', False), spatial_dropout=p_spatial_dropout)

    # Output: Predict unbounded noise -> no activation
    outputs = layers.Conv2D(3, (1, 1), activation=None, padding='same')(x)

    model = models.Model(inputs={'label_input': label_input, 'image_input': image_input, 'time_input': time_input},
                         outputs=outputs)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lrate), loss=loss, metrics=metrics)

    return model
