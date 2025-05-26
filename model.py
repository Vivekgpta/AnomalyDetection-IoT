# # Model architecture and training code
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.regularizers import l2

# # DNN Model (input_dim = 34)
# dnn_model = Sequential([
#     Dense(64, input_dim=34, activation='relu'),
#     Dropout(0.3),
#     Dense(32, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='sigmoid')
# ])
# dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# def build_generator(input_dim=100, output_dim=34):
#     ...

# def build_discriminator(input_dim=34):
#     ...



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

def build_dnn(input_dim=34):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_generator(input_dim=100, output_dim=34):
    # Placeholder for generator model
    pass

def build_discriminator(input_dim=34):
    # Placeholder for discriminator model
    pass
