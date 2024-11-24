# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Conv1D, LeakyReLU
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pickle import load
import pandas as pd
from tqdm import tqdm

# Generator Model
def make_generator_model(input_dim, output_dim, feature_size):
    """
    Builds the generator model for the GAN.

    Args:
        input_dim (int): Input sequence length (time steps).
        output_dim (int): Output sequence length (forecast horizon).
        feature_size (int): Number of features in each input time step.

    Returns:
        tf.keras.Sequential: The generator model.
    """
    model = tf.keras.Sequential([LSTM(units=1024, return_sequences=True, 
                                    input_shape=(input_dim, feature_size), recurrent_dropout=0.3),
                               LSTM(units=512, return_sequences=True, recurrent_dropout=0.3),
                               LSTM(units=256, return_sequences=True, recurrent_dropout=0.3),
                               LSTM(units=128, return_sequences=True, recurrent_dropout=0.3),
                               LSTM(units=64, recurrent_dropout=0.3),
                               Dense(32),
                               Dense(16),
                               Dense(8),
                               Dense(units=output_dim)])
    return model

# Discriminator Model
def make_discriminator_model(input_dim):
    """
    Builds the discriminator model for the GAN.

    Args:
        input_dim (int): Input sequence length (time steps).

    Returns:
        tf.keras.Sequential: The discriminator model.
    """
    cnn_net = tf.keras.Sequential()
    cnn_net.add(Conv1D(8, input_shape=(input_dim+1, 1), kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(16, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(32, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(64, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(128, kernel_size=1, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(LeakyReLU())
    cnn_net.add(Dense(220, use_bias=False))
    cnn_net.add(LeakyReLU())
    cnn_net.add(Dense(220, use_bias=False, activation='relu'))
    cnn_net.add(Dense(1, activation='sigmoid'))
    return cnn_net

# Discriminator Loss
def discriminator_loss(real_output, fake_output):
    """
    Computes the discriminator loss.

    Args:
        real_output (tf.Tensor): Output of the discriminator for real data.
        fake_output (tf.Tensor): Output of the discriminator for generated data.

    Returns:
        tf.Tensor: The discriminator loss.
    """
    loss_f = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # Change to False
    real_loss = loss_f(tf.ones_like(real_output), real_output)
    fake_loss = loss_f(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Generator Loss
def generator_loss(fake_output):
    """
    Computes the generator loss.

    Args:
        fake_output (tf.Tensor): Output of the discriminator for generated data.

    Returns:
        tf.Tensor: The generator loss.
    """
    loss_f = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # Change to False
    loss = loss_f(tf.ones_like(fake_output), fake_output)
    return loss

# Training Step
@tf.function
def train_step(real_x, real_y, yc, generator, discriminator, g_optimizer, d_optimizer):
    """
    Performs a single training step for the GAN.

    Args:
        real_x (tf.Tensor): Input features for the real data.
        real_y (tf.Tensor): Target values for the real data.
        yc (tf.Tensor): Conditional input data.
        generator (tf.keras.Model): Generator model.
        discriminator (tf.keras.Model): Discriminator model.
        g_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the generator.
        d_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the discriminator.

    Returns:
        tuple: Real and generated outputs, and loss metrics.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(real_x, training=True)
        generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
        d_fake_input = tf.concat([tf.cast(generated_data_reshape, tf.float64), yc], axis=1)
        real_y_reshape = tf.reshape(real_y, [real_y.shape[0], real_y.shape[1], 1])
        d_real_input = tf.concat([real_y_reshape, yc], axis=1)

        real_output = discriminator(d_real_input, training=True)
        fake_output = discriminator(d_fake_input, training=True)

        g_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return real_y, generated_data, {'d_loss': disc_loss, 'g_loss': g_loss}

# Training Loop
def train(real_x, real_y, yc, Epochs, generator, discriminator, g_optimizer, d_optimizer, checkpoint=50):
    """
    Trains the GAN for the specified number of epochs.

    Args:
        real_x (numpy.ndarray): Input features for real data.
        real_y (numpy.ndarray): Target values for real data.
        yc (numpy.ndarray): Conditional input data.
        Epochs (int): Number of training epochs.
        generator (tf.keras.Model): Generator model.
        discriminator (tf.keras.Model): Discriminator model.
        g_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the generator.
        d_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the discriminator.
        checkpoint (int): Epoch interval to save model checkpoints.

    Returns:
        tuple: Predicted and real prices, and normalized RMSE.
    """
    train_info = {}
    train_info["discriminator_loss"] = []
    train_info["generator_loss"] = []

    for epoch in tqdm(range(Epochs)):
        real_price, fake_price, loss = train_step(real_x, real_y, yc, generator, discriminator, g_optimizer, d_optimizer)
        G_losses = []
        D_losses = []
        Real_price = []
        Predicted_price = []
        D_losses.append(loss['d_loss'].numpy())
        G_losses.append(loss['g_loss'].numpy())
        Predicted_price.append(fake_price.numpy())
        Real_price.append(real_price.numpy())

        # Save model every X checkpoints
        if (epoch + 1) % checkpoint == 0:
            generator.save(f'./models_gan/{stock_name}/generator_V_{epoch}.keras')
            discriminator.save(f'./models_gan/{stock_name}/discriminator_V_{epoch}.keras')
            print('epoch', epoch + 1, 'discriminator_loss', loss['d_loss'].numpy(), 'generator_loss', loss['g_loss'].numpy())
    
        train_info["discriminator_loss"].append(D_losses)
        train_info["generator_loss"].append(G_losses)
  
    Predicted_price = np.array(Predicted_price)
    Predicted_price = Predicted_price.reshape(Predicted_price.shape[1], Predicted_price.shape[2])
    Real_price = np.array(Real_price)
    Real_price = Real_price.reshape(Real_price.shape[1], Real_price.shape[2])

    plt.subplot(2, 1, 1)
    plt.plot(train_info["discriminator_loss"], label='Disc_loss', color='#000000')
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_info["generator_loss"], label='Gen_loss', color='#000000')
    plt.xlabel('Epoch')
    plt.ylabel('Generator Loss')
    plt.legend()

    plt.show()

    return Predicted_price, Real_price, np.sqrt(mean_squared_error(Real_price, Predicted_price)) / np.mean(Real_price)

# Plot Training Results
def plot_results(Real_price, Predicted_price, index_train):
    """
    Plots the training results (real vs. predicted prices).

    Args:
        Real_price (numpy.ndarray): Real stock prices.
        Predicted_price (numpy.ndarray): Predicted stock prices by the generator.
        index_train (pandas.Index): Index for the training data.
    """
    X_scaler = load(open('/content/X_scaler.pkl', 'rb'))
    y_scaler = load(open('/content/y_scaler.pkl', 'rb'))
    train_predict_index = index_train

    rescaled_Real_price = y_scaler.inverse_transform(Real_price)
    rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_price)

    predict_result = pd.DataFrame()
    for i in range(rescaled_Predicted_price.shape[0]):
        y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns=["predicted_price"], index=train_predict_index[i:i+output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
  
    real_price = pd.DataFrame()
    for i in range(rescaled_Real_price.shape[0]):
        y_train = pd.DataFrame(rescaled_Real_price[i], columns=["real_price"], index=train_predict_index[i:i+output_dim])
        real_price = pd.concat([real_price, y_train], axis=1, sort=False)
  
    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_price['real_mean'] = real_price.mean(axis=1)

    plt.figure(figsize=(16, 8))
    plt.plot(real_price["real_mean"])
    plt.plot(predict_result["predicted_mean"], color='r')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
    plt.title("The result of Training", fontsize=20)
    plt.show()

    predicted = predict_result["predicted_mean"]
    real = real_price["real_mean"]
    For_MSE = pd.concat([predicted, real], axis=1)
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('-- Train RMSE -- ', RMSE)

# Test Code
@tf.function 
def eval_op(generator, real_x):
    """
    Performs evaluation using the generator model.

    Args:
        generator (tf.keras.Model): Generator model.
        real_x (tf.Tensor): Input features for the evaluation data.

    Returns:
        tf.Tensor: Generated output.
    """
    generated_data = generator(real_x, training=False)
    return generated_data

# Plot Test Results
def plot_test_data(Real_test_price, Predicted_test_price, index_test):
    """
    Plots the test results (real vs. predicted prices).

    Args:
        Real_test_price (numpy.ndarray): Real stock prices for test data.
        Predicted_test_price (numpy.ndarray): Predicted stock prices by the generator for test data.
        index_test (pandas.Index): Index for the test data.
    """
    X_scaler = load(open('X_scaler.pkl', 'rb'))
    y_scaler = load(open('y_scaler.pkl', 'rb'))
    test_predict_index = index_test

    rescaled_Real_price = y_scaler.inverse_transform(Real_test_price)
    rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_test_price)

    predict_result = pd.DataFrame()
    for i in range(rescaled_Predicted_price.shape[0]):
        y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns=["predicted_price"], index=test_predict_index[i:i+output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
  
    real_price = pd.DataFrame()
    for i in range(rescaled_Real_price.shape[0]):
        y_train = pd.DataFrame(rescaled_Real_price[i], columns=["real_price"], index=test_predict_index[i:i+output_dim])
        real_price = pd.concat([real_price, y_train], axis=1, sort=False)
  
    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_price['real_mean'] = real_price.mean(axis=1)

    predicted = predict_result["predicted_mean"]
    real = real_price["real_mean"]
    For_MSE = pd.concat([predicted, real], axis=1)
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('Test RMSE: ', RMSE)
    
    plt.figure(figsize=(16, 8))
    plt.plot(real_price["real_mean"], color='#00008B')
    plt.plot(predict_result["predicted_mean"], color='#8B0000', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
    plt.title(f"Prediction on test data for {stock_name}", fontsize=20)
    plt.show()


def evaluate_model(generator, Real_test_price, Predicted_test_price, index_test):
    """
    Evaluates the model's predictions using MSE, MAPE, and RMSE.
    Matches the logic from plot_test_data for RMSE calculation.

    Args:
        generator (tf.keras.Model): The generator model.
        Real_test_price (numpy.ndarray): The true test target values.
        Predicted_test_price (numpy.ndarray): The predicted test target values.
        index_test (numpy.ndarray): The index for test samples.

    Returns:
        dict: A dictionary containing MSE, MAPE, and RMSE.
    """
    # Load scalers
    X_scaler = load(open('X_scaler.pkl', 'rb'))
    y_scaler = load(open('y_scaler.pkl', 'rb'))
    
    # Rescale predictions and true values
    rescaled_Real_price = y_scaler.inverse_transform(Real_test_price)
    rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_test_price)

    # Create DataFrames for alignment
    predict_result = pd.DataFrame()
    for i in range(rescaled_Predicted_price.shape[0]):
        y_predict = pd.DataFrame(
            rescaled_Predicted_price[i], 
            columns=["predicted_price"], 
            index=index_test[i : i + rescaled_Predicted_price.shape[1]]
        )
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    real_price = pd.DataFrame()
    for i in range(rescaled_Real_price.shape[0]):
        y_train = pd.DataFrame(
            rescaled_Real_price[i], 
            columns=["real_price"], 
            index=index_test[i : i + rescaled_Real_price.shape[1]]
        )
        real_price = pd.concat([real_price, y_train], axis=1, sort=False)

    # Compute mean predictions and real prices
    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_price['real_mean'] = real_price.mean(axis=1)

    # Extract final predictions and real values
    predicted = predict_result["predicted_mean"]
    real = real_price["real_mean"]

    # Calculate metrics
    mse = mean_squared_error(real, predicted)
    mape = mean_absolute_percentage_error(real, predicted)
    rmse = np.sqrt(mse)

    # Print results
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    return {"MSE": mse, "MAPE": mape, "RMSE": rmse}
