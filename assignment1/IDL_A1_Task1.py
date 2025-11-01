# %%
import keras
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.model_selection import train_test_split

# %%
#Load the Fasion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data() 

#Create the validation set from the training set
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=1, stratify=y_train_full)
X_valid, X_train = X_valid/ 255.0, X_train/ 255.0
#Since we are going to train the neural network using Gradient Descent, we must scale the input features down to the 0-1 range 

# %% [markdown]
# ###Check the class balance
# 

# %%
#Check the class balance of the training and validation set
train_counts = Counter(y_train)
val_counts = Counter(y_valid)
test_counts = Counter(y_test)

#Convert to list format for plotting
train_classes, train_freq = zip(*sorted(train_counts.items()))
val_classes, val_freq = zip(*sorted(val_counts.items()))
test_classes, test_freq = zip(*sorted(test_counts.items()))

#Plot class balance
fig, axes = plt.subplots(1, 3, figsize=(14, 6))

sns.barplot(x=train_classes, y=train_freq, ax=axes[0])
axes[0].set_title("Class Balance in Training Set")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Frequency")

sns.barplot(x=val_classes, y=val_freq, ax=axes[1])
axes[1].set_title("Class Balance in Validation Set")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Frequency")

sns.barplot(x=test_classes, y=test_freq, ax=axes[2])
axes[2].set_title("Class Balance in Test Set")
axes[2].set_xlabel("Class")
axes[2].set_ylabel("Frequency")

plt.show()

# %% [markdown]
# #MLP Model
# Set epochs = 5,
# 
# early convergence thingy and to get preliminary results
# MLP model runs faster so we can test more stuff
# 
# why we used validation accuracy

# %% [markdown]
# ###Find the best optimizer and its learning rate
# We start with tuning the optimizer and learning rate, as this parameter dictates how the model converges during training, ultimately influencing other hyperparameters such regularization, weight initialization, and network architecture. Different optimizers influence the stability and speed of convergence, while the learning rate affects step sizes.
# 
# We tested 9 different optimizers with learning rates of 0.001, 0.01 and 0.1.

# %%
#Optimizer and Learning Rate
keras.utils.set_random_seed(1)

optimizers = {
    'adam': [keras.optimizers.Adam, [0.001, 0.01, 0.1]],
    'rmsprop': [keras.optimizers.RMSprop, [0.001, 0.01, 0.1]],
    'sgd': [keras.optimizers.SGD, [0.001, 0.01, 0.1]],
    'adamw': [keras.optimizers.AdamW, [0.001, 0.01, 0.1]],
    'adadelta': [keras.optimizers.Adadelta, [0.001, 0.01, 0.1]],
    'adagrad': [keras.optimizers.Adagrad, [0.001, 0.01, 0.1]],
    'adamax': [keras.optimizers.Adamax, [0.001, 0.01, 0.1]],
    'nadam': [keras.optimizers.Nadam, [0.001, 0.01, 0.1]],
    'ftrl': [keras.optimizers.Ftrl, [0.001, 0.01, 0.1]],
}

model_results = []

#Iterate over optimizers and learning rates
for opt_name, (optimizer_class, learning_rates) in optimizers.items():
    for lr in learning_rates:

        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(28, 28)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(300, activation="relu"))
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(10, activation="softmax"))

        #Compile the model
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer_class(learning_rate=lr), metrics=["accuracy"])

        #Train the model
        history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid), verbose=0)

        #Evaluate the model
        val_accuracy = round(max(history.history['val_accuracy']), 5)
        test_loss, test_accuracy = map(lambda x: round(x, 5), model.evaluate(X_test, y_test, verbose=0))

        #Save model and results to best_models
        model_results.append({
            'Optimizer': opt_name,
            'Learning Rate': lr,
            'Test Accuracy': test_accuracy,
            'Test Loss': test_loss,
            'Validation Accuracy': val_accuracy
        })

        #Print the results
        print(f"Optimizer: {opt_name}")
        print(f"Learning Rate: {lr}")
        print(f"Validation Accuracy: {val_accuracy:.5f}")
        print(f"Test Accuracy: {test_accuracy:.5f}")
        print(f"Test Loss: {test_loss:.5f}")
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()
        print("---------------------------------------------------------------------------")

# Sort the models by Validation Accuracy in descending order
model_results = sorted(model_results, key=lambda x: x['Validation Accuracy'], reverse=True)

# Print the top 3 models based on Validation Accuracy
print("\nTop 3 models based on Validation Accuracy:")
for i, model_info in enumerate(model_results[:3], 1):
    print(f"Model {i}")
    print(f"Optimizer: {model_info['Optimizer']}")
    print(f"Learning Rate: {model_info['Learning Rate']}")
    print(f"Validation Accuracy: {model_info['Validation Accuracy']:.5f}")
    print(f"Test Accuracy: {model_info['Test Accuracy']:.5f}")
    print(f"Test Loss: {model_info['Test Loss']:.5f}")
    print("---------------------------------------------------------------------------")

# %% [markdown]
# ###Find the best model architecture

# %% [markdown]
# #####With dropout

# %%
#Model Architecture
keras.utils.set_random_seed(1)

model_configs = [
    {"name": "Default MLP", "layers": [300, 100]},
    {"name": "Shallow MLP", "layers": [128]},
    {"name": "Moderate MLP", "layers": [256, 128]},
    {"name": "Wide MLP", "layers": [512]},
    {"name": "Deep MLP", "layers": [256, 128, 64]},
    {"name": "Very Deep MLP", "layers": [512, 256, 128, 64]}
]

#Iterate over each architecture configuration
for config in model_configs:
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(28, 28)))
    model.add(keras.layers.Flatten())

    #Add hidden layers with specified neurons and dropout regularization
    for units in config["layers"]:
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.Dropout(0.2))

    #Output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    #Train the model
    optimizer = keras.optimizers.Nadam(learning_rate=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid), verbose=0)

    #Evaluate on test data
    val_accuracy = max(history.history['val_accuracy'])
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    #Print the model summary and performance
    model.summary()
    print(config["name"])
    print(f"Validation Accuracy: {val_accuracy:.5f}")
    print(f"Test Accuracy: {test_accuracy:.5f}")
    print(f"Test Loss: {test_loss:.5f}")
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title(f"{config['name']} Training Process")
    plt.show()
    print("---------------------------------------------------------------------------")

# %% [markdown]
# #####Without dropout

# %%
#Model Architecture
keras.utils.set_random_seed(1)

model_configs = [
    {"name": "Default MLP", "layers": [300, 100]},
    {"name": "Shallow MLP", "layers": [128]},
    {"name": "Moderate MLP", "layers": [256, 128]},
    {"name": "Wide MLP", "layers": [512]},
    {"name": "Deep MLP", "layers": [256, 128, 64]},
    {"name": "Very Deep MLP", "layers": [512, 256, 128, 64]}
]

#Iterate over each architecture configuration
for config in model_configs:
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(28, 28)))
    model.add(keras.layers.Flatten())

    #Add hidden layers with specified neurons (without dropout)
    for units in config["layers"]:
        model.add(keras.layers.Dense(units, activation='relu'))

    #Output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    #Train the model
    optimizer = keras.optimizers.Nadam(learning_rate=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid), verbose=0)

    #Evaluate on test data
    val_accuracy = max(history.history['val_accuracy'])
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    #Print the model summary and performance
    model.summary()
    print(config["name"])
    print(f"Validation Accuracy: {val_accuracy:.5f}")
    print(f"Test Accuracy: {test_accuracy:.5f}")
    print(f"Test Loss: {test_loss:.5f}")
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title(f"{config['name']} Training Process")
    plt.show()
    print("---------------------------------------------------------------------------")

# %% [markdown]
# ###Finding the best activation function

# %%
#Activation Function
keras.utils.set_random_seed(1)

activation_functions = {
    'relu': 'relu',
    'leakyrelu': keras.layers.LeakyReLU(),
    'elu': 'elu',
    'softmax': 'softmax',
    'sigmoid': 'sigmoid',
    'tanH': 'tanh'
}

model_results = []

#Iterate over each activation function
for act_name, activation in activation_functions.items():

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(28, 28)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation=activation))
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128, activation=activation))
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation="softmax"))

    #Compile the model
    optimizer = keras.optimizers.Nadam(learning_rate=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    #Train the model
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid), verbose=0)

    #Evaluate the model
    val_accuracy = round(max(history.history['val_accuracy']), 5)
    test_loss, test_accuracy = map(lambda x: round(x, 5), model.evaluate(X_test, y_test, verbose=0))

    #Save model and results to model_results
    model_results.append({
        'Activation Function': act_name,
        'Test Accuracy': test_accuracy,
        'Test Loss': test_loss,
        'Validation Accuracy': val_accuracy
    })

    #Print the results
    print(f"Activation Function: {act_name}")
    print(f"Validation Accuracy: {val_accuracy:.5f}")
    print(f"Test Accuracy: {test_accuracy:.5f}")
    print(f"Test Loss: {test_loss:.5f}")
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    print("---------------------------------------------------------------------------")

#Sort the models by Validation Accuracy in descending order
model_results = sorted(model_results, key=lambda x: x['Validation Accuracy'], reverse=True)

#Print the top 3 models based on Validation Accuracy
print("\nTop 3 models based on Validation Accuracy:")
for i, model_info in enumerate(model_results[:3], 1):
    print(f"Model {i}")
    print(f"Activation Function: {model_info['Activation Function']}")
    print(f"Validation Accuracy: {model_info['Validation Accuracy']:.5f}")
    print(f"Test Accuracy: {model_info['Test Accuracy']:.5f}")
    print(f"Test Loss: {model_info['Test Loss']:.5f}")
    print("---------------------------------------------------------------------------")

# %% [markdown]
# ###Finding the best initializer

# %%
#Initializers
keras.utils.set_random_seed(1)

initializers = {
    'RandomNormal': keras.initializers.RandomNormal(),
    'RandomUniform': keras.initializers.RandomUniform(),
    'TruncatedNormal': keras.initializers.TruncatedNormal(),
    'Zeros': keras.initializers.Zeros(),
    'Ones': keras.initializers.Ones(),
    'GlorotNormal': keras.initializers.GlorotNormal(),
    'GlorotUniform': keras.initializers.GlorotUniform(), # default
    'HeNormal': keras.initializers.HeNormal(),
    'HeUniform': keras.initializers.HeUniform(),
    'Orthogonal': keras.initializers.Orthogonal(),
    'Constant': keras.initializers.Constant(),
    'VarianceScaling': keras.initializers.VarianceScaling(),
    'LecunNormal': keras.initializers.LecunNormal(),
    'LecunUniform': keras.initializers.LecunUniform(),
    'Identity': keras.initializers.Identity()
}

best_models = []

#Iterate over each initializer
for init_name, initializer in initializers.items():

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(28, 28)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer))
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer))
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation="softmax"))  # initializer is kept as default

    #Compile the model
    optimizer = keras.optimizers.Nadam(learning_rate=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    #Train the model
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid), verbose=0)

    #Evaluate the model
    val_accuracy = max(history.history['val_accuracy'])
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    #Save model and results to best_models
    best_models.append({
        'Initializer': init_name,
        'Test Accuracy': test_accuracy,
        'Test Loss': test_loss,
        'Validation Accuracy': val_accuracy
    })

    #Print the results
    print(f"Initializer: {init_name}")
    print(f"Validation Accuracy: {val_accuracy:.5f}")
    print(f"Test Accuracy: {test_accuracy:.5f}")
    print(f"Test Loss: {test_loss:.5f}")
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    print("---------------------------------------------------------------------------")

#Sort the models by Validation Accuracy in descending order
best_models = sorted(best_models, key=lambda x: x['Validation Accuracy'], reverse=True)

#Print the top 3 models based on Validation Accuracy
print("\nTop 3 models based on Validation Accuracy:")
for i, model_info in enumerate(best_models[:3], 1):
    print(f"Model {i}")
    print(f"Initializer: {model_info['Initializer']}")
    print(f"Validation Accuracy: {model_info['Validation Accuracy']}")
    print(f"Test Accuracy: {model_info['Test Accuracy']}")
    print(f"Test Loss: {model_info['Test Loss']}")
    print("---------------------------------------------------------------------------")

# %% [markdown]
# ###Finding the best regularisation

# %%
#Regularisation
keras.utils.set_random_seed(1)

#Define regularization methods with lambda values or dropout rates
regularizations = {
    'L1': [keras.regularizers.L1(l) for l in [0.0001, 0.001, 0.01]],
    'L2': [keras.regularizers.L2(l) for l in [0.0001, 0.001, 0.01]],
    'L1_L2': [keras.regularizers.L1L2(l1=l1, l2=l2) for l1 in [0.0001, 0.001, 0.01] for l2 in [0.0001, 0.001, 0.01]],
    'Dropout': [keras.layers.Dropout(rate) for rate in [0.2, 0.3, 0.5]]
}

model_results = []
initializer = keras.initializers.GlorotNormal()

#Iterate over each regularization method and its values
for reg_name, reg_values in regularizations.items():
    for reg in reg_values:

        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(28, 28)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer, kernel_regularizer=reg if reg_name != 'Dropout' else None))

        #Apply Dropout if specified
        if reg_name == 'Dropout':
            model.add(reg)  # Add Dropout layer

        model.add(keras.layers.Dense(128, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer, kernel_regularizer=reg if reg_name != 'Dropout' else None))

        #Apply Dropout if specified
        if reg_name == 'Dropout':
            model.add(reg)  # Add Dropout layer

        model.add(keras.layers.Dense(10, activation="softmax"))

        #Compile the model
        optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        #Train the model
        history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid), verbose=0)

        #Evaluate the model
        val_accuracy = round(max(history.history['val_accuracy']), 5)
        test_loss, test_accuracy = map(lambda x: round(x, 5), model.evaluate(X_test, y_test, verbose=0))

        #Save model and results to best_models
        model_results.append({
            'Regularization Method': reg_name,
            'Lambda': reg,
            'Test Accuracy': test_accuracy,
            'Test Loss': test_loss,
            'Validation Accuracy': val_accuracy
        })

        #Print the results
        print(f"Regularization Method: {reg_name}, Lambda: {reg}")
        print(f"Validation Accuracy: {val_accuracy:.5f}")
        print(f"Test Accuracy: {test_accuracy:.5f}")
        print(f"Test Loss: {test_loss:.5f}")
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()
        print("---------------------------------------------------------------------------")

#Sort the models by Validation Accuracy in descending order
model_results = sorted(model_results, key=lambda x: x['Validation Accuracy'], reverse=True)

#Print the top 3 models based on Validation Accuracy
print("\nTop 3 models based on Test Accuracy:")
for i, model_info in enumerate(model_results[:3], 1):
    print(f"Model {i}")
    print(f"Regularization Method: {model_info['Regularization Method']}")
    print(f"Lambda: {model_info['Lambda']}")
    print(f"Validation Accuracy: {model_info['Validation Accuracy']:.5f}")
    print(f"Test Accuracy: {model_info['Test Accuracy']:.5f}")
    print(f"Test Loss: {model_info['Test Loss']:.5f}")
    print("---------------------------------------------------------------------------")

# %% [markdown]
# ###Perform grid search to find the Top 3 best model

# %%
keras.utils.set_random_seed(1)

#Hyperparameters for grid search
optimizers = {
    'nadam': lambda: keras.optimizers.Nadam(learning_rate=0.001),
    'adagrad': lambda: keras.optimizers.Adagrad(learning_rate=0.1),
    'adam': lambda: keras.optimizers.Adam(learning_rate=0.001),
}

activation_functions = {
    'relu': 'relu',
    'leakyrelu': keras.layers.LeakyReLU(),
    'elu': 'elu'
}

initializers = {
    'GlorotNormal': keras.initializers.GlorotNormal(),
    'TruncatedNormal': keras.initializers.TruncatedNormal(),
    'GlorotUniform': keras.initializers.GlorotUniform(), # default
}

regularizations = {
    'None': [None],  #No regularization
    'L2': [keras.regularizers.L2(0.0001)],
    'Dropout': [keras.layers.Dropout(rate) for rate in [0.2, 0.3]]
}

model_results = []

#Perform grid search
for init_name, initializer in initializers.items():
    for act_name, activation in activation_functions.items():
        for opt_name, optimizer_fn in optimizers.items():
            for reg_name, reg_values in regularizations.items():
                for reg in reg_values:

                    #Define the model
                    model = keras.models.Sequential()
                    model.add(keras.layers.Input(shape=(28, 28)))
                    model.add(keras.layers.Flatten())
                    model.add(keras.layers.Dense(300, activation=activation, kernel_initializer=initializer,
                                                 kernel_regularizer=reg if reg_name == 'L2' else None))

                    #Apply Dropout if specified
                    if reg_name == 'Dropout':
                        model.add(reg)  #Add Dropout layer

                    model.add(keras.layers.Dense(100, activation=activation, kernel_initializer=initializer,
                                                 kernel_regularizer=reg if reg_name == 'L2' else None))

                    #Apply Dropout if specified
                    if reg_name == 'Dropout':
                        model.add(reg)  #Add Dropout layer

                    model.add(keras.layers.Dense(10, activation="softmax"))

                    #Instantiate a new optimizer for each model instance
                    optimizer = optimizer_fn()
                    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

                    #Train the model
                    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid), verbose=0)

                    #Evaluate the model
                    val_accuracy = round(max(history.history['val_accuracy']), 5)
                    test_loss, test_accuracy = map(lambda x: round(x, 5), model.evaluate(X_test, y_test, verbose=0))

                    #Store model results
                    model_results.append({
                        'Initializer': init_name,
                        'Activation': act_name,
                        'Optimizer': opt_name,
                        'Regularization Method': reg_name,
                        'Lambda': reg,
                        'Test Accuracy': test_accuracy,
                        'Test Loss': test_loss,
                        'Validation Accuracy': val_accuracy
                    })

                    #Print the results
                    print(f"Initializer: {init_name}, Activation: {act_name}, Optimizer: {opt_name}, "
                          f"Regularization Method: {reg_name}, Lambda: {reg}")
                    print(f"Validation Accuracy: {val_accuracy:.5f}")
                    print(f"Test Accuracy: {test_accuracy:.5f}")
                    print(f"Test Loss: {test_loss:.5f}")
                    pd.DataFrame(history.history).plot(figsize=(8, 5))
                    plt.grid(True)
                    plt.gca().set_ylim(0, 1)
                    plt.show()
                    print("---------------------------------------------------------------------------")

#Sort the models by Validation Accuracy in descending order
model_results = sorted(model_results, key=lambda x: x['Validation Accuracy'], reverse=True)

#Print the top 3 models based on Validation Accuracy
print("\nTop 3 models based on Validation Accuracy:")
for i, model_info in enumerate(model_results[:3], 1):
    print(f"Model {i}")
    print(f"Initializer: {model_info['Initializer']}")
    print(f"Activation: {model_info['Activation']}")
    print(f"Optimizer: {model_info['Optimizer']}")
    print(f"Regularization Method: {model_info['Regularization Method']}")
    print(f"Lambda: {model_info['Lambda']}")
    print(f"Validation Accuracy: {model_info['Validation Accuracy']:.5f}")
    print(f"Test Accuracy: {model_info['Test Accuracy']:.5f}")
    print(f"Test Loss: {model_info['Test Loss']:.5f}")
    print("---------------------------------------------------------------------------")


# %% [markdown]
# #CNN Model

# %% [markdown]
# ###Finding the best optimizer and learning rate

# %%
#Optimizer and Learning Rates
keras.utils.set_random_seed(1)

optimizers = {
    'adam': [keras.optimizers.Adam, [0.001, 0.01]],
    'sgd': [keras.optimizers.SGD, [0.001, 0.01]],
    'rmsprop': [keras.optimizers.RMSprop, [0.001, 0.01]],
}

model_results = []

#Iterate over optimizers and learning rates
for opt_name, (optimizer_class, learning_rates) in optimizers.items():
    for lr in learning_rates:

        model = keras.models.Sequential([
          keras.layers.Input(shape=[28, 28, 1]),
          keras.layers.Conv2D(32, 5, activation="relu", padding="same"),
          keras.layers.MaxPooling2D(2),
          keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
          keras.layers.MaxPooling2D(2),
          keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
          keras.layers.MaxPooling2D(2),
          keras.layers.Flatten(),
          keras.layers.Dense(64, activation="relu"),
          keras.layers.Dropout(0.3),
          keras.layers.Dense(10, activation="softmax")
        ])

        #Compile the model
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer_class(learning_rate=lr), metrics=["accuracy"])

        #Train the model
        history = model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid), verbose=0)

        #Evaluate the model
        val_accuracy = round(max(history.history['val_accuracy']), 5)
        test_loss, test_accuracy = map(lambda x: round(x, 5), model.evaluate(X_test, y_test, verbose=0))

        #Save model and results to best_models
        model_results.append({
            'Optimizer': opt_name,
            'Learning Rate': lr,
            'Test Accuracy': test_accuracy,
            'Test Loss': test_loss,
            'Validation Accuracy': val_accuracy
        })

        #Print the results
        print(f"Optimizer: {opt_name}")
        print(f"Learning Rate: {lr}")
        print(f"Validation Accuracy: {val_accuracy:.5f}")
        print(f"Test Accuracy: {test_accuracy:.5f}")
        print(f"Test Loss: {test_loss:.5f}")

        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()
        print("---------------------------------------------------------------------------")

#Sort the models by Validation Accuracy in descending order
model_results = sorted(model_results, key=lambda x: x['Validation Accuracy'], reverse=True)

#Print the top 3 models based on Validation Accuracy
print("\nTop 3 models based on Validation Accuracy:")
for i, model_info in enumerate(model_results[:3], 1):
    print(f"Model {i}")
    print(f"Optimizer: {model_info['Optimizer']}")
    print(f"Learning Rate: {model_info['Learning Rate']}")
    print(f"Validation Accuracy: {model_info['Validation Accuracy']:.5f}")
    print(f"Test Accuracy: {model_info['Test Accuracy']:.5f}")
    print(f"Test Loss: {model_info['Test Loss']:.5f}")
    print("---------------------------------------------------------------------------")

# %% [markdown]
# ###Finding the best architecture

# %%
#Define each model architecture as a function
def model_1():
    return keras.models.Sequential([
        keras.layers.Input(shape=[28, 28, 1]),
        keras.layers.Conv2D(32, 5, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])

def model_2():
    return keras.models.Sequential([
        keras.layers.Input(shape=[28, 28, 1]),
        keras.layers.Conv2D(32, 5, activation="relu", padding="same"),
        keras.layers.GlobalMaxPooling2D(), # Global Max Pooling applied here
        keras.layers.Dense(64, activation="relu"), # Dense layer after Global Max Pooling
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])

def model_3():
    return keras.models.Sequential([
        keras.layers.Input(shape=[28, 28, 1]),
        keras.layers.Conv2D(32, 5, activation="relu", padding="same"),
        keras.layers.AveragePooling2D(2),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.AveragePooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.AveragePooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])

def model_4():
    return keras.models.Sequential([
        keras.layers.Input(shape=[28, 28, 1]),
        keras.layers.Conv2D(32, 5, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(64, activation="relu"), # Dense layer after GlobalAveragePooling2D
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])

def model_5():
    return keras.models.Sequential([
        keras.layers.Input(shape=[28, 28, 1]),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation="softmax")
    ])

def model_6():
    return keras.models.Sequential([
        keras.layers.Input(shape=[28, 28, 1]),
        keras.layers.Conv2D(32, 7, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 5, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax")
    ])

def model_7():
    return keras.models.Sequential([
        keras.layers.Input(shape=[28, 28, 1]),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation="softmax")
    ])

def model_8():
    return keras.models.Sequential([
        keras.layers.Input(shape=[28, 28, 1]),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.AveragePooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 5, activation="relu", padding="same"),
        keras.layers.AveragePooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation="softmax")
    ])

def model_9():
    return keras.models.Sequential([
        keras.layers.Input(shape=[28, 28, 1]),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax")
    ])

model_architectures = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9]

#Optimizer and Learning Rates
keras.utils.set_random_seed(1)

optimizers = {
    'adam': [keras.optimizers.Adam, [0.001]]
}

model_results = []

#Iterate over architectures, optimizers, and learning rates
for model_fn in model_architectures:
    for opt_name, (optimizer_class, learning_rates) in optimizers.items():
        for lr in learning_rates:

            #Initialize and compile the model
            model = model_fn()
            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer_class(learning_rate=lr),
                metrics=["accuracy"]
            )

            #Train the model
            history = model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid), verbose=0)

            #Evaluate the model
            val_accuracy = round(max(history.history['val_accuracy']), 5)
            test_loss, test_accuracy = map(lambda x: round(x, 5), model.evaluate(X_test, y_test, verbose=0))

            #Save model and results to model_results
            model_results.append({
                'Model': model_fn.__name__,
                'Optimizer': opt_name,
                'Learning Rate': lr,
                'Test Accuracy': test_accuracy,
                'Test Loss': test_loss,
                'Validation Accuracy': val_accuracy
            })

            #Print the results
            print(f"Model: {model_fn.__name__}")
            print(f"Optimizer: {opt_name}")
            print(f"Learning Rate: {lr}")
            print(f"Validation Accuracy: {val_accuracy:.5f}")
            print(f"Test Accuracy: {test_accuracy:.5f}")
            print(f"Test Loss: {test_loss:.5f}")

            pd.DataFrame(history.history).plot(figsize=(8, 5))
            plt.grid(True)
            plt.gca().set_ylim(0, 1)
            plt.show()
            print("---------------------------------------------------------------------------")

#Sort the models by Validation Accuracy in descending order
model_results = sorted(model_results, key=lambda x: x['Validation Accuracy'], reverse=True)

#Print the top 3 models based on Validation Accuracy
print("\nTop 3 models based on Validation Accuracy:")
for i, model_info in enumerate(model_results[:3], 1):
    print(f"Model {i}")
    print(f"Model Architecture: {model_info['Model']}")
    print(f"Optimizer: {model_info['Optimizer']}")
    print(f"Learning Rate: {model_info['Learning Rate']}")
    print(f"Validation Accuracy: {model_info['Validation Accuracy']:.5f}")
    print(f"Test Accuracy: {model_info['Test Accuracy']:.5f}")
    print(f"Test Loss: {model_info['Test Loss']:.5f}")
    print("---------------------------------------------------------------------------")

# %% [markdown]
# ###Finding the best activation function

# %%
#Activation Function
keras.utils.set_random_seed(1)

activation_functions = {
    'relu': 'relu',
    'leakyrelu': keras.layers.LeakyReLU(),
    'elu': 'elu',
    'softmax': 'softmax',
    'sigmoid': 'sigmoid',
    'tanH': 'tanh'
}

model_results = []

#Iterate over each activation function
for act_name, activation in activation_functions.items():

    model = keras.models.Sequential([
        keras.layers.Input(shape=[28, 28, 1]),
        keras.layers.Conv2D(64, 3, activation=activation, padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation=activation, padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation=activation, padding="same"),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation=activation),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation=activation),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax")
    ])

    #Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    #Train the model
    history = model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid), verbose=0)

    #Evaluate the model
    val_accuracy = round(max(history.history['val_accuracy']), 5)
    test_loss, test_accuracy = map(lambda x: round(x, 5), model.evaluate(X_test, y_test, verbose=0))

    #Save model and results to model_results
    model_results.append({
        'Activation Function': act_name,
        'Test Accuracy': test_accuracy,
        'Test Loss': test_loss,
        'Validation Accuracy': val_accuracy
    })

    #Print the results
    print(f"Activation Function: {act_name}")
    print(f"Validation Accuracy: {val_accuracy:.5f}")
    print(f"Test Accuracy: {test_accuracy:.5f}")
    print(f"Test Loss: {test_loss:.5f}")
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    print("---------------------------------------------------------------------------")

#Sort the models by Validation Accuracy in descending order
model_results = sorted(model_results, key=lambda x: x['Validation Accuracy'], reverse=True)

#Print the top 3 models based on Validation Accuracy
print("\nTop 3 models based on Validation Accuracy:")
for i, model_info in enumerate(model_results[:3], 1):
    print(f"Model {i}")
    print(f"Activation Function: {model_info['Activation Function']}")
    print(f"Validation Accuracy: {model_info['Validation Accuracy']:.5f}")
    print(f"Test Accuracy: {model_info['Test Accuracy']:.5f}")
    print(f"Test Loss: {model_info['Test Loss']:.5f}")
    print("---------------------------------------------------------------------------")

# %% [markdown]
# ###Finding the best weight initializer

# %%
#Initializers
keras.utils.set_random_seed(1)

initializers = {
    'RandomNormal': keras.initializers.RandomNormal(),
    'GlorotUniform': keras.initializers.GlorotUniform(),  # default
    'HeNormal': keras.initializers.HeNormal(),
    'LecunNormal': keras.initializers.LecunNormal(),
    'Orthogonal': keras.initializers.Orthogonal(),
    'VarianceScaling': keras.initializers.VarianceScaling()
}

best_models = []

#Iterate over each initializer
for init_name, initializer in initializers.items():

    model = keras.models.Sequential([
        keras.layers.Input(shape=[28, 28, 1]),
        keras.layers.Conv2D(64, 3, activation='elu', padding="same", kernel_initializer=initializer),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation='elu', padding="same", kernel_initializer=initializer),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation='elu', padding="same", kernel_initializer=initializer),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='elu', kernel_initializer=initializer),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='elu', kernel_initializer=initializer),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax")
    ])

    #Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    #Train the model
    history = model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid), verbose=0)

    #Evaluate the model
    val_accuracy = max(history.history['val_accuracy'])
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    #Save model and results to best_models
    best_models.append({
        'Initializer': init_name,
        'Test Accuracy': test_accuracy,
        'Test Loss': test_loss,
        'Validation Accuracy': val_accuracy
    })

    #Print the results
    print(f"Initializer: {init_name}")
    print(f"Validation Accuracy: {val_accuracy:.5f}")
    print(f"Test Accuracy: {test_accuracy:.5f}")
    print(f"Test Loss: {test_loss:.5f}")
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    print("---------------------------------------------------------------------------")

#Sort the models by Validation Accuracy in descending order
best_models = sorted(best_models, key=lambda x: x['Validation Accuracy'], reverse=True)

#Print the top 3 models based on Validation Accuracy
print("\nTop 3 models based on Validation Accuracy:")
for i, model_info in enumerate(best_models[:3], 1):
    print(f"Model {i}")
    print(f"Initializer: {model_info['Initializer']}")
    print(f"Validation Accuracy: {model_info['Validation Accuracy']}")
    print(f"Test Accuracy: {model_info['Test Accuracy']}")
    print(f"Test Loss: {model_info['Test Loss']}")
    print("---------------------------------------------------------------------------")

# %% [markdown]
# ###Finding the best regularisation

# %%
keras.utils.set_random_seed(1)

# Define regularization methods with lambda values or dropout rates, including None
regularizations = {
    'L1': [keras.regularizers.L1(l) for l in [0.001, 0.01]],
    'L2': [keras.regularizers.L2(l) for l in [0.001, 0.01]],
    'L1_L2': [keras.regularizers.L1L2(l1=l1, l2=l2) for l1 in [0.001, 0.01] for l2 in [0.001, 0.01]],
    'Dropout': [0.2, 0.4],  # Dropout rates as specified
    'None': [None]  # No regularization or dropout
}

model_results = []
initializer = keras.initializers.RandomNormal()

# Iterate over each regularization method and its values
for reg_name, reg_values in regularizations.items():
    for reg in reg_values:

        # Start building the model
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=[28, 28, 1]))

        # If regularization method is L1, L2, or L1_L2, use kernel_regularizer in Conv2D layers
        kernel_regularizer = reg if reg_name in ['L1', 'L2', 'L1_L2'] else None

        # Add Conv2D layers with kernel regularization if specified
        model.add(keras.layers.Conv2D(
            64, 3, activation='elu', padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=kernel_regularizer
        ))
        model.add(keras.layers.MaxPooling2D(2))

        model.add(keras.layers.Conv2D(
            128, 3, activation='elu', padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=kernel_regularizer
        ))
        model.add(keras.layers.MaxPooling2D(2))

        model.add(keras.layers.Conv2D(
            256, 3, activation='elu', padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=kernel_regularizer
        ))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='elu', kernel_initializer=initializer))

        # Apply dropout if the regularization type is 'Dropout' and add Dropout layer
        if reg_name == 'Dropout':
            model.add(keras.layers.Dropout(reg))  # Use dropout rate from the grid
            model.add(keras.layers.Dense(128, activation='elu', kernel_initializer=initializer))
            model.add(keras.layers.Dropout(reg))  # Second dropout layer if required
        else:
            model.add(keras.layers.Dense(128, activation='elu', kernel_initializer=initializer))

        # Final output layer
        model.add(keras.layers.Dense(10, activation="softmax"))

        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        # Train the model
        history = model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid), verbose=0)

        # Evaluate the model
        val_accuracy = round(max(history.history['val_accuracy']), 5)
        test_loss, test_accuracy = map(lambda x: round(x, 5), model.evaluate(X_test, y_test, verbose=0))

        # Save model and results to best_models
        model_results.append({
            'Regularization Method': reg_name,
            'Lambda': reg if reg_name != 'None' else 'None',
            'Test Accuracy': test_accuracy,
            'Test Loss': test_loss,
            'Validation Accuracy': val_accuracy
        })

        # Print the results
        print(f"Regularization Method: {reg_name}, Lambda: {reg if reg_name != 'None' else 'None'}")
        print(f"Validation Accuracy: {val_accuracy:.5f}")
        print(f"Test Accuracy: {test_accuracy:.5f}")
        print(f"Test Loss: {test_loss:.5f}")
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()
        print("---------------------------------------------------------------------------")

# Sort the models by Validation Accuracy in descending order
model_results = sorted(model_results, key=lambda x: x['Validation Accuracy'], reverse=True)

# Print the top 3 models based on Test Accuracy
print("\nTop 3 models based on Test Accuracy:")
for i, model_info in enumerate(model_results[:3], 1):
    print(f"Model {i}")
    print(f"Regularization Method: {model_info['Regularization Method']}")
    print(f"Lambda: {model_info['Lambda']}")
    print(f"Validation Accuracy: {model_info['Validation Accuracy']:.5f}")
    print(f"Test Accuracy: {model_info['Test Accuracy']:.5f}")
    print(f"Test Loss: {model_info['Test Loss']:.5f}")
    print("---------------------------------------------------------------------------")


# %% [markdown]
# ###Grid Search to find the Top 3 best models for CNN

# %%
keras.utils.set_random_seed(1)

# Grid
model_configs = {
    'initializers': {
        'RandomNormal': keras.initializers.RandomNormal(),
        'Orthogonal': keras.initializers.Orthogonal(),
        'VarianceScaling': keras.initializers.VarianceScaling()
    },
    'activation_functions': {
        'relu': 'relu',
        'leakyrelu': keras.layers.LeakyReLU(),
        'elu': 'elu'
    },
    'optimizers': {
        'adam': [keras.optimizers.Adam, [0.001]],
        'rmsprop': [keras.optimizers.RMSprop, [0.001, 0.01]]
    },
    'regularizations': {
        'Dropout': [0.2, 0.4],
        'None': [None]
    }
}

model_results = []

# Perform grid search over configurations
for init_name, initializer in model_configs['initializers'].items():
    for act_name, activation in model_configs['activation_functions'].items():
        for opt_name, (optimizer_fn, lr_list) in model_configs['optimizers'].items():
            for lr in lr_list:
                for reg_name, reg_values in model_configs['regularizations'].items():
                    for reg in reg_values:

                        # Define model structure
                        model = keras.models.Sequential([
                            keras.layers.Input(shape=[28, 28, 1]),
                            keras.layers.Conv2D(64, 3, activation=activation, padding="same", kernel_initializer=initializer),
                            keras.layers.MaxPooling2D(2),
                            keras.layers.Conv2D(128, 3, activation=activation, padding="same", kernel_initializer=initializer),
                            keras.layers.MaxPooling2D(2),
                            keras.layers.Conv2D(256, 3, activation=activation, padding="same", kernel_initializer=initializer),
                            keras.layers.Flatten(),
                            keras.layers.Dense(256, activation=activation, kernel_initializer=initializer),
                            keras.layers.Dropout(reg if reg_name == 'Dropout' else 0),
                            keras.layers.Dense(128, activation=activation, kernel_initializer=initializer),
                            keras.layers.Dropout(reg if reg_name == 'Dropout' else 0),
                            keras.layers.Dense(10, activation="softmax")
                        ])

                        # Compile the model
                        optimizer = optimizer_fn(learning_rate=lr)
                        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

                        # Train the model
                        history = model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid), verbose=0)

                        # Evaluate model
                        val_accuracy = round(max(history.history['val_accuracy']), 5)
                        test_loss, test_accuracy = map(lambda x: round(x, 5), model.evaluate(X_test, y_test, verbose=0))

                        # Store results
                        model_results.append({
                            'Initializer': init_name,
                            'Activation': act_name,
                            'Optimizer': opt_name,
                            'Learning Rate': lr,
                            'Regularization Method': reg_name,
                            'Lambda': reg,
                            'Test Accuracy': test_accuracy,
                            'Test Loss': test_loss,
                            'Validation Accuracy': val_accuracy
                        })

                        # Output results and plot training process
                        print(f"Initializer: {init_name}, Activation: {act_name}, Optimizer: {opt_name} ({lr}), "
                              f"Regularization Method: {reg_name}, Lambda: {reg}")
                        print(f"Validation Accuracy: {val_accuracy:.5f}")
                        print(f"Test Accuracy: {test_accuracy:.5f}")
                        print(f"Test Loss: {test_loss:.5f}")
                        pd.DataFrame(history.history).plot(figsize=(8, 5))
                        plt.grid(True)
                        plt.gca().set_ylim(0, 1)
                        plt.title(f"{init_name}-{act_name}-{opt_name}-{lr}-{reg_name}")
                        plt.show()
                        print("---------------------------------------------------------------------------")

# Sort and display top 3 models based on validation accuracy
model_results = sorted(model_results, key=lambda x: x['Validation Accuracy'], reverse=True)

print("\nTop 3 models based on Validation Accuracy:")
for i, model_info in enumerate(model_results[:3], 1):
    print(f"Model {i}")
    print(f"Initializer: {model_info['Initializer']}")
    print(f"Activation: {model_info['Activation']}")
    print(f"Optimizer: {model_info['Optimizer']} (Learning Rate: {model_info['Learning Rate']})")
    print(f"Regularization Method: {model_info['Regularization Method']}")
    print(f"Lambda: {model_info['Lambda']}")
    print(f"Validation Accuracy: {model_info['Validation Accuracy']:.5f}")
    print(f"Test Accuracy: {model_info['Test Accuracy']:.5f}")
    print(f"Test Loss: {model_info['Test Loss']:.5f}")
    print("---------------------------------------------------------------------------")


# %% [markdown]
# #Prepare the CIFAR dataset

# %% [markdown]
# ###MLP

# %%
keras.utils.set_random_seed(1)

#Define input shape for CIFAR-10
input_shape = (32, 32, 3)
num_classes = 10  # CIFAR-10 has 10 classes

#Define function to create a model with specific parameters
def create_model(initializer, activation, optimizer_fn):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(300, activation=activation, kernel_initializer=initializer))
    model.add(keras.layers.Dense(100, activation=activation, kernel_initializer=initializer))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    optimizer = optimizer_fn()
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

#Model configurations based on top 3 results
top_models = [
    {
        "Initializer": keras.initializers.GlorotNormal(),
        "Activation": "relu",
        "Optimizer": keras.optimizers.Nadam(learning_rate=0.001),
        "Validation Accuracy": 0.89050,
        "Test Accuracy": 0.85850,
        "Test Loss": 57.57814
    },
    {
        "Initializer": keras.initializers.GlorotNormal(),
        "Activation": "relu",
        "Optimizer": keras.optimizers.Adagrad(learning_rate=0.1),
        "Validation Accuracy": 0.88983,
        "Test Accuracy": 0.84600,
        "Test Loss": 59.07931
    },
    {
        "Initializer": keras.initializers.GlorotNormal(),
        "Activation": keras.layers.ELU(),
        "Optimizer": keras.optimizers.Nadam(learning_rate=0.001),
        "Validation Accuracy": 0.88867,
        "Test Accuracy": 0.80390,
        "Test Loss": 63.75527
    }
]

#Prepare CIFAR-10 data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

#Training each model and evaluating on CIFAR-10
for i, config in enumerate(top_models, 1):
    print(f"Training Model {i} with Initializer: {config['Initializer']}, Activation: {config['Activation']}, Optimizer: {config['Optimizer'].__class__.__name__}")
    model = create_model(config['Initializer'], config['Activation'], lambda: config['Optimizer'])

    #Train and evaluate model
    history = model.fit(X_train, y_train, epochs=30, validation_split=0, verbose=1)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    #Output results
    print(f"Model {i}")
    print(f"Test Accuracy: {test_accuracy:.5f}")
    print(f"Test Loss: {test_loss:.5f}")
    print("---------------------------------------------------------------------------")

# %% [markdown]
# ###CNN
# 

# %%
# Set random seed for reproducibility
keras.utils.set_random_seed(1)

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train, X_test = X_train.astype('float32') / 255.0, X_test.astype('float32') / 255.0

# Grid configurations for initializers, activations, optimizers, and regularization
model_configs = {
    'initializers': {
        'Orthogonal': keras.initializers.Orthogonal(),
        'VarianceScaling': keras.initializers.VarianceScaling(),
        'RandomNormal': keras.initializers.RandomNormal()
    },
    'activation_functions': {
        'relu': 'relu'
    },
    'optimizers': {
        'adam': [keras.optimizers.Adam, [0.001]]
    },
    'regularizations': {
        'None': [None]
    }
}

# Top 3 models based on Validation Accuracy
top_models = [
    {
        'Initializer': 'Orthogonal',
        'Activation': 'relu',
        'Optimizer': 'adam',
        'Regularization Method': 'None',
        'Lambda': None
    },
    {
        'Initializer': 'VarianceScaling',
        'Activation': 'relu',
        'Optimizer': 'adam',
        'Regularization Method': 'None',
        'Lambda': None
    },
    {
        'Initializer': 'RandomNormal',
        'Activation': 'relu',
        'Optimizer': 'adam',
        'Regularization Method': 'None',
        'Lambda': None
    }
]

# Construct, compile, and evaluate each model
model_results = []
for model_info in top_models:
    # Retrieve model parameters
    init_name = model_info['Initializer']
    activation = model_info['Activation']
    opt_name = model_info['Optimizer']
    reg_name = model_info['Regularization Method']
    reg = model_info['Lambda']

    # Define model structure
    model = keras.models.Sequential([
        keras.layers.Input(shape=[32, 32, 3]),  # CIFAR-10 images are 32x32 with 3 color channels
        keras.layers.Conv2D(64, 3, activation=activation, padding="same", kernel_initializer=model_configs['initializers'][init_name]),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation=activation, padding="same", kernel_initializer=model_configs['initializers'][init_name]),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation=activation, padding="same", kernel_initializer=model_configs['initializers'][init_name]),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation=activation, kernel_initializer=model_configs['initializers'][init_name]),
        keras.layers.Dense(128, activation=activation, kernel_initializer=model_configs['initializers'][init_name]),
        keras.layers.Dense(10, activation="softmax")
    ])

    # Compile the model
    optimizer = model_configs['optimizers'][opt_name][0](learning_rate=model_configs['optimizers'][opt_name][1][0])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Train the model (no validation data)
    history = model.fit(X_train, y_train, epochs=5, verbose=0, validation_split=0)

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Store results
    model_results.append({
        'Initializer': init_name,
        'Activation': activation,
        'Optimizer': opt_name,
        'Regularization Method': reg_name,
        'Lambda': reg,
        'Test Accuracy': round(test_accuracy, 5),
        'Test Loss': round(test_loss, 5)
    })

    # Output results
    print(f"Initializer: {init_name}, Activation: {activation}, Optimizer: {opt_name}, "
          f"Regularization Method: {reg_name}, Lambda: {reg}")
    print(f"Test Accuracy: {test_accuracy:.5f}")
    print(f"Test Loss: {test_loss:.5f}")
    print("---------------------------------------------------------------------------")

# Sort and display the results
model_results = sorted(model_results, key=lambda x: x['Test Accuracy'], reverse=True)

# Print top models
print("\nTop models based on Test Accuracy:")
for i, model_info in enumerate(model_results, 1):
    print(f"Model {i}")
    print(f"Initializer: {model_info['Initializer']}")
    print(f"Activation: {model_info['Activation']}")
    print(f"Optimizer: {model_info['Optimizer']}")
    print(f"Regularization Method: {model_info['Regularization Method']}")
    print(f"Lambda: {model_info['Lambda']}")
    print(f"Test Accuracy: {model_info['Test Accuracy']:.5f}")
    print(f"Test Loss: {model_info['Test Loss']:.5f}")
    print("---------------------------------------------------------------------------")




