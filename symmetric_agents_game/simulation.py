from visual_module import *
from agent import *
import random
import os
from keras.layers import Dense

image_feature_size = 2048  # You'd need to specify the exact size based on your dataset

linear_layer = Dense(image_feature_size, use_bias=True, kernel_initializer='glorot_uniform')

batch_size = 96

training_image_directory = './ImagesGenerated/for_training'
testing_image_directory = './ImagesGenerated/for_testing'
validation_image_directory = './ImagesGenerated/for_validation'

def tensorflow_to_pytorch(tensor):
    # Convert TensorFlow tensor to numpy array
    numpy_array = tensor.numpy()

    # Convert numpy array to PyTorch tensor
    pytorch_tensor = torch.from_numpy(numpy_array).float()

    # Reorder the channels if necessary (from HxWxC to CxHxW)
    pytorch_tensor = pytorch_tensor.permute(0, 3, 1, 2)

    return pytorch_tensor

def referential_game():
    num_agents = 4
    pairs = [CommunicationPair() for _ in range(num_agents)]
    group_1 = pairs[:2]
    group_2 = pairs[2:]
    print("Phase 1: Agents interacting within their groups")
    training(group_1, 2)
    training(group_2, 2)
    print("Phase 2:Agents interacting across groups")
    training(pairs, 4)

def simulate_interaction_batch(batch_size, image_directory):
    # Define the directory where images for training are located.
    # image_directory = '/Users/keweitao/PycharmProjects/LanguageEvolutionSimulation/ImagesGenerated/for_training'

    # List all labels (i.e., sub-directories) within the main image directory.
    labels = [label for label in os.listdir(image_directory) if os.path.isdir(os.path.join(image_directory, label))]

    # Randomly select target labels for the given batch size.
    target_label_indices = np.random.choice(len(labels), size=batch_size)

    # For each target label, select two confounder labels that are different from the target.
    confounder_label_indices = []
    for target_idx in target_label_indices:
        available_indices = [idx for idx in range(len(labels)) if idx != target_idx]
        confounder_idxs = np.random.choice(available_indices, size=2, replace=False)
        confounder_label_indices.append([confounder_idxs[0], confounder_idxs[1]])

    confounder_label_indices = np.array(confounder_label_indices)

    tensor_indicating_location_list = []

    # Accumulators for tensors to be batch processed
    target_image_tensors_list = []
    confounder_image_tensors_list = []

    for target_idx, confounder_idxs in zip(target_label_indices, confounder_label_indices):
        # Retrieve labels using the randomly chosen indices.
        target_label = labels[target_idx]
        confounder_labels = [labels[idx] for idx in confounder_idxs]

        # Get the directory for the target label and list all images inside it.
        target_label_directory = os.path.join(image_directory, target_label)
        target_label_images = os.listdir(target_label_directory)
        target_image_filename = random.choice(target_label_images)
        target_image_path = os.path.join(target_label_directory, target_image_filename)

        # Convert target image to tensor and append to list.
        target_image_tensor = tf.keras.preprocessing.image.img_to_array(
            tf.keras.preprocessing.image.load_img(target_image_path))
        target_image_tensors_list.append(target_image_tensor)

        # Add the target label to the confounders list and shuffle to get a randomized order.
        confounder_labels.append(target_label)
        random.shuffle(confounder_labels)

        for label in confounder_labels:
            # Get confounder image path.
            confounder_label_directory = os.path.join(image_directory, label)
            confounder_label_images = os.listdir(confounder_label_directory)
            confounder_image_filename = random.choice(confounder_label_images)
            confounder_image_path = os.path.join(confounder_label_directory, confounder_image_filename)

            # Convert image to tensor and append to list.
            confounder_image_tensor = tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(confounder_image_path))
            confounder_image_tensors_list.append(confounder_image_tensor)

        # Determine the index of the target label post-shuffling.
        target_label_index = confounder_labels.index(target_label)
        # Convert the index into a one-hot encoded tensor.
        tensor_indicating_location_single = tf.one_hot(target_label_index, len(confounder_labels))
        tensor_indicating_location_list.append(tensor_indicating_location_single.numpy())

    # Convert list of image tensors to a single tensor and process.
    target_image_features = process_image(tensorflow_to_pytorch(tf.convert_to_tensor(target_image_tensors_list)))
    confounder_image_features = process_image(tensorflow_to_pytorch(tf.convert_to_tensor(confounder_image_tensors_list)))

    tensor_indicating_location = np.array(tensor_indicating_location_list)

    confounder_image_features = tf.reshape(confounder_image_features, [batch_size, 3, image_feature_size])

    return target_image_features, confounder_image_features, tensor_indicating_location

def zst_simulate_interaction_batch(batch_size, target_image_directory, confounder_image_directory):
    # List all labels (i.e., sub-directories) within the target and confounder image directories.
    target_labels = [label for label in os.listdir(target_image_directory) if os.path.isdir(os.path.join(target_image_directory, label))]
    confounder_labels = [label for label in os.listdir(confounder_image_directory) if os.path.isdir(os.path.join(confounder_image_directory, label))]

    # Randomly select target labels for the given batch size.
    target_label_indices = np.random.choice(len(target_labels), size=batch_size)

    tensor_indicating_location_list = []
    target_image_tensors_list = []
    confounder_image_tensors_list = []

    for target_idx in target_label_indices:
        # Retrieve target label using the randomly chosen index.
        target_label = target_labels[target_idx]

        # Get the directory for the target label and list all images inside it.
        target_label_directory = os.path.join(target_image_directory, target_label)
        target_label_images = os.listdir(target_label_directory)
        target_image_filename = random.choice(target_label_images)
        target_image_path = os.path.join(target_label_directory, target_image_filename)

        # Convert target image to tensor and append to list.
        target_image_tensor = tf.keras.preprocessing.image.img_to_array(
            tf.keras.preprocessing.image.load_img(target_image_path))
        target_image_tensors_list.append(target_image_tensor)

        # Randomly select two confounder labels.
        selected_confounder_labels = np.random.choice(confounder_labels, size=2, replace=False).tolist()

        # Add the target label to the confounders list and shuffle to get a randomized order.
        all_labels = selected_confounder_labels + [target_label]
        random.shuffle(all_labels)

        for label in all_labels:
            # Get image path based on whether it's a target or confounder.
            if label == target_label:
                image_directory = target_image_directory
            else:
                image_directory = confounder_image_directory

            label_directory = os.path.join(image_directory, label)
            label_images = os.listdir(label_directory)
            image_filename = random.choice(label_images)
            image_path = os.path.join(label_directory, image_filename)

            # Convert image to tensor and append to list.
            image_tensor = tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(image_path))
            confounder_image_tensors_list.append(image_tensor)

        # Determine the index of the target label post-shuffling.
        target_label_index = all_labels.index(target_label)
        # Convert the index into a one-hot encoded tensor.
        tensor_indicating_location_single = tf.one_hot(target_label_index, len(all_labels))
        tensor_indicating_location_list.append(tensor_indicating_location_single.numpy())

     # Convert list of image tensors to a single tensor and process.
    target_image_features = process_image(tensorflow_to_pytorch(tf.convert_to_tensor(target_image_tensors_list)))
    confounder_image_features = process_image(
    tensorflow_to_pytorch(tf.convert_to_tensor(confounder_image_tensors_list)))
    tensor_indicating_location = np.array(tensor_indicating_location_list)
    confounder_image_features = tf.reshape(confounder_image_features, [batch_size, 3, image_feature_size])

    return target_image_features, confounder_image_features, tensor_indicating_location

def training(pairs, num_simulation_epochs):
    # num_simulation_epochs = 6
    num_training_steps_per_epoch = 36
    val_num_training_steps_per_epoch = 18

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00002)

    for simulation_epoch in range(num_simulation_epochs):
        cumulative_loss = 0.0
        for _ in range(num_training_steps_per_epoch):
            sender_agent, receiver_agent = select_agents(pairs)
            target_image_features, confounder_image_features, label = simulate_interaction_batch(batch_size, training_image_directory)

            with tf.GradientTape(persistent=True) as tape:
                sender_output = sender_agent(target_image_features)
                # tf.debugging.assert_all_finite(sender_output, message="sender_output has NaN or Inf values!")

                receiver_decisions = receiver_agent(sender_output)
                # tf.debugging.assert_all_finite(receiver_decisions, message="receiver_decisions has NaN or Inf values!")

                loss = compute_batch_loss(label, tf.stack(receiver_decisions, axis=0), confounder_image_features)
                cumulative_loss += loss.numpy()

                sender_grads = tape.gradient(loss, sender_agent.trainable_variables)
                receiver_grads = tape.gradient(loss, receiver_agent.trainable_variables)
                optimizer.apply_gradients(zip(receiver_grads, receiver_agent.trainable_variables))
                optimizer.apply_gradients(zip(sender_grads, sender_agent.trainable_variables))

            del tape
        average_loss = cumulative_loss / num_training_steps_per_epoch
        print(f"Simulation Epoch {simulation_epoch + 1}/{num_simulation_epochs}, Average Loss: {average_loss}")
        # Compute Validation Loss
        val_cumulative_loss = 0.0
        for _ in range(val_num_training_steps_per_epoch):
            sender_agent, receiver_agent = select_agents(pairs)
            target_image_features, confounder_image_features, val_label = simulate_interaction_batch(batch_size, testing_image_directory)
            val_sender_output = sender_agent(target_image_features, training=False)
            val_receiver_decisions = receiver_agent(val_sender_output, training=False)
            val_loss = compute_batch_loss(val_label, tf.stack(val_receiver_decisions, axis=0), confounder_image_features)
            val_cumulative_loss += val_loss.numpy()
        val_average_loss = val_cumulative_loss / val_num_training_steps_per_epoch
        print(f"Validation Loss after Epoch {simulation_epoch + 1}: {val_average_loss}")

    zst_val_cumulative_loss = 0.0
    for _ in range(val_num_training_steps_per_epoch):
        sender_agent, receiver_agent = select_agents(pairs)
        target_image_features, confounder_image_features, val_label = zst_simulate_interaction_batch(batch_size,
                                                                                                 validation_image_directory, testing_image_directory)
        val_sender_output = sender_agent(target_image_features, training=False)
        val_receiver_decisions = receiver_agent(val_sender_output, training=False)
        val_loss = compute_batch_loss(val_label, tf.stack(val_receiver_decisions, axis=0), confounder_image_features)
        zst_val_cumulative_loss += val_loss.numpy()
    zst_val_average_loss = zst_val_cumulative_loss / val_num_training_steps_per_epoch
    print(f"Zero shot validation loss: {zst_val_average_loss}")


def compute_q(image_features, g_x):
    # Ensure element-wise multiplication across the batch and then sum over the feature dimensions.
    # print("image_feature: ", image_features.shape)
    # print("decision: ", g_x.shape)
    return tf.reduce_sum(image_features * g_x, axis=-1)

def compute_batch_loss(target_location, receiver_decisions, confounder_image_features):
    # 1. Identify target image feature using label and confounder_image_features
    # print("receiver_decisions: ", receiver_decisions.shape)
    target_indices = tf.argmax(target_location, axis=-1)
    target_features = tf.gather_nd(confounder_image_features,
                                   indices=tf.stack([tf.cast(tf.range(target_location.shape[0]), tf.int32),
                                                     tf.cast(target_indices, tf.int32)], axis=-1))
    # print("confounder_image_features", confounder_image_features.shape)

    # 2. Compute q-value for target image feature
    q_target = compute_q(target_features, receiver_decisions)

    # 3. Compute q-values for distractor image features
    mask = tf.one_hot(target_indices, depth=target_location.shape[1], dtype=tf.bool, on_value=False, off_value=True)
    distractor_features = tf.boolean_mask(confounder_image_features, mask)

    # Reshape to get back the original shape but without the target features
    distractor_features = tf.reshape(distractor_features, [-1, target_location.shape[1]-1, receiver_decisions.shape[1]])

    q_distractor_values = [compute_q(distractor_features[:, i, :], receiver_decisions)
                           for i in range(distractor_features.shape[1])]

    q_dk = tf.reduce_sum(tf.stack(q_distractor_values, axis=-1), axis=-1)

    # 4. Compute the loss
    losses = tf.maximum(0., 1 - q_target + q_dk)
    total_loss = tf.reduce_mean(losses)

    return total_loss


def select_agents(pairs):
    # Randomly select two distinct pairs
    pair_indices = np.random.choice(len(pairs), size=2, replace=True)

    sender_from_first_pair = pairs[pair_indices[0]].sender
    receiver_from_second_pair = pairs[pair_indices[1]].receiver

    return sender_from_first_pair, receiver_from_second_pair


referential_game()

