import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorboard import program

# Load the model from the .h5 file
model_path = "C:/Users/ansh/OneDrive/Desktop/Research/6. SVM/tb.h5"
custom_objects = {'<Optimizer Class Name>': tf.keras.optimizers.<Optimizer Class Name>}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)


# Create a TensorFlow session and run TensorBoard on it
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)

# Define a function to visualize the model using TensorBoard
def visualize_model(model):
    # Create a writer for TensorBoard
    writer = tf.compat.v1.summary.FileWriter('./logs')
    writer.add_graph(sess.graph)

    # Close the writer and session
    writer.flush()
    writer.close()

    # Start TensorBoard and open it in your web browser
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', './logs'])
    url = tb.launch()
    print(f"TensorBoard is running at {url}")

# Call the visualize_model function with the loaded model
visualize_model(model)
