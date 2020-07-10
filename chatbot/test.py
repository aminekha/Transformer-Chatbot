from .utils import *
from .preprocess import *


tf.keras.backend.clear_session()

class Chatbot:
  def __init__(self):
    pass

  # Hyper-parameters
  NUM_LAYERS = 2
  D_MODEL = 256
  NUM_HEADS = 8
  UNITS = 512
  DROPOUT = 0.1
  VOCAB_SIZE = 8333

  print("Creating model... ")
  model = transformer(
      vocab_size=VOCAB_SIZE,
      num_layers=NUM_LAYERS,
      units=UNITS,
      d_model=D_MODEL,
      num_heads=NUM_HEADS,
      dropout=DROPOUT)

  print("Model Created.")

  def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


  model.load_weights("chatbot/weights/weights.h5")
  print("Weights Loaded.")


  def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
      predictions = Chatbot.model(inputs=[sentence, output], training=False)

      # select the last word from the seq_len dimension
      predictions = predictions[:, -1:, :]
      predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

      # return the result if the predicted_id is equal to the end token
      if tf.equal(predicted_id, END_TOKEN[0]):
        break

      # concatenated the predicted_id to the output which is given to the decoder
      # as its input.
      output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


  def predict(sentence):
    prediction = Chatbot.evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence


# chatbot = Chatbot()

# while True:
#   my_input = input("> ")
#   output = chatbot.predict(my_input)
#   if my_input == "goodbye":
#     break