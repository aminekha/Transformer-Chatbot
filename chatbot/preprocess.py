from .utils import *

# Maximum number of samples to preprocess
MAX_SAMPLES = 50000

# Downloads a file from a URL if it not already in the cache
# path_to_zip = "cornell_movie_dialogs_corpus.zip"
path_to_dataset = "chatbot/dataset"

path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
path_to_movie_conversations = os.path.join(path_to_dataset, 'movie_conversations.txt')

def preprocess_sentence(sentence):
  """
    This function takes a sentence as argument and return
    a preprocessed copy of the sentence.
    It remove any unecessary spaces and ponctuation,
    and lower the characters.
  """
  # lower the sentence and remove any unecessary spaces using strip
  sentence = sentence.lower().strip()
  # Create a space between the word and the ponctuation
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  return sentence



def load_conversations():
  """
  This function loads questions and answers, preprocess the sentences,
  and save them to 2 variables questions and answers.
  """
  id2line = {}
  with open(path_to_movie_lines, errors='ignore') as file:
    # read every line from the movie file
    lines = file.readlines()
  for line in lines:
    parts = line.replace('\n', '').split(' +++$+++ ')
    id2line[parts[0]] = parts[4]

  inputs, outputs = [], []
  with open(path_to_movie_conversations, 'r') as file:
    lines = file.readlines()
  for line in lines:
    parts = line.replace('\n', '').split(' +++$+++ ')

    # get conversation in a list of line ID
    conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
    for i in range(len(conversation) - 1):
      inputs.append(preprocess_sentence(id2line[conversation[i]]))
      outputs.append(preprocess_sentence(id2line[conversation[i+1]]))
      if len(inputs) >= MAX_SAMPLES:
        return inputs, outputs
  return inputs, outputs

print("Loading Conversations...")
questions, answers = load_conversations()

print("Creating tokenizer...")
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13
)

# define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size+1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size+2

# Maximum sentence length
MAX_LENGTH = 80


def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []
  
  for (sentence1, sentence2) in zip(inputs,outputs):
    # Tokenize sentence
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    # Check tokenized sentence max length
    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)
  
  # Tokenize the inputs and outputs
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding="post"
  )
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding="post"
  )

  return tokenized_inputs, tokenized_outputs

print("Tokenize and filter...")
questions, answers = tokenize_and_filter(questions, answers)

