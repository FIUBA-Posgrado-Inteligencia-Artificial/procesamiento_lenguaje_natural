import collections
import json
import warnings
import math

import numpy as np

import torch
from torch import nn


'''
Funciones de ayuda tomadas de la libreria de Keras
'''
def text_to_word_sequence(input_text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True,
                          split=' '):
  r"""Converts a text to a sequence of words (or tokens).
  Deprecated: `tf.keras.preprocessing.text.text_to_word_sequence` does not
  operate on tensors and is not recommended for new code. Prefer
  `tf.strings.regex_replace` and `tf.strings.split` which provide equivalent
  functionality and accept `tf.Tensor` input. For an overview of text handling
  in Tensorflow, see the [text loading tutorial]
  (https://www.tensorflow.org/tutorials/load_data/text).
  This function transforms a string of text into a list of words
  while ignoring `filters` which include punctuations by default.
  >>> sample_text = 'This is a sample sentence.'
  >>> tf.keras.preprocessing.text.text_to_word_sequence(sample_text)
  ['this', 'is', 'a', 'sample', 'sentence']
  Args:
      input_text: Input text (string).
      filters: list (or concatenation) of characters to filter out, such as
          punctuation. Default: ``'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n'``,
            includes basic punctuation, tabs, and newlines.
      lower: boolean. Whether to convert the input to lowercase.
      split: str. Separator for word splitting.
  Returns:
      A list of words (or tokens).
  """
  if lower:
    input_text = input_text.lower()

  translate_dict = {c: split for c in filters}
  translate_map = str.maketrans(translate_dict)
  input_text = input_text.translate(translate_map)

  seq = input_text.split(split)
  return [i for i in seq if i]

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
  """Pads sequences to the same length.
  This function transforms a list (of length `num_samples`)
  of sequences (lists of integers)
  into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
  `num_timesteps` is either the `maxlen` argument if provided,
  or the length of the longest sequence in the list.
  Sequences that are shorter than `num_timesteps`
  are padded with `value` until they are `num_timesteps` long.
  Sequences longer than `num_timesteps` are truncated
  so that they fit the desired length.
  The position where padding or truncation happens is determined by
  the arguments `padding` and `truncating`, respectively.
  Pre-padding or removing values from the beginning of the sequence is the
  default.
  >>> sequence = [[1], [2, 3], [4, 5, 6]]
  >>> tf.keras.preprocessing.sequence.pad_sequences(sequence)
  array([[0, 0, 1],
         [0, 2, 3],
         [4, 5, 6]], dtype=int32)
  >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, value=-1)
  array([[-1, -1,  1],
         [-1,  2,  3],
         [ 4,  5,  6]], dtype=int32)
  >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post')
  array([[1, 0, 0],
         [2, 3, 0],
         [4, 5, 6]], dtype=int32)
  >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=2)
  array([[0, 1],
         [2, 3],
         [5, 6]], dtype=int32)
  Args:
      sequences: List of sequences (each sequence is a list of integers).
      maxlen: Optional Int, maximum length of all sequences. If not provided,
          sequences will be padded to the length of the longest individual
          sequence.
      dtype: (Optional, defaults to `"int32"`). Type of the output sequences.
          To pad sequences with variable length strings, you can use `object`.
      padding: String, "pre" or "post" (optional, defaults to `"pre"`):
          pad either before or after each sequence.
      truncating: String, "pre" or "post" (optional, defaults to `"pre"`):
          remove values from sequences larger than
          `maxlen`, either at the beginning or at the end of the sequences.
      value: Float or String, padding value. (Optional, defaults to 0.)
  Returns:
      Numpy array with shape `(len(sequences), maxlen)`
  Raises:
      ValueError: In case of invalid values for `truncating` or `padding`,
          or in case of invalid shape for a `sequences` entry.
  """
  if not hasattr(sequences, '__len__'):
    raise ValueError('`sequences` must be iterable.')
  num_samples = len(sequences)

  lengths = []
  sample_shape = ()
  flag = True

  # take the sample shape from the first non empty sequence
  # checking for consistency in the main loop below.

  for x in sequences:
    try:
      lengths.append(len(x))
      if flag and len(x):
        sample_shape = np.asarray(x).shape[1:]
        flag = False
    except TypeError as e:
      raise ValueError('`sequences` must be a list of iterables. '
                       f'Found non-iterable: {str(x)}') from e

  if maxlen is None:
    maxlen = np.max(lengths)

  is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(
      dtype, np.unicode_)
  if isinstance(value, str) and dtype != object and not is_dtype_str:
    raise ValueError(
        f'`dtype` {dtype} is not compatible with `value`\'s type: '
        f'{type(value)}\nYou should set `dtype=object` for variable length '
        'strings.')

  x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
  for idx, s in enumerate(sequences):
    if not len(s):  # pylint: disable=g-explicit-length-test
      continue  # empty list/array was found
    if truncating == 'pre':
      trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
    elif truncating == 'post':
      trunc = s[:maxlen]
    else:
      raise ValueError(f'Truncating type "{truncating}" not understood')

    # check `trunc` has expected shape
    trunc = np.asarray(trunc, dtype=dtype)
    if trunc.shape[1:] != sample_shape:
      raise ValueError(f'Shape of sample {trunc.shape[1:]} of sequence at '
                       f'position {idx} is different from expected shape '
                       f'{sample_shape}')

    if padding == 'post':
      x[idx, :len(trunc)] = trunc
    elif padding == 'pre':
      x[idx, -len(trunc):] = trunc
    else:
      raise ValueError(f'Padding type "{padding}" not understood')
  return x

class Tokenizer(object):
  """Text tokenization utility class.
  Deprecated: `tf.keras.preprocessing.text.Tokenizer` does not operate on
  tensors and is not recommended for new code. Prefer
  `tf.keras.layers.TextVectorization` which provides equivalent functionality
  through a layer which accepts `tf.Tensor` input. See the
  [text loading tutorial](https://www.tensorflow.org/tutorials/load_data/text)
  for an overview of the layer and text handling in tensorflow.
  This class allows to vectorize a text corpus, by turning each
  text into either a sequence of integers (each integer being the index
  of a token in a dictionary) or into a vector where the coefficient
  for each token could be binary, based on word count, based on tf-idf...
  By default, all punctuation is removed, turning the texts into
  space-separated sequences of words
  (words maybe include the `'` character). These sequences are then
  split into lists of tokens. They will then be indexed or vectorized.
  `0` is a reserved index that won't be assigned to any word.
  Args:
      num_words: the maximum number of words to keep, based
          on word frequency. Only the most common `num_words-1` words will
          be kept.
      filters: a string where each element is a character that will be
          filtered from the texts. The default is all punctuation, plus
          tabs and line breaks, minus the `'` character.
      lower: boolean. Whether to convert the texts to lowercase.
      split: str. Separator for word splitting.
      char_level: if True, every character will be treated as a token.
      oov_token: if given, it will be added to word_index and used to
          replace out-of-vocabulary words during text_to_sequence calls
      analyzer: function. Custom analyzer to split the text.
          The default analyzer is text_to_word_sequence
  """

  def __init__(self,
               num_words=None,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
               lower=True,
               split=' ',
               char_level=False,
               oov_token=None,
               analyzer=None,
               **kwargs):
    # Legacy support
    if 'nb_words' in kwargs:
      warnings.warn('The `nb_words` argument in `Tokenizer` '
                    'has been renamed `num_words`.')
      num_words = kwargs.pop('nb_words')
    document_count = kwargs.pop('document_count', 0)
    if kwargs:
      raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    self.word_counts = collections.OrderedDict()
    self.word_docs = collections.defaultdict(int)
    self.filters = filters
    self.split = split
    self.lower = lower
    self.num_words = num_words
    self.document_count = document_count
    self.char_level = char_level
    self.oov_token = oov_token
    self.index_docs = collections.defaultdict(int)
    self.word_index = {}
    self.index_word = {}
    self.analyzer = analyzer

  def fit_on_texts(self, texts):
    """Updates internal vocabulary based on a list of texts.
    In the case where texts contains lists,
    we assume each entry of the lists to be a token.
    Required before using `texts_to_sequences` or `texts_to_matrix`.
    Args:
        texts: can be a list of strings,
            a generator of strings (for memory-efficiency),
            or a list of list of strings.
    """
    for text in texts:
      self.document_count += 1
      if self.char_level or isinstance(text, list):
        if self.lower:
          if isinstance(text, list):
            text = [text_elem.lower() for text_elem in text]
          else:
            text = text.lower()
        seq = text
      else:
        if self.analyzer is None:
          seq = text_to_word_sequence(
              text, filters=self.filters, lower=self.lower, split=self.split)
        else:
          seq = self.analyzer(text)
      for w in seq:
        if w in self.word_counts:
          self.word_counts[w] += 1
        else:
          self.word_counts[w] = 1
      for w in set(seq):
        # In how many documents each word occurs
        self.word_docs[w] += 1

    wcounts = list(self.word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    # forcing the oov_token to index 1 if it exists
    if self.oov_token is None:
      sorted_voc = []
    else:
      sorted_voc = [self.oov_token]
    sorted_voc.extend(wc[0] for wc in wcounts)

    # note that index 0 is reserved, never assigned to an existing word
    self.word_index = dict(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))

    self.index_word = {c: w for w, c in self.word_index.items()}

    for w, c in list(self.word_docs.items()):
      self.index_docs[self.word_index[w]] = c

  def fit_on_sequences(self, sequences):
    """Updates internal vocabulary based on a list of sequences.
    Required before using `sequences_to_matrix`
    (if `fit_on_texts` was never called).
    Args:
        sequences: A list of sequence.
            A "sequence" is a list of integer word indices.
    """
    self.document_count += len(sequences)
    for seq in sequences:
      seq = set(seq)
      for i in seq:
        self.index_docs[i] += 1

  def texts_to_sequences(self, texts):
    """Transforms each text in texts to a sequence of integers.
    Only top `num_words-1` most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.
    Args:
        texts: A list of texts (strings).
    Returns:
        A list of sequences.
    """
    return list(self.texts_to_sequences_generator(texts))

  def texts_to_sequences_generator(self, texts):
    """Transforms each text in `texts` to a sequence of integers.
    Each item in texts can also be a list,
    in which case we assume each item of that list to be a token.
    Only top `num_words-1` most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.
    Args:
        texts: A list of texts (strings).
    Yields:
        Yields individual sequences.
    """
    num_words = self.num_words
    oov_token_index = self.word_index.get(self.oov_token)
    for text in texts:
      if self.char_level or isinstance(text, list):
        if self.lower:
          if isinstance(text, list):
            text = [text_elem.lower() for text_elem in text]
          else:
            text = text.lower()
        seq = text
      else:
        if self.analyzer is None:
          seq = text_to_word_sequence(
              text, filters=self.filters, lower=self.lower, split=self.split)
        else:
          seq = self.analyzer(text)
      vect = []
      for w in seq:
        i = self.word_index.get(w)
        if i is not None:
          if num_words and i >= num_words:
            if oov_token_index is not None:
              vect.append(oov_token_index)
          else:
            vect.append(i)
        elif self.oov_token is not None:
          vect.append(oov_token_index)
      yield vect

  def sequences_to_texts(self, sequences):
    """Transforms each sequence into a list of text.
    Only top `num_words-1` most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.
    Args:
        sequences: A list of sequences (list of integers).
    Returns:
        A list of texts (strings)
    """
    return list(self.sequences_to_texts_generator(sequences))

  def sequences_to_texts_generator(self, sequences):
    """Transforms each sequence in `sequences` to a list of texts(strings).
    Each sequence has to a list of integers.
    In other words, sequences should be a list of sequences
    Only top `num_words-1` most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.
    Args:
        sequences: A list of sequences.
    Yields:
        Yields individual texts.
    """
    num_words = self.num_words
    oov_token_index = self.word_index.get(self.oov_token)
    for seq in sequences:
      vect = []
      for num in seq:
        word = self.index_word.get(num)
        if word is not None:
          if num_words and num >= num_words:
            if oov_token_index is not None:
              vect.append(self.index_word[oov_token_index])
          else:
            vect.append(word)
        elif self.oov_token is not None:
          vect.append(self.index_word[oov_token_index])
      vect = ' '.join(vect)
      yield vect

  def texts_to_matrix(self, texts, mode='binary'):
    """Convert a list of texts to a Numpy matrix.
    Args:
        texts: list of strings.
        mode: one of "binary", "count", "tfidf", "freq".
    Returns:
        A Numpy matrix.
    """
    sequences = self.texts_to_sequences(texts)
    return self.sequences_to_matrix(sequences, mode=mode)

  def sequences_to_matrix(self, sequences, mode='binary'):
    """Converts a list of sequences into a Numpy matrix.
    Args:
        sequences: list of sequences
            (a sequence is a list of integer word indices).
        mode: one of "binary", "count", "tfidf", "freq"
    Returns:
        A Numpy matrix.
    Raises:
        ValueError: In case of invalid `mode` argument,
            or if the Tokenizer requires to be fit to sample data.
    """
    if not self.num_words:
      if self.word_index:
        num_words = len(self.word_index) + 1
      else:
        raise ValueError('Specify a dimension (`num_words` argument), '
                         'or fit on some text data first.')
    else:
      num_words = self.num_words

    if mode == 'tfidf' and not self.document_count:
      raise ValueError('Fit the Tokenizer on some data '
                       'before using tfidf mode.')

    x = np.zeros((len(sequences), num_words))
    for i, seq in enumerate(sequences):
      if not seq:
        continue
      counts = collections.defaultdict(int)
      for j in seq:
        if j >= num_words:
          continue
        counts[j] += 1
      for j, c in list(counts.items()):
        if mode == 'count':
          x[i][j] = c
        elif mode == 'freq':
          x[i][j] = c / len(seq)
        elif mode == 'binary':
          x[i][j] = 1
        elif mode == 'tfidf':
          # Use weighting scheme 2 in
          # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
          tf = 1 + np.log(c)
          idf = np.log(1 + self.document_count /
                       (1 + self.index_docs.get(j, 0)))
          x[i][j] = tf * idf
        else:
          raise ValueError('Unknown vectorization mode:', mode)
    return x

  def get_config(self):
    """Returns the tokenizer configuration as Python dictionary.
    The word count dictionaries used by the tokenizer get serialized
    into plain JSON, so that the configuration can be read by other
    projects.
    Returns:
        A Python dictionary with the tokenizer configuration.
    """
    json_word_counts = json.dumps(self.word_counts)
    json_word_docs = json.dumps(self.word_docs)
    json_index_docs = json.dumps(self.index_docs)
    json_word_index = json.dumps(self.word_index)
    json_index_word = json.dumps(self.index_word)

    return {
        'num_words': self.num_words,
        'filters': self.filters,
        'lower': self.lower,
        'split': self.split,
        'char_level': self.char_level,
        'oov_token': self.oov_token,
        'document_count': self.document_count,
        'word_counts': json_word_counts,
        'word_docs': json_word_docs,
        'index_docs': json_index_docs,
        'index_word': json_index_word,
        'word_index': json_word_index
    }

  def to_json(self, **kwargs):
    """Returns a JSON string containing the tokenizer configuration.
    To load a tokenizer from a JSON string, use
    `keras.preprocessing.text.tokenizer_from_json(json_string)`.
    Args:
        **kwargs: Additional keyword arguments
            to be passed to `json.dumps()`.
    Returns:
        A JSON string containing the tokenizer configuration.
    """
    config = self.get_config()
    tokenizer_config = {'class_name': self.__class__.__name__, 'config': config}
    return json.dumps(tokenizer_config, **kwargs)


class CustomRNN(nn.Module):
    """
    Referencia:
    http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    """
    def __init__(self, input_size, hidden_size, activation=nn.Tanh()):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #i_t (input gate)
        self.W_i = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(self.hidden_size))
        
        self.activation = activation

        self.init_weights()

    def init_weights(self):
        '''
        Inicializar de forma random los pesos
        '''
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        '''
        Esta función trabaja por defecto como si batch_first=True
        '''
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size).to(x.device)
        else:
            h_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]
            h_t = self.activation(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            hidden_seq.append(h_t.unsqueeze(0))

        # reshape hidden_seq for return
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, h_t


class CustomLSTM(nn.Module):
    """
    Referencia:
    https://www.youtube.com/watch?v=yMyBd7iNKho

    Implementaciones similares:
    https://stackoverflow.com/questions/49040180/change-tanh-activation-in-lstm-to-relu
    https://theaisummer.com/understanding-lstm/
    https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983    

    NOTA: La LSTM origianl de pytorch posee un bias adicional "V_f"
    por lo que notará diferencia entre la cantidad de "W" entre esta implementación clásica
    y la de pytoch. Esta implementación es fiel al paper original y a Tensorflow.

    """
    def __init__(self, input_size, hidden_size, activation=nn.Tanh(), recurrent_activation=nn.Sigmoid()):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #i_t (input gate)
        self.W_i = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(self.hidden_size))

        #f_t (forget gate)
        self.W_f = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.U_f = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(self.hidden_size))

        #c_t (cell gate)
        self.W_c = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.U_c = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(self.hidden_size))

        #o_t (output gate)
        self.W_o = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(self.hidden_size))

        self.activation = activation
        self.recurrent_activation = recurrent_activation

        self.init_weights()

    def init_weights(self):
        '''
        Inicializar de forma random los pesos
        '''
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        '''
        Esta función trabaja por defecto como si batch_first=True
        '''
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]
            i_t = self.recurrent_activation(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            f_t = self.recurrent_activation(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            g_t = self.activation(x_t @ self.W_c + h_t @ self.U_c + self.b_c)
            o_t = self.recurrent_activation(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.activation(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        # reshape hidden_seq for return
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
        

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return acc


def categorical_acc(y_pred, y_test):
    y_pred_tag = y_pred.data.max(dim=1,keepdim=True)[1]
    y_test_tag = y_test.data.max(dim=1,keepdim=True)[1]

    correct_results_sum = (y_pred_tag == y_test_tag).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return acc
