from keras import backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer

class WeightedSum(MaskedLayer):
    """
    This ``Layer`` takes a matrix of vectors and a vector of row weights, and returns a weighted
    sum of the vectors.  You might use this to get some aggregate sentence representation after
    computing an attention over the sentence, for example.

    Inputs:

    - matrix: ``(batch_size, num_rows, embedding_dim)``, with mask ``(batch_size, num_rows)``
    - vector: ``(batch_size, num_rows)``, mask is ignored

    Outputs:

    - A weighted sum of the rows in the matrix, with shape ``(batch_size, embedding_dim)``, with
      mask=``None``.

    Parameters
    ----------
    use_masking: bool, default=True
        If true, we will apply the input mask to the matrix before doing the weighted sum.  If
        you've computed your vector weights with masking, so that masked entries are 0, this is
        unnecessary, and you can set this parameter to False to avoid an expensive computation.

    Notes
    -----
    You probably should have used a mask when you computed your attention weights, so any row
    that's masked in the matrix `should` already be 0 in the attention vector.  But just in case
    you didn't, we'll handle a mask on the matrix here too.  If you know that you did masking right
    on the attention, you can optionally remove the mask computation here, which will save you a
    bit of time and memory.

    While the above spec shows inputs with 3 and 2 modes, we also allow inputs of any order; we
    always sum over the second-to-last dimension of the "matrix", weighted by the last dimension of
    the "vector".  Higher-order tensors get complicated for matching things, though, so there is a
    hard constraint: all dimensions in the "matrix" before the final embedding must be matched in
    the "vector".

    For example, say I have a "matrix" with dimensions (batch_size, num_queries, num_words,
    embedding_dim), representing some kind of embedding or encoding of several multi-word queries.
    My attention "vector" must then have at least those dimensions, and could have more.  So I
    could have an attention over words per query, with shape (batch_size, num_queries, num_words),
    or I could have an attention over query words for every document in some list, with shape
    (batch_size, num_documents, num_queries, num_words).  Both of these cases are fine.  In the
    first case, the returned tensor will have shape (batch_size, num_queries, embedding_dim), and
    in the second case, it will have shape (batch_size, num_documents, num_queries, embedding_dim).
    But you `can't` have an attention "vector" that does not include all of the queries, so shape
    (batch_size, num_words) is not allowed - you haven't specified how to handle that dimension in
    the "matrix", so we can't do anything with this input.
    """
    def __init__(self, use_masking: bool=True, **kwargs):
        self.use_masking = use_masking
        super(WeightedSum, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        # We don't need to worry about a mask after we've summed over the rows of the matrix.
        # You might actually still need a mask if you used a higher-order tensor, but probably the
        # right place to handle that is with careful use of TimeDistributed.  Or submit a PR.
        return None

    @overrides
    def compute_output_shape(self, input_shapes):
        matrix_shape, attention_shape = input_shapes
        return attention_shape[:-1] + matrix_shape[-1:]

    @overrides
    def call(self, inputs, mask=None):
        # pylint: disable=redefined-variable-type
        matrix, attention_vector = inputs
        matrix = self._expand_matrix_if_necessary(matrix, attention_vector, K.ndim(matrix) - 1)
        if mask is None:
            matrix_mask = None
        else:
            matrix_mask = mask[0]
        if self.use_masking and matrix_mask is not None:
            matrix_mask = self._expand_matrix_if_necessary(matrix_mask, attention_vector, K.ndim(matrix) - 1)
            # Doing a multiply here instead of a `switch` to avoid allocating another large tensor.
            matrix = K.cast(K.expand_dims(matrix_mask), 'float32') * matrix
        return K.sum(K.expand_dims(attention_vector, axis=-1) * matrix, -2)

    @staticmethod
    def _expand_matrix_if_necessary(matrix, attention_vector, num_matrix_dims: int):
        """
        This function tiles the matrix to get whatever missing dimensions there are from
        the attention vector.  We assume that all of the matrix dimensions are already in the
        attention "vector", and we just add dimensions to the beginning of the matrix (after the
        batch dimension).  In order to do the tiling correctly, we need the attention vector itself
        (to get its runtime shape), and the number of dimensions in the matrix (so we can ignore
        them from the attention vector).

        Note that ``num_matrix_dims`` is the number of dimensions `not including the embedding
        dimension`.  That it, it's the number of dimensions that we expect to be in common with the
        attention vector.
        """
        num_attention_dims = K.ndim(attention_vector)
        if num_matrix_dims == num_attention_dims:
            # If the matrix already has all of the dimensions of the attention vector, there's
            # nothing for us to do.
            return matrix
        for _ in range(num_attention_dims - num_matrix_dims):
            matrix = K.expand_dims(matrix, axis=1)
        # This is a bit fancy - the dimensions we need to add are all of the dimensions from the
        # attention "vector" that are _not_ already in the "matrix" - the slice
        # [1:-(num_matrix_dims-1)] gives us this, cutting out the batch size at the beginning, and
        # the shared matrix dims at the end.
        missing_attention_dims = K.shape(attention_vector)[1:-(num_matrix_dims-1)]
        # This will be a tensor like [1, attention_dims, 1, 1, 1], with one `1` for every dimension
        # of the original matrix, _including_ the embedding dimension, which is ignored above.
        # That's why there's no -1 in the [1] * (num_matrix_dims) here, while there was one in the
        # line right above this one.
        tile_shape = K.concatenate([[1], missing_attention_dims, [1] * (num_matrix_dims)], 0)
        matrix = K.tile(matrix, tile_shape)
        return matrix

    @overrides
    def get_config(self):
        base_config = super(WeightedSum, self).get_config()
        config = {'use_masking': self.use_masking}
        config.update(base_config)
        return config
