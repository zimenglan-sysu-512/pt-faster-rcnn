import math
import numpy as np
import time
from sklearn.decomposition import IncrementalPCA
from sklearn.utils import gen_batches
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
import warnings
from sklearn.utils.validation import NonBLASDotWarning
warnings.simplefilter('always', NonBLASDotWarning)

class IncrementalPCA_V2(IncrementalPCA):
    def __init__(self, n_components=None, whiten=False, copy=True, batch_size=None):
        super(self.__class__, self).__init__(n_components=n_components, whiten=whiten, copy=copy, batch_size=batch_size)

    def fit(self, X, y=None):
        """Fit the model with X, using minibatches of size batch_size.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        y: Passthrough for ``Pipeline`` compatibility.

        Returns
        -------
        self: object
            Returns the instance itself.
        """
        self.components_ = None
        self.mean_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.noise_variance_ = None
        self.var_ = None
        self.n_samples_seen_ = 0

        check_type = np.float
        if X.dtype in (np.float16,np.float32,np.float64):
            check_type = X.dtype.type
        X = check_array(X,check_type)

        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        # Don't make the batch size bigger than neccessary
        total_batches = int(math.ceil(n_samples / float(self.batch_size_)))
        for index, batch in enumerate(gen_batches(n_samples, self.batch_size_)):
            print "IncrementalPCA_V2: batch {}/{}...".format(index + 1, total_batches)
            self.partial_fit(X[batch])

        return self

    def chunking_dot(self, big_matrix, small_matrix, chunk_size=5000):
        assert isinstance(big_matrix, np.ndarray)
        assert isinstance(small_matrix, np.ndarray)
        # Make a copy if the array is not already contiguous
        small_matrix = np.ascontiguousarray(small_matrix, dtype=big_matrix.dtype)
        R = np.empty((big_matrix.shape[0], small_matrix.shape[1]))
        for i in range(0, R.shape[0], chunk_size):
            end = i + chunk_size
            partial_big_matrix = big_matrix[i:end]
            R[i:end] = np.dot(partial_big_matrix, small_matrix)
        return R

    def transform(self, X, y=None):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.decomposition import IncrementalPCA
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> ipca = IncrementalPCA(n_components=2, batch_size=3)
        >>> ipca.fit(X)
        IncrementalPCA(batch_size=3, copy=True, n_components=2, whiten=False)
        >>> ipca.transform(X) # doctest: +SKIP
        """

        start_time = time.time()
        check_is_fitted(self, ['mean_', 'components_'], all_or_any=all)

        # gc.collect()

        check_type = np.float
        if X.dtype in (np.float16,np.float32,np.float64):
            check_type = X.dtype.type
        X = check_array(X,check_type)

        if self.mean_ is not None:
            X -= self.mean_

        # X_transformed = fast_dot(X, self.components_.T)
        X_transformed = self.chunking_dot(X, self.components_.T)

        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)

        print 'IncrementalPCA_V2 transform time: %.3f' % (time.time() - start_time)

        return X_transformed