import tensorflow as tf


class EmbraceNet(tf.keras.Model):

    def __init__(self, modality_num, embracement_size=256, bypass_docking=False, **kwargs):
        """
        Initialize an EmbraceNet model.
        Args:
          modality_num: Amount of modalities.
          embracement_size: The length of the output of the embracement layer ("c" in the paper).
          bypass_docking: Bypass docking step, i.e., connect the input data directly to the embracement layer.
          If True, input_data must have a shape of [batch_size, embracement_size].
        """
        super(EmbraceNet, self).__init__(**kwargs)

        self.modality_num = modality_num
        self.embracement_size = embracement_size
        self.bypass_docking = bypass_docking

        if not bypass_docking:
            self.docking_layer_list = []
            for i in range(modality_num):
                self.docking_layer_list.append(
                    tf.keras.layers.Dense(
                        units=embracement_size,
                        name='docking/%d' % i,
                        activation='relu'
                    )
                )

    def call(self, input_list, availabilities=None, selection_probabilities=None):
        """
        Forward input data to the EmbraceNet module.
        Args:
          input_list: A list of input data. Each input data should have a size as in input_size_list.
          availabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents the availability of
          data for each modality. If None, it assumes that data of all modalities are available.
          selection_probabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents probabilities
          that output of each docking layer will be selected ("p" in the paper).
          If None, the same probability of being selected will be used for each docking layer.
        Returns:
          A 2-D tensor of shape [batch_size, embracement_size] that is the embraced output.
        """

        # check input data
        assert len(input_list) == self.modality_num
        num_modalities = len(input_list)
        batch_size = input_list[0].shape[0]

        # docking layer
        docking_output_list = []
        if self.bypass_docking:
            docking_output_list = input_list
        else:
            for i, input_data in enumerate(input_list):
                x = self.docking_layer_list[i](input_data)
                docking_output_list.append(x)

        # check availabilities
        if availabilities is None:
            availabilities = tf.ones([batch_size, len(input_list)], dtype=tf.dtypes.float32)
        else:
            availabilities = tf.cast(availabilities, tf.dtypes.float32)

        # adjust selection probabilities
        if selection_probabilities is None:
            selection_probabilities = tf.ones([batch_size, len(input_list)], dtype=tf.dtypes.float32)
        selection_probabilities = tf.math.multiply(selection_probabilities, availabilities)

        probabilty_sum = tf.reduce_sum(selection_probabilities, axis=-1, keepdims=True)  # [batch_size, 1]
        selection_probabilities = tf.math.divide(selection_probabilities,
                                                 probabilty_sum)  # [batch_size, num_modalities]

        # stack docking outputs
        docking_output_stack = tf.stack(docking_output_list, axis=-1)  # [batch_size, embracement_size, num_modalities]

        # embrace
        modality_indices = tf.random.categorical(tf.math.log(selection_probabilities),
                                                 num_samples=self.embracement_size)  # [batch_size, embracement_size]
        modality_toggles = tf.one_hot(modality_indices, depth=num_modalities, axis=-1,
                                      dtype=tf.dtypes.float32)  # [batch_size, embracement_size, num_modalities]

        embracement_output_stack = tf.math.multiply(docking_output_stack,
                                                    modality_toggles)  # [batch_size, embracement_size, num_modalities]
        embracement_output = tf.math.reduce_sum(embracement_output_stack, axis=-1)  # [batch_size, embracement_size]

        return embracement_output
