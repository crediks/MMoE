class MMoE(object):
    def __init__(self, hidden_size, num_experts, num_tasks):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_tasks = num_tasks

    def get_output(self, inputs):
        expert_weight = tf.get_variable(name='expert_weight', initializer=xavier_init, shape=[inputs.get_shape()[-1], self.hidden_size, self.num_experts])          
        expert_bias = tf.get_variable(name='expert_bias', initializer=xavier_init, shape=[self.hidden_size, self.num_experts])
        
        expert_output = tf.tensordot(inputs, expert_weight, axes=1) + expert_bias
        expert_output = tf.nn.relu(expert_output, name='expert_output')

        gate_weight = tf.get_variable(name='gate_weight', initializer=xavier_init, shape=[inputs.get_shape()[-1], self.num_experts, self.num_tasks])
        gate_bias = tf.get_variable(name='gate_bias', initializer=xavier_init, shape=[self.num_experts, self.num_tasks])
        
        gate_output = tf.tensordot(inputs, gate_weight, axes=1) + gate_bias

        gate_outputs = tf.split(gate_output, num_or_size_splits=self.num_tasks, axis=2)

        final_outputs = []
        for gate_output in gate_outputs:
        	gate_output = tf.transpose(gate_output, [0,2,1])
        	gate_output = tf.nn.softmax(gate_output, name='gate_output_softmax')

        	gate_output = tf.tile(gate_output, [1, self.hidden_size, 1])

        	weighted_expert_output = expert_output * gate_output

        	final_outputs.append(tf.reduce_sum(weighted_expert_output, axis=2))
        return final_outputs



