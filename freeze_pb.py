import re
import timeit

import cv2
import math
import tensorflow as tf
from sklearn import preprocessing

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
import numpy as np

import utils

CKPT_NAME = 'RES_NET50_best0.95739_80000.ckpt'
INPUT_ORDER = {
    "BatchNormWithGlobalNormalization":
        ["conv_op", "mean_op", "var_op", "beta_op", "gamma_op"],
    "FusedBatchNorm": ["conv_op", "gamma_op", "beta_op", "mean_op", "var_op"]
}
EPSILON_ATTR = {
    "BatchNormWithGlobalNormalization": "variance_epsilon",
    "FusedBatchNorm": "epsilon"
}

# OUTPUT = 'combine_loss/embedding_dense'
# OUTPUT = 'embedding_dense'
OUTPUT = 'g_type/embedding/BiasAdd'


def test():
    img_1 = cv2.imread('images/盧廣仲/盧廣仲_000002.jpg')
    img_1 = cv2.resize(img_1, (224, 224))
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_1 = np.array(img_1, dtype=np.float32)

    img_2 = cv2.imread('images/豬哥亮/豬哥亮_000004.jpg')
    img_2 = cv2.resize(img_2, (224, 224))
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    img_2 = np.array(img_2, dtype=np.float32)

    with tf.Session() as sess:
        graph_def = tf.GraphDef()
        with gfile.FastGFile('model_out/frozen_model_star.pb', 'rb') as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        input_tensor = tf.get_default_graph().get_tensor_by_name(
            "input_images:0")
        embedding_tensor = tf.get_default_graph().get_tensor_by_name(
            f"{OUTPUT}:0")
        # tf.summary.FileWriter("output_models/", graph=tf.get_default_graph())
        result1 = sess.run(embedding_tensor, feed_dict={input_tensor: np.expand_dims(img_1, axis=0)})[0]
        result2 = sess.run(embedding_tensor, feed_dict={input_tensor: np.expand_dims(img_2, axis=0)})[0]
        # print(result1[np.argmax(result1)])
        # print(result2[np.argmax(result2)])
        print(result1)
        print(result2)
        # vector_pair = preprocessing.normalize(
        #     [result1, result2])
        # dist = np.linalg.norm(vector_pair[0] - vector_pair[1])
        dist = np.linalg.norm(result1 - result2)
        print(dist)


def freeze():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            f'model_out/{CKPT_NAME}.meta', clear_devices=True)
        saver.restore(sess, f"model_out/{CKPT_NAME}")

        input_tensor = tf.get_default_graph().get_tensor_by_name(
            "input_images:0")
        with tf.variable_scope('pre_processing'):
            out = tf.subtract(input_tensor, 127.5)
            tf.multiply(
                out, 0.0078125, name='out')

        output_nodes = [OUTPUT, 'pre_processing/out']
        frozen_graph = freeze_session(sess, output_names=output_nodes)
        frozen_graph = optimize_for_inference(
            sess, frozen_graph, ['is_training'],
            {'input_images': 'pre_processing/out'}, output_nodes)

        tf.summary.FileWriter("model_out/", graph=frozen_graph)
        tf.io.write_graph(frozen_graph, "model_out/", "frozen_model_star.pb", as_text=False)


def replace_pre_processing(input_graph, replace_nodes=None):
    if not replace_nodes:
        replace_nodes = {}
    input_nodes = input_graph.node

    nodes_after_replace = []
    for node in input_nodes:
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for input_name in input_before_removal:
            if input_name in replace_nodes:
                new_input_name = replace_nodes[input_name]
                new_node.input.append(new_input_name)
                continue
            new_node.input.append(input_name)
        nodes_after_replace.append(new_node)

    return input_graph


def freeze_session(session,
                   keep_var_names=None,
                   output_names=None,
                   blacklist=None,
                   clear_devices=True):
    graph = session.graph
    with graph.as_default():
        # freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        # output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names)
        return frozen_graph


def optimize_for_inference(sess, input_graph_def, is_train_nodes,
                           replace_nodes, output_node_names):
    ensure_graph_is_valid(input_graph_def)
    optimized_graph_def = input_graph_def
    optimized_graph_def = modify_nodes(
        optimized_graph_def, output_node_names, is_train_nodes, replace_nodes)
    optimized_graph_def = replace_pre_processing(optimized_graph_def,
                                                 replace_nodes)
    optimized_graph_def = remove_unuse_node(sess, optimized_graph_def,
                                            output_node_names)
    # optimized_graph_def = fold_batch_norms(optimized_graph_def)

    ensure_graph_is_valid(optimized_graph_def)
    return optimized_graph_def


def remove_unuse_node(sess, graph_def, output_node_names):
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess, graph_def, output_node_names)
    return frozen_graph


def ensure_graph_is_valid(graph_def):
    node_map = {}
    for node in graph_def.node:
        if node.name not in node_map:
            node_map[node.name] = node
        else:
            raise ValueError("Duplicate node names detected for ", node.name)
    for node in graph_def.node:
        for input_name in node.input:
            input_node_name = node_name_from_input(input_name)
            if input_node_name not in node_map:
                raise ValueError("Input for ", node.name, " not found: ",
                                 input_name)


def node_name_from_input(node_name):
    """Strips off ports and other decorations to get the underlying node name."""
    if node_name.startswith("^"):
        node_name = node_name[1:]
    m = re.search(r"(.*):\d+$", node_name)
    if m:
        node_name = m.group(1)
    return node_name


def modify_nodes(input_graph,
                 protected_nodes=None,
                 is_train_nodes=None,
                 replace_nodes=None):
    if not protected_nodes:
        protected_nodes = []
    if not is_train_nodes:
        is_train_nodes = []
    if not replace_nodes:
        replace_nodes = {}

    # -------------- remove -------------------
    types_to_remove = {"CheckNumerics": True}

    input_nodes = input_graph.node
    names_to_remove = {}
    for node in input_nodes:
        if node.op in types_to_remove and node.name not in protected_nodes:
            names_to_remove[node.name] = True

    nodes_after_removal = []
    for node in input_nodes:
        if node.name in names_to_remove:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub(r"^\^", "", full_input_name)
            if input_name in names_to_remove:
                continue
            new_node.input.append(full_input_name)
        nodes_after_removal.append(new_node)

    # -------------- replace -------------------
    nodes_after_replace = []
    for node in nodes_after_removal:
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for input_name in input_before_removal:
            if input_name in replace_nodes and 'pre_processing' not in node.name.split('/'):
                new_input_name = replace_nodes[input_name]
                new_node.input.append(new_input_name)
                continue
            new_node.input.append(input_name)
        nodes_after_replace.append(new_node)

    # -------------- splice -------------------
    types_to_splice = {"Identity": True}
    control_input_names = set()
    node_names_with_control_input = set()
    for node in nodes_after_replace:
        for node_input in node.input:
            if "^" in node_input:
                control_input_names.add(node_input.replace("^", ""))
                node_names_with_control_input.add(node.name)

    names_to_splice = {}
    for node in nodes_after_replace:
        if node.op in types_to_splice and node.name not in protected_nodes:
            # We don't want to remove nodes that have control edge inputs, because
            # they might be involved in subtle dependency issues that removing them
            # will jeopardize.
            if node.name not in node_names_with_control_input:
                names_to_splice[node.name] = node.input[0]

    # We also don't want to remove nodes which are used as control edge inputs.
    names_to_splice = {
        name: value
        for name, value in names_to_splice.items()
        if name not in control_input_names
    }

    nodes_after_splicing = []
    for node in nodes_after_replace:
        if node.name in names_to_splice:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub(r"^\^", "", full_input_name)
            while input_name in names_to_splice:
                full_input_name = names_to_splice[input_name]
                input_name = re.sub(r"^\^", "", full_input_name)
            new_node.input.append(full_input_name)
        nodes_after_splicing.append(new_node)

    # -------------- select -------------------
    types_to_select = {"Switch": True}

    names_to_select = {}
    for node in nodes_after_splicing:
        if node.op not in types_to_select:
            continue
        for node_i in node.input:
            if node_i in is_train_nodes:
                names_to_select[node.name] = [
                    x for x in node.input if x not in is_train_nodes
                ]

    nodes_after_select = []
    for node in nodes_after_splicing:
        if node.name in names_to_select:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            if full_input_name in names_to_select:  # input is Switch
                for input_name in names_to_select[full_input_name]:
                    new_node.input.append(input_name)
            else:
                new_node.input.append(full_input_name)
        nodes_after_select.append(new_node)

    # --------------- skip ------------------
    types_to_skip = {"Merge": True}

    names_to_skip = {}
    for node in nodes_after_select:
        if node.op in types_to_skip:
            names_to_skip[node.name] = node.input[
                0]  # FusedBatchNorm_1=0, FusedBatchNorm=1

    nodes_after_skip = []
    for node in nodes_after_select:
        if node.name in names_to_skip:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub(r"^\^", "", full_input_name)
            while input_name in names_to_skip:
                full_input_name = names_to_skip[input_name]
                input_name = re.sub(r"^\^", "", full_input_name)
            new_node.input.append(full_input_name)
        nodes_after_skip.append(new_node)

    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_skip)
    return output_graph


def fold_batch_norms(input_graph_def):
    input_node_map = {}
    for node in input_graph_def.node:
        if node.name not in input_node_map:
            input_node_map[node.name] = node
        else:
            raise ValueError("Duplicate node names detected for ", node.name)

    nodes_to_skip = {}
    new_ops = []
    for node in input_graph_def.node:
        if node.op not in ("BatchNormWithGlobalNormalization",
                           "FusedBatchNorm"):
            continue

        conv_op = node_from_map(
            input_node_map, node.input[INPUT_ORDER[node.op].index("conv_op")])
        if conv_op.op != "Conv2D" and conv_op.op != "DepthwiseConv2dNative":
            # for prev_input in conv_op.input:

            print("Didn't find expected Conv2D or DepthwiseConv2dNative"
                  " input to '%s', it is %s" % (node.name, conv_op.name))
            continue

        weights_op = node_from_map(input_node_map, conv_op.input[1])
        if weights_op.op != "Const":
            print("Didn't find expected conv Constant input to '%s',"
                  " found %s instead. Maybe because freeze_graph wasn't"
                  " run first?" % (conv_op.name, weights_op))
            continue
        weights = values_from_const(weights_op)
        if conv_op.op == "Conv2D":
            channel_count = weights.shape[3]
        elif conv_op.op == "DepthwiseConv2dNative":
            channel_count = weights.shape[2] * weights.shape[3]

        mean_op = node_from_map(
            input_node_map, node.input[INPUT_ORDER[node.op].index("mean_op")])
        if mean_op.op != "Const":
            print("Didn't find expected mean Constant input to '%s',"
                  " found %s instead. Maybe because freeze_graph wasn't"
                  " run first?" % (node.name, mean_op))
            continue
        mean_value = values_from_const(mean_op)
        if mean_value.shape != (channel_count,):
            print(
                "Incorrect shape for mean, found %s, expected %s,"
                " for node %s" % (str(mean_value.shape), str(
                    (channel_count,)), node.name))
            continue

        var_op = node_from_map(
            input_node_map, node.input[INPUT_ORDER[node.op].index("var_op")])
        if var_op.op != "Const":
            print("Didn't find expected var Constant input to '%s',"
                  " found %s instead. Maybe because freeze_graph wasn't"
                  " run first?" % (node.name, var_op))
            continue
        var_value = values_from_const(var_op)
        if var_value.shape != (channel_count,):
            print("Incorrect shape for var, found %s, expected %s,"
                  " for node %s" % (str(var_value.shape), str(
                (channel_count,)), node.name))
            continue

        beta_op = node_from_map(
            input_node_map, node.input[INPUT_ORDER[node.op].index("beta_op")])
        if beta_op.op != "Const":
            print("Didn't find expected beta Constant input to '%s',"
                  " found %s instead. Maybe because freeze_graph wasn't"
                  " run first?" % (node.name, beta_op))
            continue
        beta_value = values_from_const(beta_op)
        if beta_value.shape != (channel_count,):
            print(
                "Incorrect shape for beta, found %s, expected %s,"
                " for node %s" % (str(beta_value.shape), str(
                    (channel_count,)), node.name))
            continue

        gamma_op = node_from_map(
            input_node_map, node.input[INPUT_ORDER[node.op].index("gamma_op")])
        if gamma_op.op != "Const":
            print("Didn't find expected gamma Constant input to '%s',"
                  " found %s instead. Maybe because freeze_graph wasn't"
                  " run first?" % (node.name, gamma_op))
            continue
        gamma_value = values_from_const(gamma_op)
        if gamma_value.shape != (channel_count,):
            print(
                "Incorrect shape for gamma, found %s, expected %s,"
                " for node %s" % (str(gamma_value.shape), str(
                    (channel_count,)), node.name))
            continue

        variance_epsilon_value = node.attr[EPSILON_ATTR[node.op]].f
        nodes_to_skip[node.name] = True
        nodes_to_skip[weights_op.name] = True
        nodes_to_skip[mean_op.name] = True
        nodes_to_skip[var_op.name] = True
        nodes_to_skip[beta_op.name] = True
        nodes_to_skip[gamma_op.name] = True
        nodes_to_skip[conv_op.name] = True

        if scale_after_normalization(node):
            scale_value = ((1.0 / np.vectorize(
                math.sqrt)(var_value + variance_epsilon_value)) * gamma_value)
        else:
            scale_value = (1.0 / np.vectorize(
                math.sqrt)(var_value + variance_epsilon_value))
        offset_value = (-mean_value * scale_value) + beta_value
        scaled_weights = np.copy(weights)
        it = np.nditer(
            scaled_weights, flags=["multi_index"], op_flags=["readwrite"])
        if conv_op.op == "Conv2D":
            while not it.finished:
                current_scale = scale_value[it.multi_index[3]]
                it[0] *= current_scale
                it.iternext()
        elif conv_op.op == "DepthwiseConv2dNative":
            channel_multiplier = weights.shape[3]
            while not it.finished:
                current_scale = scale_value[
                    it.multi_index[2] * channel_multiplier + it.multi_index[3]]
                it[0] *= current_scale
                it.iternext()
        scaled_weights_op = node_def_pb2.NodeDef()
        scaled_weights_op.op = "Const"
        scaled_weights_op.name = weights_op.name
        scaled_weights_op.attr["dtype"].CopyFrom(weights_op.attr["dtype"])
        scaled_weights_op.attr["value"].CopyFrom(
            attr_value_pb2.AttrValue(
                tensor=tensor_util.make_tensor_proto(
                    scaled_weights, weights.dtype.type, weights.shape)))
        new_conv_op = node_def_pb2.NodeDef()
        new_conv_op.CopyFrom(conv_op)
        offset_op = node_def_pb2.NodeDef()
        offset_op.op = "Const"
        offset_op.name = conv_op.name + "_bn_offset"
        offset_op.attr["dtype"].CopyFrom(mean_op.attr["dtype"])
        offset_op.attr["value"].CopyFrom(
            attr_value_pb2.AttrValue(
                tensor=tensor_util.make_tensor_proto(
                    offset_value, mean_value.dtype.type, offset_value.shape)))
        bias_add_op = node_def_pb2.NodeDef()
        bias_add_op.op = "BiasAdd"
        bias_add_op.name = node.name
        bias_add_op.attr["T"].CopyFrom(conv_op.attr["T"])
        bias_add_op.attr["data_format"].CopyFrom(conv_op.attr["data_format"])
        bias_add_op.input.extend([new_conv_op.name, offset_op.name])
        new_ops.extend(
            [scaled_weights_op, new_conv_op, offset_op, bias_add_op])

    result_graph_def = graph_pb2.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        result_graph_def.node.extend([new_node])

    result_graph_def.node.extend(new_ops)
    return result_graph_def


def node_name_from_input(node_name):
    """Strips off ports and other decorations to get the underlying node name."""
    if node_name.startswith("^"):
        node_name = node_name[1:]
    m = re.search(r"(.*):\d+$", node_name)
    if m:
        node_name = m.group(1)
    return node_name


def node_from_map(node_map, name):
    """Pulls a node def from a dictionary for a given name.
    Args:
      node_map: Dictionary containing an entry indexed by name for every node.
      name: Identifies the node we want to find.
    Returns:
      NodeDef of the node with the given name.
    Raises:
      ValueError: If the node isn't present in the dictionary.
    """
    stripped_name = node_name_from_input(name)
    if stripped_name not in node_map:
        raise ValueError("No node named '%s' found in map." % name)
    return node_map[stripped_name]


def values_from_const(node_def):
    """Extracts the values from a const NodeDef as a numpy ndarray.
    Args:
      node_def: Const NodeDef that has the values we want to access.
    Returns:
      Numpy ndarray containing the values.
    Raises:
      ValueError: If the node isn't a Const.
    """
    if node_def.op != "Const":
        raise ValueError(
            "Node named '%s' should be a Const op for values_from_const." %
            node_def.name)
    input_tensor = node_def.attr["value"].tensor
    tensor_value = tensor_util.MakeNdarray(input_tensor)
    return tensor_value


def scale_after_normalization(node):
    if node.op == "BatchNormWithGlobalNormalization":
        return node.attr["scale_after_normalization"].b
    return True


if __name__ == '__main__':
    # freeze()
    test()
