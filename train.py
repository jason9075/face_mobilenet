import tensorflow as tf

from backend.loss_function import combine_loss_val
from backend.mobilenet_v2 import MobileNetV2


def main():
    num_classes = 10
    labels = [0,1,2,3,4,5,6,7,8,9,10]

    with tf.Session() as sess:
        net = MobileNetV2(input_size=(224, 224))

        logit = combine_loss_val(embedding=net.embedding, labels=labels, num_labels=num_classes, m1=1, m2=0, m3=0, s=64)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))

        tf.summary.FileWriter("./", graph=tf.get_default_graph())


if __name__ == '__main__':
    main()
