from backend.mobilenet_v2 import MobileNetV2


def main():
    num_classes = 1000
    net = MobileNetV2(num_classes)


if __name__ == '__main__':
    main()
