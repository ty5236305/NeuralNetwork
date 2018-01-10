import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])

# count = 0
#
# for data in training_data:
#     net.SGD([data], 3, 1, 0.2)
#     count = count + 1
#     print "complete {0}".format(count)


net.SGD(training_data, 30, 10, 3.0, test_data = test_data)

print "--validation_data--"
print "result: {0} / {1}".format(net.evaluate(validation_data), len(validation_data))