import urllib2
import numpy as np
from matplotlib import pyplot
from StringIO import StringIO
from caffe2.python import core, utils, workspace, model_helper, brew, net_drawer
from caffe2.proto import caffe2_pb2
import scipy.io

trainingData = scipy.io.loadmat('trainingData.mat')
trainingLabels = scipy.io.loadmat('trainingLabels.mat')
# load the features to a feature matrix.
features = trainingData['sortedmedianTable']
# load the labels to a feature matrix
labels = np.reshape(trainingLabels['sortedClassifierOutput'],444)
labels = labels.astype(int)

random_index = np.random.permutation(444)
features = features[random_index]
labels = labels[random_index]

train_features = features[:350]
train_features = train_features.reshape(350,1,1,161)
train_labels = labels[:350]
test_features = features[350:]
test_features = test_features.reshape(94,1,1,161)
test_labels = labels[350:]

def write_db(db_type, db_name, features, labels):
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()
    for i in range(features.shape[0]):
        feature_and_label = caffe2_pb2.TensorProtos()
        feature_and_label.protos.extend([
            utils.NumpyArrayToCaffe2Tensor(features[i]),
            utils.NumpyArrayToCaffe2Tensor(labels[i])])
        transaction.put(
            'tran_%03d'.format(i),
            feature_and_label.SerializeToString())
    # Close the transaction, and then close the db.
    del transaction
    del db
write_db("minidb", "event_train1.minidb", train_features, train_labels)
write_db("minidb", "event_test1.minidb", test_features, test_labels)

def AddTrainingOperators(model, softmax, label):
    # something very important happens here
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = brew.iter(model, "iter")
    # set the learning rate schedule
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - ModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)
    # let's checkpoint every 20 iterations, which should probably be fine.
    # you may need to delete tutorial_files/tutorial-mnist to re-run the tutorial
    #model.Checkpoint([ITER] + model.params, [],
    #               db="event_testModel_checkpoint_%05d.leveldb", db_type="leveldb", every=20)

def AddBookkeepingOperators(model):
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
    # Now, if we really want to be verbose, we can summarize EVERY blob
    # that the model produces; it is probably not a good idea, because that
    # is going to take time - summarization do not come for free. For this
    # demo, we will only show how to summarize the parameters and their
    # gradients.

def addModel(model, data):
    channels = 1
    if model.init_params:
        weight = model.param_init_net.GaussianFill(
            [],
            'conv1' + '_w',
            shape=[channels,1,1,5]
        )
        print("initial_conv1_w: " + str(workspace.FetchBlob('conv1_w')))
        bias = model.param_init_net.ConstantFill(
            [],
            'conv1' + '_b',
            shape=[channels, ]
        )
    else:
        weight = core.ScopedBlobReference(
            'conv1' + '_w', model.param_init_net)
        bias = core.ScopedBlobReference(
            'conv1' + '_b', model.param_init_net)

    model.params.extend([weight, bias])
    model.weights.append(weight)
    model.biases.append(bias)

    conv1 = model.net.Conv([data, weight, bias], 'conv1', dim_in=1, dim_out=channels, kernel_h=1, kernel_w=5)
    #conv1 = brew.conv(model, data, 'conv1', 1, 2, 5)
    #pool1 = brew.max_pool(model, conv1, 'pool1', kernel_h=1, kernel_w=2, stride=2)
    fc3 = brew.fc(model, conv1, 'fc3', dim_in=channels*157, dim_out=100)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 100, 10)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax

def AddAccuracy(model, softmax, label):
    accuracy = model.Accuracy([softmax, label], "accuracy")
    return accuracy

arg_scope = {"order": "NCHW"}
train_model = model_helper.ModelHelper(name="event_train", arg_scope=arg_scope)
train_data, train_labels = train_model.TensorProtosDBInput(
    [], ["data_float", "label"], batch_size=350, db='event_train1.minidb', db_type='minidb')
workspace.FeedBlob("TD",train_features.astype(np.float32))
softmax = addModel(train_model, "TD")
AddTrainingOperators(train_model, softmax, train_labels)
AddBookkeepingOperators(train_model)

test_model = model_helper.ModelHelper(
    name="event_test", arg_scope=arg_scope, init_params=False)
test_features, test_labels = train_model.TensorProtosDBInput(
    [], ["data_float", "label"], batch_size=94, db='event_test1.minidb', db_type='minidb')
softmax = addModel(test_model, test_features)
AddAccuracy(test_model, softmax, test_labels)

#from IPython import display
#graph = net_drawer.GetPydotGraph(train_model.net.Proto().op, name="events", rankdir="LR")
#display.Image(graph.create_png(), width=800)

workspace.RunNetOnce(train_model.param_init_net)

workspace.CreateNet(train_model.net, overwrite=True)

total_iters = 200
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)

for i in range(total_iters):
    workspace.RunNet(train_model.net.Proto().name)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
    print("Conv_w: " + str(workspace.FetchBlob('conv1_w')))
    print("Conv_b: " + str(workspace.FetchBlob('conv1_b')))
pyplot.plot(loss, 'b')
pyplot.plot(accuracy, 'r')
pyplot.legend(('loss', 'Accuracy'), loc='upper right')

pyplot.show()
