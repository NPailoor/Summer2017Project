import urllib2
import numpy as np
from matplotlib import pyplot
from StringIO import StringIO
from caffe2.python import core, utils, workspace, model_helper, brew, net_drawer
from caffe2.proto import caffe2_pb2
import caffe2.python.predictor.mobile_exporter
import scipy.io
import os
import copy

workspace.ResetWorkspace()
trainingData = scipy.io.loadmat('trainingData.mat')
#trainingLabels = scipy.io.loadmat('trainingLabels.mat')
# load the features to a feature matrix.
features = trainingData['sortedmedianTable']
features2 = np.genfromtxt('EventTable.csv', delimiter=',')
features2 = features2[0:1000]
# load the labels to a feature matrix
#labels = np.reshape(trainingLabels['sortedClassifierOutput'],444)
labels = np.genfromtxt('revisedLabels.csv', delimiter=',')
labels = labels.astype(int)
features = np.delete(features, [141, 363], axis=0)
labels2 = np.genfromtxt('EventsLabeled.csv', delimiter=',')
labels2 = labels2.astype(int)
features = features[:,30:70]
for i in range(0, len(labels)):
    min = np.amin(features[i,:])
    max = np.amax(features[i,:])
    features[i,:] = (features[i,:] - min)/(max - min)
features = np.append(features, features2,axis=0)
labels = np.append(labels, labels2)
#deploymentData= scipy.io.loadmat('deployment_test_data.mat')
#deploymentLabels = scipy.io.loadmat('deployment_test_labels.mat')
#dData = deploymentData['medianTable2']
#dLabels = np.reshape(deploymentLabels['classifierOutput'], 3331)
eventLabels = np.genfromtxt('potentialEventsRelabeled.csv', delimiter=',')
eventData = np.genfromtxt('potentialEvents.csv', delimiter=',')
nonEventLabels = np.genfromtxt('nonEventLabels.csv', delimiter=',')
nonEventData = np.genfromtxt('nonEvents.csv', delimiter=',')
dLabels = np.append(eventLabels,nonEventLabels)
dData = np.append(eventData,nonEventData,axis=0)
#features = np.append(features, eventData,axis=0)
#labels = np.append(labels, eventLabels)
print(features.shape)
dLabels = dLabels.astype(np.int32)
dData = dData.astype(np.float32)

random_index = np.random.permutation(len(labels))
features = features[random_index]
labels = labels[random_index]
#print(np.argmax(features, axis=0))
#features = np.random.rand(len(labels),40)

random_index = np.random.permutation(len(dLabels))
dData = dData[random_index]
dLabels = dLabels[random_index]
dDataBig = copy.deepcopy(dData)
dData = dData[:,30:70]
for i in range(0,len(dLabels)):
    min = np.amin(dData[i,:])
    max = np.amax(dData[i,:])
    dData[i,:] = (dData[i,:] - min)/(max - min)

train_features = features[:len(labels)/2]
train_features = train_features.reshape(len(labels)/2,1,1,40)
train_labels = labels[:len(labels)/2]
test_features = features[len(labels)/2:]
test_features = test_features.reshape(len(labels)-len(labels)/2,1,1,40)
test_labels = labels[len(labels)/2:]
dData = dData.reshape(len(dLabels),1,1,40)

def write_db(db_type, db_name, features, labels):
    #remove if db already exists
    if (os.path.exists(db_name)):
        os.remove(os.path.join(db_name))
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
#write_db("minidb", "event_deploy.minidb", dData, dLabels)

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
    channels = 50
    channels2 = 200
    kernel_size = 3
    if model.init_params:
        weight = model.param_init_net.XavierFill(
            [],
            'conv1' + '_w',
            shape=[channels,1,1,kernel_size]
        )
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

    conv1 = model.net.Conv([data, weight, bias], 'conv1', dim_in=1, dim_out=channels, kernel_h=1, kernel_w=kernel_size)
    #conv1 = brew.conv(model, data, 'conv1', 1, 2, 5)
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel_h=1, kernel_w=2, stride=2)
    pool_dim_out = (41-kernel_size)/2
    if model.init_params:
        weight2 = model.param_init_net.XavierFill(
            [],
            'conv2' + '_w',
            shape=[channels2,channels,1,kernel_size]
        )
        bias2 = model.param_init_net.ConstantFill(
            [],
            'conv2' + '_b',
            shape=[channels2, ]
        )
    else:
        weight2 = core.ScopedBlobReference(
            'conv2' + '_w', model.param_init_net)
        bias2 = core.ScopedBlobReference(
            'conv2' + '_b', model.param_init_net)

    model.params.extend([weight2, bias2])
    model.weights.append(weight2)
    model.biases.append(bias2)
    conv2 = model.net.Conv([pool1, weight2, bias2], 'conv2', dim_in=channels, dim_out=channels2, kernel_h=1, kernel_w=kernel_size)
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel_h=1, kernel_w=2, stride=2)
    pool_dim_out_2 = (pool_dim_out + 1 - kernel_size)/2
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=pool_dim_out_2*channels2, dim_out=1000)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 1000, 2)
    #print(workspace.FetchBlob('pred_w'))
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax

def AddAccuracy(model, softmax, label):
    accuracy = model.Accuracy([softmax, label], "accuracy")
    return accuracy

def AddInput(model, batch_size, db, db_type):
    data_float, label_int = model.TensorProtosDBInput(
        [], ["data_float", "label_int"], batch_size=batch_size,
        db=db, db_type=db_type)
    data = model.Cast(data_float, "data", to=core.DataType.FLOAT)
    label = model.Cast(label_int, "label", to=core.DataType.INT32)
    data = model.StopGradient(data, data)
    return data, label


arg_scope = {"order": "NCHW"}
train_model = model_helper.ModelHelper(name="event_train", arg_scope=arg_scope)
train_data, train_labels = AddInput(train_model, 20, 'event_train1.minidb', db_type='minidb')
#workspace.FeedBlob("TD",train_features.astype(np.float32))
#workspace.FeedBlob("TL", train_labels.astype(np.int32))
softmax = addModel(train_model, train_data)
AddTrainingOperators(train_model, softmax, train_labels)
AddBookkeepingOperators(train_model)

test_model = model_helper.ModelHelper(
    name="event_test", arg_scope=arg_scope, init_params=False)
test_features, test_labels = AddInput(test_model, 30, 'event_test1.minidb', db_type='minidb')
#workspace.FeedBlob("TestD",test_features.astype(np.float32))
#workspace.FeedBlob("TestL", train_labels.astype(np.int32))
softmax = addModel(test_model, test_features)
AddAccuracy(test_model, softmax, test_labels)



from IPython import display
graph = net_drawer.GetPydotGraph(train_model.net.Proto().op, name="events", rankdir="LR")
display.Image(graph.create_png(), width=800)

workspace.RunNetOnce(train_model.param_init_net)

workspace.CreateNet(train_model.net, overwrite=True)
total_iters = 1000
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)

for i in range(total_iters):
    workspace.RunNet(train_model.net.Proto().name)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
    #print("First Layer Weights: ")
    #print(workspace.FetchBlob('fc3_w'))
    #print("Param Grad: ")
    #print(workspace.FetchBlob('fc3_w_grad'))
    #print("Conv_w: " + str(workspace.FetchBlob('conv1_w')))
    #print("Conv_b: " + str(workspace.FetchBlob('conv1_b')))

#print(workspace.FetchBlob(softmax))
pyplot.plot(loss, 'b')
pyplot.plot(accuracy, 'r')
pyplot.legend(('loss', 'Accuracy'), loc='upper right')
pyplot.show()

# run a test pass on the test net
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net)
test_accuracy = np.zeros(100)
for i in range(100):
    workspace.RunNet(test_model.net.Proto().name)
    test_accuracy[i] = workspace.FetchBlob('accuracy')
# After the execution is done, let's plot the values
pyplot.plot(test_accuracy, 'r')
pyplot.title('Acuracy over test batches.')
print('test_accuracy: %f' % test_accuracy.mean())
pyplot.show()

workspace.FeedBlob("Test2", dData)
workspace.FeedBlob("Test2Labels", dLabels)


deploy_model = model_helper.ModelHelper(name="event_deploy", arg_scope=arg_scope,
init_params=False)
softmax2 = addModel(deploy_model, "Test2")
init_net, predict_net = caffe2.python.predictor.mobile_exporter.Export(workspace, deploy_model.net, deploy_model.params)
with open(os.path.join("init_net.pb"), 'wb') as fid:
    fid.write(init_net.SerializeToString())
with open(os.path.join("predict_net.pb"), 'wb') as fid:
    fid.write(predict_net.SerializeToString())
deploy_labels = workspace.FetchBlob("Test2Labels")
AddAccuracy(deploy_model, softmax2, "Test2Labels")

workspace.CreateNet(deploy_model.net)
workspace.RunNet(deploy_model.net.Proto().name)
print('deploy_accuracy: %f' % workspace.FetchBlob('accuracy'))
output = workspace.FetchBlob(softmax2)
output = np.rint(output)
false_positives = 0
false_negatives = 0
true_positives = 0
true_negatives = 0
print(output)
for i in range(0, len(deploy_labels)):
    if deploy_labels[i] == 0:
        if output[i,0] == 1:
            true_negatives = true_negatives + 1
        else:
            false_positives = false_positives + 1
    else:
        if output[i,0] == 1:
            false_negatives = false_negatives + 1
        else:
            true_positives = true_positives + 1

print('False positives: ' + str(false_positives))
print('False negatives: ' + str(false_negatives))
print('True positives: ' + str(true_positives))
print('True negatives: ' + str(true_negatives))

#discoveredIndices = np.where(output[:,0] == 1)[0]
#discoveredNonEvents = dDataBig[discoveredIndices,:]
#print(discoveredNonEvents.shape)
#pELabels = dLabels[discoveredIndices]
#np.savetxt('nonEvents.csv', discoveredNonEvents, delimiter=',')
#np.savetxt('nonEventLabels.csv', pELabels, delimiter=',')

#for i in range(0,50):
#    print(str(np.argmax(output[i])) + ', ' + str(labels[i]))
