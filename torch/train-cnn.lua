
require 'cunn'
require 'optim'
require 'image'
require 'hdf5'
require 'cutorch'
require 'lfs'

package.path = package.path..';torch-toolbox-master/Sanitize/?.lua'

sanitize = require 'sanitize'

fname = 'train-cnn'


--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Megasolar Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname, 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-model', 'convnet', 'type of model to train: convnet | mlp | linear')
cmd:option('-full', false, 'use full dataset (50,000 samples)')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 32, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:option('-testInterval', 10, 'test interval')
cmd:option('-saveInterval', 500, 'save interval')
cmd:option('-inputChannel', 7, 'nb of input channels')
cmd:option('-maxEpoch', 1000, 'maximum nb of epochs')
cmd:option('-testBatchSize', 10000, 'test batch size')
cmd:option('-gpu', 1, 'id of gpu to use')
cmd:option('-dropout', false, '')
cmd:option('-batchNorm', false, '')
cmd:option('-negativeBatchSize', 128, 'negative batch size for negative mining')
cmd:option('-batchSizeNM', 32, 'mini-batch size for negative mining')
cmd:option('-epochStartNM', 5000, 'nb of epochs to start negative mining')
cmd:option('-trainDataPath', '', 'path of hdf5 dataset for training')
cmd:option('-testDataPath', '', 'path of hdf5 dataset for test')
cmd:option('-negativeRatio', 1, 'batch ratio, 1 : x = positive : negative')
cmd:option('-trainnorm', 'mean_stdv.csv', 'output training data mean and stdv path')
cmd:text()
opt = cmd:parse(arg)
--

print (opt.optimization)

cutorch.manualSeed(opt.seed)
cutorch.setDevice(opt.gpu)

classes = {'positive', 'negative'}
accLogger = optim.Logger(paths.concat(opt.save, 'accuracy.log'))
errLogger = optim.Logger(paths.concat(opt.save, 'error.log'   ))

if opt.network == '' then
    -- define model to train
    model = nn.Sequential()
    if opt.model == 'convnet' then
        ------------------------------------------------------------
        -- convolutional network
        ------------------------------------------------------------
        -- stage 1 
        model:add(nn.SpatialConvolution(opt.inputChannel, 32, 3, 3))
        model:add(nn.ReLU())
        if opt.batchNorm ~= false then
            model:add(nn.SpatialBatchNormalization(32))
        end
        
        -- stage 2 
        model:add(nn.SpatialConvolution(32, 32, 3, 3))
        model:add(nn.ReLU())
        if opt.batchNorm ~= false then
            model:add(nn.SpatialBatchNormalization(32))
        end
        
        -- stage 3 
        model:add(nn.SpatialConvolution(32, 32, 3, 3))
        model:add(nn.ReLU())
        if opt.batchNorm ~= false then
            model:add(nn.SpatialBatchNormalization(32))
        end
        
        
        -- stage 4 
        model:add(nn.Reshape(32*10*10))
        if opt.dropout ~= false then
            model:add(nn.Dropout(0.5))
        end
        model:add(nn.Linear(32*10*10,#classes))

        ------------------------------------------------------------
    else
        print('Unknown model type')
        cmd:text()
        error()
    end
else
    print('<trainer> reloading previously trained network')
    model = nn.Sequential()
    model = torch.load(opt.network)
end

-- verbose
print('using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
model:cuda()

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

criterion = nn.ClassNLLCriterion():cuda()

----------------------------------------------------------------------
-- Import Data 
--

function getFileList(dirPath, ext)
    list = {}
    for file in lfs.dir(dirPath) do
        if string.find(file, ext..'$') then
            table.insert(list,file)
        end
    end
    return list
end

numTrainPositive = 0 -- the number of training positive data
numTrainNegative = 0 -- the number of training negative data

if lfs.attributes(opt.trainDataPath)['mode'] == 'file' then
    -- import training data
    print("Import training data...")
    myFile = hdf5.open(opt.trainDataPath,'r')
    trainData = myFile:read(''):all()
    myFile:close()
    labelNum = trainData.label:size()[1]
    
    for i = 1,labelNum do
        if trainData.label[i] == 0 then
            trainData.label[i] = 1
            numTrainPositive = numTrainPositive + 1
        else 
            trainData.label[i] = 2
            numTrainNegative = numTrainNegative + 1
        end
    end

    numAllTrainData = labelNum
    
elseif lfs.attributes(opt.trainDataPath)['mode'] == 'directory' then
    hdfList = getFileList(opt.trainDataPath,'hdf5')
    dataList = {}
    for i=1, #hdfList do
        local myFile = hdf5.open(opt.trainDataPath..'/'..hdfList[i])
        print('Importing:'..opt.trainDataPath..'/'..hdfList[i])
        tempTrainData = myFile:read(''):all()
        myFile:close()
        labelNum = tempTrainData.label:size()[1]
    
        for i = 1,labelNum do
            if tempTrainData.label[i] == 0 then
                tempTrainData.label[i] = 1
                numTrainPositive = numTrainPositive + 1
            else 
                tempTrainData.label[i] = 2
                numTrainNegative = numTrainNegative + 1
            end
        end
        table.insert(dataList, tempTrainData)
    end
    numAllTrainData = 0
    for i=1, #dataList do
        numAllTrainData = numAllTrainData + dataList[i]['data']:size()[1]
    end
    print(numAllTrainData)
    trainData = {
        data=torch.Tensor(numAllTrainData, opt.inputChannel, dataList[1]['data']:size()[3], dataList[1]['data']:size()[4]),
        label=torch.Tensor(numAllTrainData),
        size = function() return numAllTrainData end
    }
    k = 1
    for i=1, #dataList do
        for j=1, dataList[i]['data']:size()[1] do
            trainData.data[k] = dataList[i]['data'][j]
            trainData.label[k] = dataList[i]['label'][j]
            k = k + 1
        end
    end
    
else
    error("invalid dataset")
end

dataMean = {}
dataStd = {}

file = io.open(opt.trainnorm,"w")
for t=1,opt.inputChannel do
    local trainMean = trainData.data[{{},t,{},{}}]:mean()
    local trainStd = trainData.data[{{},t,{},{}}]:std()
    trainData.data[{ {},t,{},{} }]:add(-trainMean)
    trainData.data[{ {},t,{},{} }]:div(trainStd)
    print(t.."ch mean:"..trainMean.." stdv:"..trainStd)
    file:write(trainMean..","..trainStd.."\n")
    table.insert(dataMean, trainMean)
    table.insert(dataStd, trainStd)
end
file:close()

positiveList = {}
negativeList = {}
for i = 1, trainData:size() do
    if trainData.label[i] == 1 then
        table.insert(positiveList,i)
    else
        table.insert(negativeList,i)
    end
end

trainData = {
    data = trainData.data,
    labels = trainData.label,
    size = function() return numAllTrainData end,
    addressList = {positiveList,negativeList}
}



-- import test data
if lfs.attributes(opt.testDataPath)['mode'] == 'file' then
 
    print("Import test data...")
    print('Importing:'..opt.testDataPath)
    myFile = hdf5.open(opt.testDataPath,'r')
    testData = myFile:read(''):all()
    myFile:close()
    labelNum = testData.label:size()[1]
    
    for i = 1,labelNum do
        if testData.label[i] == 0 then
            testData.label[i] = 1
        else 
            testData.label[i] = 2
        end
    end

    numAllTestData = labelNum

elseif lfs.attributes(opt.testDataPath)['mode'] == 'directory' then
    hdfList = getFileList(opt.testDataPath,'hdf5')
    dataList = {}
    for i=1, #hdfList do
        local myFile = hdf5.open(opt.testDataPath..'/'..hdfList[i])
        print('Importing:'..opt.testDataPath..'/'..hdfList[i])
        tempTestData = myFile:read(''):all()
        myFile:close()
        labelNum = tempTestData.label:size()[1]
    
        for i = 1,labelNum do
            if tempTestData.label[i] == 0 then
                tempTestData.label[i] = 1
            else 
                tempTestData.label[i] = 2
            end
        end
        table.insert(dataList, tempTestData)
    end
    numAllTestData = 0
    for i=1, #dataList do
        numAllTestData = numAllTestData + dataList[i]['data']:size()[1]
    end
    print(numAllTestData) 
    testData = {
        data=torch.Tensor(numAllTestData, opt.inputChannel, dataList[1]['data']:size()[3], dataList[1]['data']:size()[4]),
        label=torch.Tensor(numAllTestData),
        size = function() return numAllTestData end
    }
    k = 1
    for i=1, #dataList do
        for j=1, dataList[i]['data']:size()[1] do
            testData.data[k] = dataList[i]['data'][j]
            testData.label[k] = dataList[i]['label'][j]
            k = k + 1
        end
    end

else
    error("invalid dataset")
end

for t=1,opt.inputChannel do
    testData.data[{ {},t,{},{} }]:add(-dataMean[t])
    testData.data[{ {},t,{},{} }]:div(dataStd[t])
    print(t.."ch mean:"..dataMean[t].." stdv:"..dataStd[t])
end

testData = {
    data = testData.data, 
    labels = testData.label, 
    size = function() return numAllTestData end
}

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
print (opt.save)
accLogger.showPlot = false
errLogger.showPlot = false

function dataAugmentation(batchData)
    outputBatchData = torch.Tensor(batchData:size()[1],batchData:size()[2],batchData:size()[3],batchData:size()[4]):float()
    inputBatchSize = batchData:size()[1]
    hRand = torch.rand(inputBatchSize)
    vRand = torch.rand(inputBatchSize)
    for i=1, inputBatchSize do
        if hRand[i] > 0.5 then
            outputBatchData[i] = image.hflip(batchData[i]:float())
        else
            outputBatchData[i] = batchData[i]:float()
        end
        if vRand[i] > 0.5 then
            outputBatchData[i] = image.vflip(outputBatchData[i])
        end
    end
    return outputBatchData
end

-- training function
function train(dataset)
    -- epoch tracker
    epoch = epoch or 1
    
    model:training()

    -- local vars
    local time = sys.clock()
    local trainError = 0

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1,dataset:size(),opt.batchSize do
   
        
        local inputs = dataset.data[{{t,math.min(t+opt.batchSize-1,dataset:size())},{},{},{}}]
        local targets = dataset.labels[{{t,math.min(t+opt.batchSize-1,dataset:size())}}]
        
        -- data augmentation
        inputs = dataAugmentation(inputs):cuda()
        targets = targets:cuda()
        
        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- f is the average of all criterions
            local f = 0
            
              
            local output = model:forward(inputs)
            local err = criterion:forward(output, targets)
            f = f + err
            -- estimate df/dW
            local df_do = criterion:backward(output, targets)
            model:backward(inputs, df_do)
            for i = 1,math.min(opt.batchSize,dataset:size()-t+1) do
                -- update confusion
                confusion:add(output[i], targets[i])
            end
            

            trainError = trainError + f

            -- return f and df/dX
            return f,gradParameters
        end

        -- optimize on current mini-batch
        if opt.optimization == 'CG' then
            config = config or {maxIter = opt.maxIter}
            optim.cg(feval, parameters, config)

        elseif opt.optimization == 'LBFGS' then
            config = config or {learningRate = opt.learningRate,
                                maxIter = opt.maxIter,
                                nCorrection = 10}
            optim.lbfgs(feval, parameters, config)

        elseif opt.optimization == 'SGD' then
            config = config or {learningRate = opt.learningRate,
                                weightDecay = opt.weightDecay,
                                momentum = opt.momentum,
                                learningRateDecay = 5e-7}
            optim.sgd(feval, parameters, config)

        elseif opt.optimization == 'ASGD' then
            config = config or {eta0 = opt.learningRate,
                                t0 = nbTrainingPatches * opt.t0}
            _,_,average = optim.asgd(feval, parameters, config)
            
         elseif opt.optimization == 'ADAM' then
            config = config or {learningRate = opt.learningRate,
								learningRateDecay =  5e-7,
								beta1 = 0.9,
								beta2 = 0.999,
								epsilon = 1e-8,
								weightDecay = opt.weightDecay}
            optim.adam(feval, parameters, config)

        else
            error('unknown optimization method')
        end
    end

    -- train error
    trainError = trainError / math.floor(dataset:size()/opt.batchSize)

    -- time taken
    time = sys.clock() - time
    allTime = time
    time = time / dataset:size()
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
    print("<trainer> time to learn all samples = " .. (allTime*1000) .. 'ms')

    -- print confusion matrix
    print(confusion)
    local trainAccuracy = confusion.totalValid * 100
    confusion:zero()

    -- save/log current net
    if epoch%opt.saveInterval==0 then
        local filename = paths.concat(opt.save, 'cnn_ep'..epoch..'.net')
        os.execute('mkdir -p ' .. paths.dirname(filename))
        if paths.filep(filename) then
            os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
        end
        print('<trainer> saving network to '..filename)
        torch.save(filename, sanitize(model))
    end

    -- next epoch
    epoch = epoch + 1

    return trainAccuracy, trainError
end

-- training function using negative mining
function trainMod(dataset)
    -- epoch tracker
    epoch = epoch or 1
    
    model:training()
    

    
    -- local vars
    local time = sys.clock()
    local trainError = 0

    batch_rate = 1 + opt.negativeRatio

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    
    
    local dataSize = dataset.data:size()
    pIdx = torch.randperm(#dataset.addressList[1])
    nIdx = torch.randperm(#dataset.addressList[2])
    addressListmin=math.min(#dataset.addressList[1],#dataset.addressList[2])
    BatchData = {
        data = torch.Tensor(addressListmin*batch_rate,dataSize[2],dataSize[3],dataSize[4]),
        labels = torch.Tensor(addressListmin*batch_rate),
        size = function() return addressListmin*batch_rate end,
    }
	
    for i=1, addressListmin do
        BatchData.data[batch_rate*i-batch_rate+1] = dataset.data[dataset.addressList[1][pIdx[i]]]
        BatchData.labels[batch_rate*i-batch_rate+1] = dataset.labels[dataset.addressList[1][pIdx[i]]]
        for j=1, batch_rate-1 do
            BatchData.data[batch_rate*i-batch_rate+j+1] = dataset.data[dataset.addressList[2][nIdx[(batch_rate-1)*i+1-j]]]
            BatchData.labels[batch_rate*i-batch_rate+j+1] = dataset.labels[dataset.addressList[2][nIdx[(batch_rate-1)*i+1-j]]]
        end
    end
    
    
    for t = 1,BatchData:size(),opt.batchSize do

        
        local inputs = BatchData.data[{{t,math.min(t+opt.batchSize-1,BatchData:size())},{},{},{}}]
        local targets = BatchData.labels[{{t,math.min(t+opt.batchSize-1,BatchData:size())}}]
        
        -- data augmentation
        inputs = dataAugmentation(inputs):cuda()
        targets = targets:cuda()
        
        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- f is the average of all criterions
            local f = 0
            
           
            local output = model:forward(inputs)
            local err = criterion:forward(output, targets)
            f = f + err
            -- estimate df/dW
            local df_do = criterion:backward(output, targets)
            model:backward(inputs, df_do)
            for i = 1,math.min(opt.batchSize,BatchData:size()-t+1) do
                -- update confusion
                confusion:add(output[i], targets[i])
            end
            
            -- normalize gradients and f(X)

            trainError = trainError + f

            -- return f and df/dX
            return f,gradParameters
        end

        -- optimize on current mini-batch
        if opt.optimization == 'CG' then
            config = config or {maxIter = opt.maxIter}
            optim.cg(feval, parameters, config)

        elseif opt.optimization == 'LBFGS' then
            config = config or {learningRate = opt.learningRate,
                                maxIter = opt.maxIter,
                                nCorrection = 10}
            optim.lbfgs(feval, parameters, config)

        elseif opt.optimization == 'SGD' then
            config = config or {learningRate = opt.learningRate,
                                weightDecay = opt.weightDecay,
                                momentum = opt.momentum,
                                learningRateDecay = 5e-7}
            optim.sgd(feval, parameters, config)

        elseif opt.optimization == 'ASGD' then
            config = config or {eta0 = opt.learningRate,
                                t0 = nbTrainingPatches * opt.t0}
            _,_,average = optim.asgd(feval, parameters, config)
            
		elseif opt.optimization == 'ADAM' then
            config = config or {learningRate = opt.learningRate,
								learningRateDecay =  5e-7,
								beta1 = 0.9,
								beta2 = 0.999,
								epsilon = 1e-8,
								weightDecay = opt.weightDecay}
            optim.adam(feval, parameters, config)
        else
            error('unknown optimization method')
        end
    end

    -- train error
    trainError = trainError / math.floor(BatchData:size()/opt.batchSize)

    -- time taken
    time = sys.clock() - time
    allTime = time
    time = time / BatchData:size()
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
    print("<trainer> time to learn all samples = " .. (allTime*1000) .. 'ms')

    -- print confusion matrix
    print(confusion)
    local trainAccuracy = confusion.totalValid * 100
    confusion:zero()

    -- save/log current net
    if epoch%opt.saveInterval==0 then
        local filename = paths.concat(opt.save, 'cnn_nm_ep'..epoch..'.net')
        os.execute('mkdir -p ' .. paths.dirname(filename))
        if paths.filep(filename) then
            os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
        end
        print('<trainerNM> saving network to '..filename)
        torch.save(filename, sanitize(model))
    end

    -- next epoch
    epoch = epoch + 1

    return trainAccuracy, trainError
end

-- training function using negative mining
function trainNM(dataset)
    -- epoch tracker
    epoch = epoch or 1
    
    model:training()
    
    -- batchNM = 128
    
    -- local vars
    local time = sys.clock()
    local trainError = 0

    -- do one epoch
    print('<trainerNM> on training set:')
    print("<trainerNM> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSizeNM .. ']')
    
   
    -- get 128 negative data
    i = 1
    local dataSize = dataset.data:size()
    local negativeList = {}
    local nInputs = torch.Tensor(opt.negativeBatchSize,dataSize[2],dataSize[3],dataSize[4])


    local nTargets = torch.Tensor(opt.negativeBatchSize)    
   
    while i <= opt.negativeBatchSize do
        randN = (torch.rand(1)*(dataSize[1]-1)):floor()[1]+1
        j = 1
        count = 0
        while j <= #negativeList do
            if randN == negativeList[j] then
                count = count + 1
            end
            j = j + 1
        end

        if dataset.labels[randN] == 2 and count == 0 then
            table.insert(negativeList,randN)

            nInputs[i] = dataset.data[randN]:double()

            nTargets[i] = dataset.labels[randN]

            i = i + 1
        end
    end
    

    
    -- get all positive data
    positiveList = {}
    for i = 1, dataset:size() do
        if dataset.labels[i] == 1 then
            table.insert(positiveList,i)
        end
    end
    
    pIdx = torch.randperm(#positiveList)
    
    
    -- get loss of the negative data
    local output = model:forward(nInputs:cuda())
    --local err = criterion:forward(output, nTargets:cuda())
    local errList = torch.Tensor(opt.negativeBatchSize)
    
    for i = 1, opt.negativeBatchSize do
        errList[i] = criterion:forward(output[i], nTargets[i])
    end
    

    -- sort the loss in descending order

    y, idx = torch.sort(errList:double(),1,true)
    
    -- select 16 larger loss for training and create mini-batch
    local inputs = torch.Tensor(opt.batchSizeNM,dataSize[2],dataSize[3],dataSize[4]):cuda()
    local targets = torch.Tensor(opt.batchSizeNM):cuda()
    for i = 1, (opt.batchSizeNM/2) do

        inputs[i] = nInputs[idx[i]]
        targets[i] = nTargets[idx[i]]
    end
    for i=1, (opt.batchSizeNM/2) do
        inputs[i+(opt.batchSizeNM/2)] = dataset.data[positiveList[pIdx[i]]]
        targets[i+(opt.batchSizeNM/2)] = dataset.labels[positiveList[pIdx[i]]]
    end
    
    
    -- data augmentation
    inputs = dataAugmentation(inputs):cuda()
    targets:cuda()
    
    -- training
    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
        -- get new parameters
        if x ~= parameters then
            parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        -- f is the average of all criterions
        local f = 0
        
        local output = model:forward(inputs)
        local err = criterion:forward(output, targets)
        f = f + err
        -- estimate df/dW
        local df_do = criterion:backward(output, targets)
        model:backward(inputs, df_do)
        for i = 1,opt.batchSizeNM do
            -- update confusion
            confusion:add(output[i], targets[i])
        end
        
        trainError = trainError + f

        -- return f and df/dX
        return f,gradParameters
    end

    -- optimize on current mini-batch
    if opt.optimization == 'CG' then
        config = config or {maxIter = opt.maxIter}
        optim.cg(feval, parameters, config)

    elseif opt.optimization == 'LBFGS' then
        config = config or {learningRate = opt.learningRate,
                            maxIter = opt.maxIter,
                            nCorrection = 10}
        optim.lbfgs(feval, parameters, config)

    elseif opt.optimization == 'SGD' then
        config = config or {learningRate = opt.learningRate,
                            weightDecay = opt.weightDecay,
                            momentum = opt.momentum,
                            learningRateDecay = 5e-7}
        optim.sgd(feval, parameters, config)

    elseif opt.optimization == 'ASGD' then
        config = config or {eta0 = opt.learningRate,
                            t0 = nbTrainingPatches * opt.t0}
        _,_,average = optim.asgd(feval, parameters, config)
        
	elseif opt.optimization == 'ADAM' then
		config = config or {learningRate = opt.learningRate,
						learningRateDecay =  5e-7,
						beta1 = 0.9,
						beta2 = 0.999,
						epsilon = 1e-8,
						weightDecay = opt.weightDecay}
		optim.adam(feval, parameters, config)

    else
        error('unknown optimization method')
    end
    
    -- train error
    -- time taken
    time = sys.clock() - time
    allTime = time
    print("<trainerNM> time to learn all samples = " .. (allTime*1000) .. 'ms')

    -- print confusion matrix
    print(confusion)
    local trainAccuracy = confusion.totalValid * 100
    confusion:zero()

    -- save/log current net
    if epoch%opt.saveInterval==0 then
        local filename = paths.concat(opt.save, 'cnn_nm_ep'..epoch..'.net')
        os.execute('mkdir -p ' .. paths.dirname(filename))
        if paths.filep(filename) then
            os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
        end
        print('<trainerNM> saving network to '..filename)
        torch.save(filename, sanitize(model))
    end

    -- next epoch
    epoch = epoch + 1

    return trainAccuracy, trainError
end


-- test function
function testRapid(dataset)
    
    model:evaluate()
    
    -- local vars
    local testError = 0
    local time = sys.clock()

    -- averaged param use?
    if average then
        cachedparams = parameters:clone()
        parameters:copy(average)
    end
    
    --local testBatch = 30000
    local numErr = 0
    --print('1')
    for t = 1,dataset:size(),opt.testBatchSize do
        -- disp progress
        -- xlua.progress(t, dataset:size())
        local batchStart = t
        local batchEnd = math.min(t+opt.testBatchSize-1,dataset:size())
        local inputs = dataset.data[{{batchStart,batchEnd},{},{},{}}]:cuda()
        local targets = dataset.labels[{{batchStart,batchEnd}}]:cuda()

        local preds = model:forward(inputs)
        for i = 1, math.min(opt.testBatchSize,dataset:size()-t+1) do
            confusion:add(preds[i], targets[i])
        end

        err = criterion:forward(preds, targets)
        testError = testError + err
        numErr = numErr + 1
    end

    -- timing
    time = sys.clock() - time
    allTime = time
    time = time / dataset:size()
    print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')
    print("<trainer> time to test all samples = " .. (allTime*1000) .. 'ms')

    -- testing error estimation
    -- testError = testError / dataset:size()
    testError = testError / numErr

    -- print confusion matrix
    print(confusion)
    local testAccuracy = confusion.totalValid * 100
    confusion:zero()

    -- averaged param use?
    if average then
        -- restore parameters
        parameters:copy(cachedparams)
    end

    return testAccuracy, testError
end

n=1

while n <= opt.maxEpoch do
    -- train/test
    if n <= opt.epochStartNM then
        trainAcc, trainErr = trainMod(trainData)
        if n%opt.testInterval == 0 or n == 1 then
            testAcc,  testErr  = testRapid(testData)
        end
    else
        -- negative mining
        trainAcc, trainErr = trainNM(trainData)
        if n%(opt.testInterval*10) == 0 then
            testAcc,  testErr  = testRapid(testData)
        end
    end


    -- update logger
    accLogger:add{['% train accuracy'] = trainAcc, ['% test accuracy'] = testAcc}
    errLogger:add{['% train error']    = trainErr, ['% test error']    = testErr}

    -- plot logger
    accLogger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
    errLogger:style{['% train error']    = '-', ['% test error']    = '-'}
    accLogger:plot()
    errLogger:plot()

    collectgarbage()

    collectgarbage()

    n = n + 1
end
