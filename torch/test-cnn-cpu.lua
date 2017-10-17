
require 'nn'
require 'optim'
require 'image'
require 'hdf5'
require 'torch'
require 'lfs'

--
cmd = torch.CmdLine()
cmd:text()
cmd:text('CNN TEST')
cmd:text()
cmd:text('Options:')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-inputChannel', 7, 'nb of input channels')
cmd:option('-testBatchSize', 10000, 'test batch size')
cmd:option('-trainDataPath', '', 'path of hdf5 dataset for training')
cmd:option('-testDataPath', '', 'path of hdf5 dataset for test')
cmd:option('-threshold', 0.5, 'threshold of softmax')
cmd:option('-meanstd_file', 'mean_stdv.csv', 'training mean and stdv data')
cmd:text()
opt = cmd:parse(arg)
--

model = torch.load(opt.network):double()
criterion = nn.ClassNLLCriterion():double()

function split(str, delim)
    -- Eliminate bad cases...
    if string.find(str, delim) == nil then
        return { str }
    end

    local result = {}
    local pat = "(.-)" .. delim .. "()"
    local lastPos
    for part, pos in string.gfind(str, pat) do
        table.insert(result, part)
        lastPos = pos
    end
    table.insert(result, string.sub(str, lastPos))
    return result
end

file = io.open(opt.meanstd_file,"r")
dataMean={}
dataStd ={}
for line in file:lines() do
    local datas = split(line, ",")
    table.insert(dataMean,datas[1])
    table.insert(dataStd,datas[2])
end

function getFileList(dirPath, ext)
    list = {}
    for file in lfs.dir(dirPath) do
        if string.find(file, ext..'$') then
            table.insert(list,file)
        end
    end
    return list
end

-- this matrix records the current confusion across classes
classes = {'positive', 'negative'}
confusion = optim.ConfusionMatrix(classes)

probList = {}
labelList = {}


-- import test data
if lfs.attributes(opt.testDataPath)['mode'] == 'file' then
    -- import training data
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
    
    testIdx = testData.idx
    

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


-- test function
function testRapid(dataset)
    
    model:evaluate()
    
    -- local vars
    local testError = 0
    local time = sys.clock()

    if average then
        cachedparams = parameters:clone()
        parameters:copy(average)
    end


    local numErr = 0

    for t = 1,dataset:size(),opt.testBatchSize do

        local batchStart = t
        local batchEnd = math.min(t+opt.testBatchSize-1,dataset:size())
        local inputs = dataset.data[{{batchStart,batchEnd},{},{},{}}]
        local targets = dataset.labels[{{batchStart,batchEnd}}]

        local preds = model:forward(inputs)
        for i = 1, math.min(opt.testBatchSize,dataset:size()-t+1) do
            confusion:add(preds[i], targets[i])
            table.insert(probList, preds[i]:clone())
            table.insert(labelList, targets[i])
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
    testError = testError / numErr

    local testAccuracy = confusion.totalValid * 100
    confusion:zero()

    -- averaged param use?
    if average then
        -- restore parameters
        parameters:copy(cachedparams)
    end

    return testAccuracy, testError
end

testAcc, testErr = testRapid(testData)


falsePositiveList = {}
falseNegativeList = {}
truePositiveList = {}

splitTestDataPath = string.split(opt.testDataPath,'/')
hdf5Name = string.split(splitTestDataPath[#splitTestDataPath],'.hdf5')
csvDataPath = opt.threshold..'_'..hdf5Name[1]..'.csv'
lfs.mkdir(hdf5Name[1])

confusion = {{0,0},{0,0}}

for i=1, #probList do
    if labelList[i] == 1 then
        if probList[i][1] >= math.log(opt.threshold) then
            table.insert(truePositiveList,{probList[i][1],testIdx[i][1],testIdx[i][2]})
            confusion[1][1] = confusion[1][1] + 1
        else
            table.insert(falseNegativeList,{probList[i][1],testIdx[i][1],testIdx[i][2]})
            confusion[1][2] = confusion[1][2] + 1
        end
    else
        if probList[i][1] >= math.log(opt.threshold) then
            table.insert(falsePositiveList,{probList[i][1],testIdx[i][1],testIdx[i][2]})
            confusion[2][1] = confusion[2][1] + 1
        else
            confusion[2][2] = confusion[2][2] + 1
        end
    end
end

str = {'ConfusionMatrix:\n'}
table.insert(str, "[[     "..confusion[1][1].."     "..confusion[1][2].."]   \n")
table.insert(str, " [     "..confusion[2][1].."     "..confusion[2][2].."]]  \n")
print(table.concat(str))


f = io.open(hdf5Name[1].."/FP_"..csvDataPath,"w")
for i=1, #falsePositiveList do
    f:write(falsePositiveList[i][1]..","..falsePositiveList[i][2]..","..falsePositiveList[i][3].."\n")
end
f:close()

f = io.open(hdf5Name[1].."/TP_"..csvDataPath,"w")
for i=1, #truePositiveList do
    f:write(truePositiveList[i][1]..","..truePositiveList[i][2]..","..truePositiveList[i][3].."\n")
end
f:close()

f = io.open(hdf5Name[1].."/FN_"..csvDataPath,"w")
for i=1, #falseNegativeList do
    f:write(falseNegativeList[i][1]..","..falseNegativeList[i][2]..","..falseNegativeList[i][3].."\n")
end
f:close()
