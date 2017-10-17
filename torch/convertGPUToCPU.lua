

require 'cunn'
require 'optim'
require 'image'
require 'hdf5'
require 'cutorch'

--
cmd = torch.CmdLine()
cmd:text()
cmd:text('CNN TEST')
cmd:text()
cmd:text('Options:')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-outputNetwork', '', 'name of output network')
cmd:option('-gpu', 1, 'id of gpu to use')
cmd:text()
opt = cmd:parse(arg)
--


cutorch.setDevice(opt.gpu)

model = torch.load(opt.network)
networkCpu = string.split(opt.network, '.net')[1]..'_cpu.net'

if torch.typename(model.modules[1].weight) == 'torch.CudaTensor' then
    print('Converted network to CPU-mode.')
    if opt.outputNetwork == '' then
        torch.save(networkCpu, model:double())
    else
        torch.save(opt.outputNetwork, model:double())
    end
else
    print('Input network is already CPU-mode.')
end
