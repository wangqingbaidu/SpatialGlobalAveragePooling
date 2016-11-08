local SpatialGlobalAveragePooling, Parent = torch.class('nn.SpatialGlobalAveragePooling', 'nn.Module')

function SpatialGlobalAveragePooling:__init()
  Parent.__init(self)
end

function SpatialGlobalAveragePooling:updateOutput(input)
  input.THNN.SpatialGlobalAveragePooling_updateOutput(
      input:cdata(),
      self.output:cdata())    
  return self.output
end

function SpatialGlobalAveragePooling:updateGradInput(input, gradOutput)
  if self.gradInput then
    input.THNN.SpatialGlobalAveragePooling_updateGradInput(
       input:cdata(),
       gradOutput:cdata(),
       self.gradInput:cdata()
       )       
  return self.gradInput
  end
end
