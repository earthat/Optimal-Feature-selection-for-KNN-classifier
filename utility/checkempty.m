function GrassHopperPositions = checkempty(GrassHopperPositions,dim)
while numel(find(GrassHopperPositions==0))==numel(GrassHopperPositions)
    GrassHopperPositions=round(rand(1,dim));
end
